
import math
import os
import sys
import time
import gc
import shutil
import tempfile
import signal
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from propagate.backend.backend_abc import Backend
from propagate.genome import Genome
from propagate.optimizers import Optimizer
from propagate.trainer import SimpleTrainer

# -----------------------------------------------------------------------------
# Helper: Worker Extension to inspect/collect LoRA tensors on the generic worker
# This mirrors vllm_lorautils.py approach but adapted if necessary.
# We mostly rely on the fact that if LoRA is enabled, lora_manager exists.
# -----------------------------------------------------------------------------
class TPULoRAWorkerExtension:
    """
    Helper class to access LoRA weights within the vLLM worker on TPU.
    Since TPU execution might involve PyTorch-JAX interop (via PunicaWrapperTPU),
    we expect LoRA weights to be accessible via the lora_manager in PyTorch format.
    """
    def __init__(self, model_runner):
        self.model_runner = model_runner
        print("### WARNING: EXPERIMENTAL, WILL NOT WORK ###")

    def collect_lora_tensors(self, adapter_id: int) -> Dict[str, Any]:
        """
        Collects (lora_a, lora_b) tensors for a given adapter_id from the lora_manager.
        """
        lora_manager = self.model_runner.lora_manager
        if lora_manager is None:
            return {}
        
        adapter_manager = lora_manager._adapter_manager
        
        # vLLM internals: find the slot/id mapping
        try:
            slot = adapter_manager.lora_index_to_id.index(adapter_id)
        except ValueError:
            # adapter_id might not be loaded or mapped yet
            return {}

        found_tensors = {}

        for mod_name, mod in adapter_manager.modules.items():
            # In latest vLLM, these are usually StackedLoRALayer or similar
            # They hold lora_a_stacked, lora_b_stacked as lists or tensors
            
            a_sub_layers = getattr(mod, "lora_a_stacked", None)
            b_sub_layers = getattr(mod, "lora_b_stacked", None)

            if not isinstance(a_sub_layers, (list, tuple)) or not isinstance(b_sub_layers, (list, tuple)):
                continue

            is_packed = len(a_sub_layers) > 1

            for i in range(len(a_sub_layers)):
                a_stacked = a_sub_layers[i]
                b_stacked = b_sub_layers[i]

                if a_stacked is None or b_stacked is None:
                    continue
                
                # Check shapes
                if not (isinstance(a_stacked, torch.Tensor) and a_stacked.dim() > 0 and a_stacked.shape[0] > slot):
                    continue
                if not (isinstance(b_stacked, torch.Tensor) and b_stacked.dim() > 0 and b_stacked.shape[0] > slot):
                    continue

                # These are PyTorch tensors. 
                # On TPU backend using torchax/punica, these are likely on CPU or XLA device.
                # We modify them in-place.
                lora_a = a_stacked[slot]
                lora_b = b_stacked[slot]
                
                key_name = mod_name
                if is_packed:
                    key_name = f"{mod_name}#sub{i}"
                
                found_tensors[key_name] = (lora_a, lora_b)

        return found_tensors

class VllMTPUTPLoRABackend(Backend):
    def __init__(self, model_name: str, sampler: SamplingParams, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False, GPU_FRACTION_VLLM_WORKER: float = 0.6, tensor_parallel_size: int = 8, lora_rank: int = 8, lora_perturb_target: str = "b-", init_lora_weights: str = True, lora_model_source: str = None, norm_scale_update: bool = True, repeat_tokens_buffer_count: int = 20, repeat_times_kill: int = 15, rep_check_every: int = 100, repeat_tokens_begin_scan_count: int = 500, repeat_tokens_lookback_length: int = 500):
        
        super().__init__(backend_name="vLLM TPU TP LoRA Backend", model_name=model_name, NUM_GPUS=tensor_parallel_size, CPUS_PER_GPU=1, GPU_FRACTION_VLLM_WORKER=GPU_FRACTION_VLLM_WORKER, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
        self.lora_model_source = lora_model_source if lora_model_source else model_name        
        self.lora_rank = lora_rank
        self.lora_perturb_target = lora_perturb_target
        self.init_lora_weights = init_lora_weights
        self.norm_scale_update = norm_scale_update
        self.tensor_parallel_size = tensor_parallel_size
        if "a" not in lora_perturb_target.lower() and "b" not in lora_perturb_target.lower():
            raise ValueError(f"Invalid lora_perturb_target: {lora_perturb_target}. Must be 'a' or 'b' or 'a-' or 'b-' or 'ab'.")

        self.repeat_tokens_buffer_count = repeat_tokens_buffer_count
        self.repeat_times_kill = repeat_times_kill
        self.rep_check_every = rep_check_every
        self.repeat_tokens_begin_scan_count = repeat_tokens_begin_scan_count
        self.repeat_tokens_lookback_length = repeat_tokens_lookback_length
        
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.pop("TPU_MULTIHOST_BACKEND", None)

        self._norm_cache = {}

    def startup(self, trainer: SimpleTrainer):
        """Initialize vLLM TPU engine with LoRA support."""
        print(f"#-- Initializing vLLM TPU LoRA Backend (TP={self.tensor_parallel_size}, Rank={self.lora_rank}) --#")

        self.population_size = trainer.population_size
        if trainer.mirror:
            self.population_size *= 2
        
        max_loras_per_worker = self.population_size

        print(f"#-- Starting LLM with enable_lora=True, max_loras={max_loras_per_worker} --#")
        self.llm = LLM(
            model=self.model_name, 
            tensor_parallel_size=self.tensor_parallel_size, 
            trust_remote_code=True, 
            dtype="bfloat16", 
            max_model_len=self.max_model_len, 
            gpu_memory_utilization=self.GPU_FRACTION_VLLM_WORKER,
            enable_lora=True,
            max_loras=max_loras_per_worker,
            max_lora_rank=max(self.lora_rank, 8),
            max_cpu_loras=1000,
        )

        print("#-- Creating LoRA adapters (PEFT) --#")
        
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        print(f"Loading base model structure from {self.lora_model_source} for LoRA init...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.lora_model_source,
            torch_dtype=torch.float16,
            device_map="cpu", 
        )
        
        if self.init_lora_weights == "zero":
            print(f"#-- Initializing rank {self.lora_rank} LoRA adapters with zeros --#")
            lora_cfg = LoraConfig(
                r=self.lora_rank,
                lora_alpha=1, 
                target_modules=target_modules,
                use_rslora=True,
                init_lora_weights=False
            )
            peft_model = get_peft_model(base_model, lora_cfg)
            for name, param in peft_model.named_parameters():
                if "lora_" in name:
                    param.data.zero_()
        else:
            print(f"#-- Initializing rank {self.lora_rank} LoRA adapters with {self.init_lora_weights} --#")
            lora_cfg = LoraConfig(
                r=self.lora_rank,
                lora_alpha=1,
                target_modules=default_target_modules,
                init_lora_weights=self.init_lora_weights,
                use_rslora=True
            )
            peft_model = get_peft_model(base_model, lora_cfg)

        self._lora_tmp_root = tempfile.mkdtemp(prefix="vllm_tpu_loras_")

        lora_names = [f"lora_{i}" for i in range(self.population_size)]
        self.lora_paths = {}
        for name in lora_names:
            p = os.path.join(self._lora_tmp_root, name)
            os.makedirs(p, exist_ok=True)
            peft_model.save_pretrained(p, safe_serialization=True)
            self.lora_paths[name] = p

        del peft_model
        del base_model
        gc.collect()

        print(f"#-- Preloading {len(self.lora_paths)} LoRA adapters --#")
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        dummy_prompt = " "
        for idx, name in enumerate(lora_names):
            lora_req = LoRARequest(name, idx + 1, self.lora_paths[name])
            self.llm.generate([dummy_prompt], sp, lora_request=lora_req, use_tqdm=False)
        print("#-- vLLM TPU LoRA Backend Initialized Successfully --#")

    def _get_worker_extension(self):
        """
        Retreive the worker extension or access to model runner.
        In single process (no Ray), we can access llm engine directly.
        """
        driver_worker = self.llm.llm_engine.model_executor.driver_worker
        return TPULoRAWorkerExtension(driver_worker.model_runner)

    def update(self, optimizer: Optimizer):
        """Update the LoRA weights permanently with a genome as the source."""
        print("#-- Updating LoRA Model Weights (TPU) --#")
        ext = self._get_worker_extension()
        eps = 1e-5

        if not hasattr(self, 'optimizer_state_per_adapter'):
            self.optimizer_state_per_adapter = {}

        # Scan active adapters
        # We use lora_paths keys to know what we should have
        for name in self.lora_paths.keys():
            try:
                lid = int(name.split("_")[-1]) + 1
            except:
                continue
            
            if name not in self.optimizer_state_per_adapter:
                self.optimizer_state_per_adapter[name] = {}
            
            start_time = time.time()
            weights = ext.collect_lora_tensors(lid)
            if not weights:
                continue
            
            state = self.optimizer_state_per_adapter[name]
            rand_counter = 0
            
            for mod_name, (lora_a, lora_b) in sorted(weights.items()):
                layer_norm_scale = 1.0
                if self.norm_scale_update:
                    norm_a = torch.norm(lora_a)
                    norm_b = torch.norm(lora_b)
                    combined_norm = torch.sqrt(norm_a.pow(2) + norm_b.pow(2))
                    layer_norm_scale = 1.0 / (combined_norm + eps)
                
                if "a" in self.lora_perturb_target.lower():
                    optimizer.step_update(lora_a.data, rand_counter, (name, mod_name, "a"), lr_scalar=float(layer_norm_scale), state=state)
                    rand_counter += 1
                if "b" in self.lora_perturb_target.lower():
                    optimizer.step_update(lora_b.data, rand_counter, (name, mod_name, "b"), lr_scalar=float(layer_norm_scale), state=state)
                    rand_counter += 1
        if self.lora_perturb_target == "a-":
            self.lora_perturb_target = "b-"
        elif self.lora_perturb_target == "b-":
            self.lora_perturb_target = "a-"

    def _perturb_weights(self, genomes: List[Genome], mode: str = "perturb"):
        """
        Perturb (or restore) weights for multiple genomes/adapters.
        """
        ext = self._get_worker_extension()
        
        if len(genomes) > len(self.lora_paths):
             print(f"Warning: More genomes ({len(genomes)}) than available adapters ({len(self.lora_paths)})")
        
        # Clear cache at the start of a perturb cycle to ensure we calculate fresh norms for the base weights
        if mode == "perturb":
            self._norm_cache = {}
        
        eps = 1e-5
        
        for i, genome in enumerate(genomes):
            # Map genome to adapter
            adapter_name = f"lora_{i}"
            if adapter_name not in self.lora_paths:
                continue
                
            try:
                lid = int(adapter_name.split("_")[-1]) + 1
            except (ValueError, IndexError):
                raise ValueError(f"Could not parse LoRA ID from adapter name: {adapter_name}")
                
            weights = ext.collect_lora_tensors(lid)
            if not weights:
                continue
                
            # Perturb
            for seed, weight in zip(genome.seeds, genome.perturb_scales):
                rand_counter = 0
                for mod_name, (lora_a, lora_b) in sorted(weights.items()):     
                    cache_key = (lid, mod_name)
                    layer_norm_scale = 1.0

                    if mode == "perturb":
                        if cache_key in self._norm_cache:
                             layer_norm_scale = self._norm_cache[cache_key]
                        else:
                            if self.norm_scale_update:               
                                norm_a = torch.norm(lora_a)
                                norm_b = torch.norm(lora_b)
                                combined_norm = torch.sqrt(norm_a.pow(2) + norm_b.pow(2))
                                layer_norm_scale = 1.0 / (combined_norm + eps)
                            self._norm_cache[cache_key] = layer_norm_scale
                    else:
                        if cache_key in self._norm_cache:
                            layer_norm_scale = self._norm_cache[cache_key]
                        else:
                            if self.norm_scale_update:
                                norm_a = torch.norm(lora_a)
                                norm_b = torch.norm(lora_b)
                                combined_norm = torch.sqrt(norm_a.pow(2) + norm_b.pow(2))
                                layer_norm_scale = 1.0 / (combined_norm + eps)
                            else:
                                layer_norm_scale = 1.0

                    op_weight = weight
                    if mode == "restore":
                         op_weight = -weight
                    
                    if "a" in self.lora_perturb_target.lower():
                        gen = torch.Generator(device=lora_a.device)
                        gen.manual_seed(int(seed) + rand_counter)
                        rand_counter += 1
                        noise = torch.randn(lora_a.shape, generator=gen, device=lora_a.device, dtype=lora_a.dtype)
                        lora_a.data.add_(noise, alpha=float(op_weight * layer_norm_scale))
                    
                    if "b" in self.lora_perturb_target.lower():
                        gen = torch.Generator(device=lora_b.device)
                        gen.manual_seed(int(seed) + rand_counter)
                        rand_counter += 1
                        noise = torch.randn(lora_b.shape, generator=gen, device=lora_b.device, dtype=lora_b.dtype)
                        lora_b.data.add_(noise, alpha=float(op_weight * layer_norm_scale))

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        if len(genomes) > self.population_size:
            raise ValueError("Number of genomes must be less than or equal to population size.")
            
        prompts = []
        for idk, i in enumerate(inputs):
            prompt_genome = []
            input_genome_content = []
            for j in i:
                input_genome_content.append(j[-1]['content'])
                s = self.llm.get_tokenizer().apply_chat_template(j, tokenize=False, add_generation_prompt=True)
                if suffix is not None:
                    s = s + suffix
                prompt_genome.append(s)
            prompts.append(prompt_genome)
            genomes[idk].latest_inputs = input_genome_content

        start_time_all = time.time()
        
        # 1. Perturb
        if self.time_self:
            print("#-- Perturbing Weights --#")
        self._perturb_weights(genomes, mode="perturb")
        
        all_prompts_flat = []
        all_lora_reqs_flat = []
        
        # We need to map back results
        # Structure: genomes -> prompts
        
        for g_idx, (genome, genome_prompts) in enumerate(zip(genomes, prompts)):
            adapter_name = f"lora_{g_idx}"
            if adapter_name not in self.lora_paths:
                raise RuntimeError(f"Adapter {adapter_name} not found in lora_paths. Available: {list(self.lora_paths.keys())}")
                
            path = self.lora_paths[adapter_name]
            try:
                lid = int(adapter_name.split("_")[-1]) + 1
            except (ValueError, IndexError):
                raise ValueError(f"Could not parse LoRA ID from adapter name: {adapter_name}")
            
            lora_req = LoRARequest(adapter_name, lid, path)
            
            for p in genome_prompts:
                all_prompts_flat.append(p)
                all_lora_reqs_flat.append(lora_req)
        
        if self.time_self:
            print(f"#-- Generating {len(all_prompts_flat)} requests --#")
            
        outputs = self.llm.generate(
            all_prompts_flat, 
            sampling_params=self.sampler, 
            lora_request=all_lora_reqs_flat, 
            use_tqdm=self.use_tqdm
        )
        
        # 3. Restore
        if self.time_self:
            print("#-- Restoring Weights --#")
        self._perturb_weights(genomes, mode="restore")
        
        # 4. Map outputs back
        # outputs list corresponds to all_prompts_flat
        curr = 0
        for i, (genome, genome_prompts) in enumerate(zip(genomes, prompts)):
            count = len(genome_prompts)
            genome_outs = outputs[curr : curr + count]
            genome.latest_outputs = [o.outputs[0].text for o in genome_outs]
            curr += count

        if self.time_self:
            print(f"#-- All genomes generated in {time.time() - start_time_all:.2f}s --#")
    
    def save_weights_to_disk(self, filepath: str):
        print("WARNING: Save weights not fully implemented for LoRA TPU backend. Saving base model usually.")
        # We could save the LoRA adapters from their temp paths.
        pass

    def __del__(self):
        # Cleanup temp dir
        if hasattr(self, '_lora_tmp_root') and self._lora_tmp_root and os.path.exists(self._lora_tmp_root):
            shutil.rmtree(self._lora_tmp_root, ignore_errors=True)
