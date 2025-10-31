import math
from typing import Dict, List, Tuple
from uuid import uuid4
from libs.backend.backend_abc import Backend

import signal
import sys
import os
import ray
import torch
import time 
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

from libs.genome import Genome

from ray.util import collective
from torch.distributed import ReduceOp

from peft import LoraConfig, get_peft_model
from vllm.lora.request import LoRARequest
import shutil
import tempfile
import gc

class VLLMBackendLoRA(Backend):
    tokenizer: AutoTokenizer
    sampler: SamplingParams

    def __init__(self, model_name: str, NUM_GPUS: int, CPUS_PER_GPU: int, GPU_FRACTION_VLLM_WORKER: float, Sampler: SamplingParams, use_tqdm: bool = False, time_self: bool = False, population_size: int = 28, lora_rank: int = 16, max_loras: int = 7):
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        os.environ.pop("RAY_ADDRESS", None)
        os.environ.pop("RAY_HEAD_IP", None)
        os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)

        #--------------------------------------------------------#
        #                CUSTOM CLASSES DEFINITION               #
        #--------------------------------------------------------#
        class MyLLM(LLM):
            def __init__(self, *args, **kwargs):
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(GPU_FRACTION_VLLM_WORKER)
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                super().__init__(*args, **kwargs)

            def generate_multi_lora(self, prompts, sampling_params, lora_specs, use_tqdm=False):
                from uuid import uuid4

                try:
                    from tqdm.auto import tqdm as _tqdm
                    _tqdm_available = True
                except Exception:
                    _tqdm_available = False
                    _tqdm = None

                engine = self.llm_engine

                # Map request_id -> (genome_idx, prompt_idx)
                req_to_idx = {}
                # Results buffer
                results = [[None for _ in range(len(prompts))] for _ in range(len(lora_specs))]

                # Track per-request generated token counts to compute accurate tok/s
                prev_gen_tokens = {}  # rid -> last seen generated token count
                total_generated_tokens = 0
                completed = 0

                # Enqueue all requests
                for g_idx, spec in enumerate(lora_specs):
                    lora_req = LoRARequest(spec["name"], spec["id"], spec["path"])
                    for p_idx, prompt in enumerate(prompts):
                        rid = f"{g_idx}:{p_idx}:{uuid4().hex}"
                        engine.add_request(
                            request_id=rid,
                            prompt=prompt,
                            params=sampling_params, 
                            lora_request=lora_req,
                        )
                        req_to_idx[rid] = (g_idx, p_idx)
                        prev_gen_tokens[rid] = 0

                remaining = len(req_to_idx)
                start_t = time.perf_counter()

                pbar = None
                if use_tqdm and _tqdm_available:
                    pbar = _tqdm(total=remaining, desc="vLLM multi-LoRA", leave=True)

                # Drive engine until all requests finish
                while remaining > 0:
                    request_outputs = engine.step()

                    step_new_tokens = 0

                    for ro in request_outputs:
                        # Accumulate generated tokens incrementally for throughput stats
                        out = ro.outputs[0] if (hasattr(ro, "outputs") and ro.outputs) else None
                        if out is not None and hasattr(out, "token_ids"):
                            curr = len(out.token_ids)  # generated tokens so far for this request
                            prev = prev_gen_tokens.get(ro.request_id, 0)
                            if curr > prev:
                                step_new_tokens += (curr - prev)
                                prev_gen_tokens[ro.request_id] = curr

                        if getattr(ro, "finished", False):
                            g_idx, p_idx = req_to_idx[ro.request_id]
                            text = out.text if (out is not None and hasattr(out, "text")) else ""
                            results[g_idx][p_idx] = text

                            remaining -= 1
                            completed += 1
                            if pbar is not None:
                                pbar.update(1)

                    # Update running throughput stats
                    if step_new_tokens > 0:
                        total_generated_tokens += step_new_tokens
                        if pbar is not None:
                            elapsed = max(1e-6, time.perf_counter() - start_t)
                            tok_s = total_generated_tokens / elapsed
                            req_s = completed / elapsed
                            pbar.set_postfix_str(f"tok/s={tok_s:.1f}, req/s={req_s:.2f}, gen_tok={total_generated_tokens}")

                if pbar is not None:
                    # Final stats update
                    elapsed = max(1e-6, time.perf_counter() - start_t)
                    tok_s = total_generated_tokens / elapsed
                    req_s = completed / elapsed
                    pbar.set_postfix_str(f"tok/s={tok_s:.1f}, req/s={req_s:.2f}, gen_tok={total_generated_tokens}")
                    pbar.close()

                return results
        #-----------------------------------------------------#

        print("#-- Initializing Backend [VLLMBackendTP] --#")
        print(f"#-- GPUS: {NUM_GPUS}, CPUS per GPU: {CPUS_PER_GPU}, GPU Fraction VLLM Worker: {GPU_FRACTION_VLLM_WORKER} --#")
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

        pgs = [placement_group([{"GPU": 1, "CPU": CPUS_PER_GPU}]) for _ in range(NUM_GPUS)]
        ray.get([pg.ready() for pg in pgs])

        strategies = [
            PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )
            for pg in pgs
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampler = Sampler
        self.use_tqdm = use_tqdm
        self.time_self = time_self
        self.world_size = NUM_GPUS

        print("#-- Spawning Training Actors with vLLM backends --#")
        self.inference_engines = [
            ray.remote(
                num_cpus=0,
                num_gpus=0,
                scheduling_strategy=strategy,
            )(MyLLM).remote(
                model=model_name,
                enforce_eager=False,
                worker_extension_cls="libs.backend.vllm_lorautils.WorkerExtension",
                tensor_parallel_size=1,
                #distributed_executor_backend="ray",
                dtype="float16",
                enable_prefix_caching=False,
                gpu_memory_utilization=GPU_FRACTION_VLLM_WORKER,
                enable_lora=True,
                max_loras=max_loras,
                max_lora_rank=lora_rank,
                max_cpu_loras=1000,
            )
            for strategy in strategies
        ]

        if self.world_size > 1:
            print("#-- Initializing Ray Collective group for GPU sync --#")
            ray.get([llm.collective_rpc.remote("init_collective_group", args=(self.world_size, rank,)) for rank, llm in enumerate(self.inference_engines)])
        else:
            print("#-- Skipping collective group (1 GPU) --#")

        def cleanup():
            if self.world_size > 1:
                try:
                    ray.get([llm.collective_rpc.remote("destroy_collective_group") for llm in self.inference_engines])
                    print("\n#-- Collective group destroyed --#")
                except Exception as e:
                    print(f"#-- Error destroying collective group: {e} --#")            
            for llm in self.inference_engines:
                try:
                    ray.kill(llm)
                except Exception:
                    pass
            for pg in pgs:
                try:
                    remove_placement_group(pg)
                except Exception:
                    pass

        def sig_handler(sig, frame):
            cleanup()
            try:
                if hasattr(self, "_lora_tmp_root") and os.path.exists(self._lora_tmp_root):
                    shutil.rmtree(self._lora_tmp_root, ignore_errors=True)
            except Exception as e:
                print(f"#-- Error cleaning up temp LoRA dir: {e} --#")
            sys.exit(0)

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        print("#-- Creating LoRA adapters --#")
        default_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            target_modules=default_target_modules,
        )
        peft_model = get_peft_model(base_model, lora_cfg)
        with torch.no_grad():
          for name, param in peft_model.named_parameters():
              if "lora_" in name:
                  param.data.zero_()

        self._lora_tmp_root = tempfile.mkdtemp(prefix="vllm_loras_")

        max_loras_per_worker = math.ceil(population_size / NUM_GPUS)
        num_adapters_to_create = max(max_loras, max_loras_per_worker)

        lora_names = [f"lora_{i}" for i in range(num_adapters_to_create)]
        self.lora_paths = {} 
        for name in lora_names:
            p = os.path.join(self._lora_tmp_root, name) 
            os.makedirs(p, exist_ok=True)
            peft_model.save_pretrained(p, safe_serialization=True)
            self.lora_paths[name] = p

        # 3) Free the HF objects from memory
        try:
            del peft_model
        except Exception:
            pass
        try:
            del base_model
        except Exception:
            pass
        gc.collect()

        print(f"#-- Preloading {len(lora_names)} LoRA adapters into {len(self.inference_engines)} engine(s) --#")
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        dummy_prompt = [" "]
        for llm in self.inference_engines:
            for idx, name in enumerate(lora_names):
                lora_req = LoRARequest(name, idx + 1, self.lora_paths[name])
                ray.get(llm.generate.remote(dummy_prompt, sp, lora_request=lora_req, use_tqdm=False))

        print("#-- Locating LoRA adapters --#")
        ray.get(self.inference_engines[0].collective_rpc.remote("self_report_lora_params_sanity_check"))
        print("#-- Backend Initialized --#")
        pass

    def __del__(self):
        if hasattr(self, "_lora_tmp_root") and os.path.exists(self._lora_tmp_root):
            shutil.rmtree(self._lora_tmp_root, ignore_errors=True)

    def update(self, genome: Genome):
        """Update the model permanently with a genome as the source."""
        ray.get([llm.collective_rpc.remote("perturb_self_weights_all", args=(genome,)) for llm in self.inference_engines])

        if self.world_size > 1:
            ray.get([llm.collective_rpc.remote("perform_all_reduce_sync") for llm in self.inference_engines])

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[Dict[str, str]]]):
        prompts = []
        for i in inputs:
            s = self.tokenizer.apply_chat_template(
                i,
                tokenize=False,
                add_generation_prompt=True
            )
            if suffix is not None:
                s = s + suffix
            prompts.append(s)

        genome_chunks = np.array_split(genomes, self.world_size)

        if self.time_self:
            start_time = time.time()
            
        perturb_handles = []
        for eng_idx, llm in enumerate(self.inference_engines):
            my_genomes = genome_chunks[eng_idx]
            
            if len(my_genomes) == 0:
                continue
                
            h = llm.collective_rpc.remote(
                "perturb_self_weights_multi", args=(my_genomes.tolist(),)
            )
            perturb_handles.append(h)
            
        ray.get(perturb_handles)
        if self.time_self:
            print(f"#-- All adapters perturbed in {time.time() - start_time:.2f}s --#")
            gen_start_time = time.time()

        all_gen_handles = []
        genome_chunks_kept = []  # to map results back

        for eng_idx, llm in enumerate(self.inference_engines):
            my_genomes = genome_chunks[eng_idx]
            if len(my_genomes) == 0:
                continue

            lora_specs = []
            for local_idx, _genome in enumerate(my_genomes):
                local_lora_id = local_idx + 1
                local_lora_name = f"lora_{local_idx}"
                try:
                    lora_path = self.lora_paths[local_lora_name]
                except KeyError:
                    print(f"Error: No preloaded path found for {local_lora_name}")
                    print(f"Available paths: {list(self.lora_paths.keys())}")
                    raise
                lora_specs.append({"name": local_lora_name, "id": local_lora_id, "path": lora_path})

            h = llm.generate_multi_lora.remote(
                prompts,
                self.sampler,
                lora_specs,
                self.use_tqdm
            )
            all_gen_handles.append(h)
            genome_chunks_kept.append(my_genomes)

        all_outputs = ray.get(all_gen_handles)
        
        if self.time_self:
            end_time = time.time()
            print(f"#-- All genome outputs generated in {end_time - gen_start_time:.2f} seconds --#")

        for chunk_results, genomes_in_chunk in zip(all_outputs, genome_chunks_kept):
            for genome, outputs_for_genome in zip(genomes_in_chunk, chunk_results):
                genome.latest_outputs = outputs_for_genome
        
        restore_handles = []
        for eng_idx, llm in enumerate(self.inference_engines):
            my_genomes = genome_chunks[eng_idx]
            
            if len(my_genomes) > 0:
                h = llm.collective_rpc.remote(
                    "restore_self_weights_multi", args=(my_genomes.tolist(),)
                )
                restore_handles.append(h)
        
        ray.get(restore_handles)
        if self.time_self:
            print(f"#-- All adapters restored in {time.time() - end_time:.2f}s --#")
            
    def save_weights_to_disk(self, filepath: str):
        ray.get(self.inference_engines[0].collective_rpc.remote("save_weights_to_disk", args=(filepath,)))