import math
from typing import Dict, List, Tuple
from libs.backend import Backend

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
        self.population_size = population_size

        print("#-- Spawning Training Actors with vLLM backends --#")
        self.inference_engines = [
            ray.remote(
                num_cpus=0,
                num_gpus=0,
                scheduling_strategy=strategy,
            )(MyLLM).remote(
                model=model_name,
                enforce_eager=False,
                worker_extension_cls="libs.vllm_utils.WorkerExtension",
                tensor_parallel_size=1,
                distributed_executor_backend="ray",
                dtype="float16",
                enable_prefix_caching=False,
                enable_lora=True,
                max_loras=max_loras,
                max_lora_rank=lora_rank,
                max_cpu_loras=self.population_size,
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

        _lora_tmp_root = tempfile.mkdtemp(prefix="vllm_loras_")
        self.lora_names = [f"lora_{i}" for i in range(population_size)]
        lora_paths = {}
        for name in self.lora_names:
            p = os.path.join(_lora_tmp_root, name)
            os.makedirs(p, exist_ok=True)
            peft_model.save_pretrained(p, safe_serialization=True)
            lora_paths[name] = p

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

        print(f"#-- Preloading {len(self.lora_names)} LoRA adapters into {len(self.inference_engines)} engine(s) --#")
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        dummy_prompt = [" "]
        for llm in self.inference_engines:
            for idx, name in enumerate(self.lora_names):
                lora_req = LoRARequest(name, idx + 1, lora_paths[name])
                ray.get(llm.generate.remote(dummy_prompt, sp, lora_request=lora_req, use_tqdm=False))
        for path in list(lora_paths.values()):
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(_lora_tmp_root, ignore_errors=True)

        self.report_lora_params(self.lora_names)
        print("#-- Backend Initialized --#")
        pass
    
    def evaluate_countdown_handle(self, llm, prompts):
        """Return a generation handle so we can schedule round-robin."""
        start = time.time()
        sampling_params = SamplingParams(
            temperature=0.0,
            seed=42,
            max_tokens=1024,
        )
                
        handle = llm.generate.remote(prompts, sampling_params, use_tqdm=self.use_tqdm)
        return handle, start

    def update(self, genome: Genome):
        """Update the model permanently with a genome as the source."""
        ray.get([llm.collective_rpc.remote("perturb_self_weights", args=(genome,)) for llm in self.inference_engines])

        if self.world_size > 1:
            ray.get([llm.collective_rpc.remote("perform_all_reduce_sync") for llm in self.inference_engines])

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[Dict[str, str]]]):
        """
        Generate outputs based on the genome and inputs.
        Updates the genomes with their new outputs.
        """
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

        gs = iter(genomes)
        inflight = {}

        for eng_idx, llm in enumerate(self.inference_engines):
            try:
                genome = next(gs)
            except StopIteration:
                break
            ray.get(
                llm.collective_rpc.remote(
                    "perturb_self_weights", args=(genome,)
                )
            )
            handle, start_ts = self.evaluate_countdown_handle(llm, prompts)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": eng_idx,
                "genome": genome,
                "start_ts": start_ts,
            }
        start_time = time.time()
        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            genome = meta["genome"]

            genome.latest_outputs = [o.outputs[0].text if hasattr(o, "outputs") and len(o.outputs) > 0 and hasattr(o.outputs[0], "text") else "" for o in outputs]

            llm = meta["engine"]
            # Remove the exploration perturbation
            ray.get(llm.collective_rpc.remote("restore_self_weights", args=(genome,)))
            try:
                genome = next(gs)
            except StopIteration:
                continue
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(genome,)))
            handle, start_ts = self.evaluate_countdown_handle(llm, prompts)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": meta["engine_idx"],
                "genome": genome,
                "start_ts": start_ts,
            }
            if self.time_self:
                end_time = time.time()
                print(f"#-- Genome outputs generated in {end_time - start_time:.2f} seconds --#")
                start_time = end_time
                
        if self.world_size > 1:
            ray.get([llm.collective_rpc.remote("perform_all_reduce_sync") for llm in self.inference_engines])
            
    def save_weights_to_disk(self, filepath: str):
        ray.get(self.inference_engines[0].collective_rpc.remote("save_weights_to_disk", args=(filepath,)))

    def report_lora_params(self, expected_adapter_names: List[str] = None):
        lora_params = ray.get(self.inference_engines[0].collective_rpc.remote("report_lora_params", args=(expected_adapter_names,)))
        return lora_params
