# Note: This script is largely the work of https://github.com/dibbla and has been modified from the repo at https://github.com/VsonicV/es-fine-tuning-paper/tree/main
import math
from typing import Dict, List
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

class VLLMBackend(Backend):
    tokenizer: AutoTokenizer
    sampler: SamplingParams

    def __init__(self, model_name: str, NUM_GPUS: int, CPUS_PER_GPU: int, GPU_FRACTION_VLLM_WORKER: float, Sampler: SamplingParams, use_tqdm: bool = False, time_self: bool = False):
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
                enable_prefix_caching=False
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

