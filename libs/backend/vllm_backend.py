# Note: This script is largely the work of https://github.com/dibbla and has been modified from the repo at https://github.com/VsonicV/es-fine-tuning-paper/tree/main
from typing import Dict, List
from libs.backend.backend_abc import Backend

import signal
import sys
import os
import ray
import time 
from transformers import AutoTokenizer
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_ip, get_open_port

from libs.genome import Genome
from libs.optimizers import Optimizer

class VLLMBackend(Backend):
    def __init__(self, model_name: str, NUM_GPUS: int, CPUS_PER_GPU: int, GPU_FRACTION_VLLM_WORKER: float, sampler: SamplingParams, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False):
        super().__init__(backend_name="Standard vLLM Backend", model_name=model_name, NUM_GPUS=NUM_GPUS, CPUS_PER_GPU=CPUS_PER_GPU, GPU_FRACTION_VLLM_WORKER=GPU_FRACTION_VLLM_WORKER, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)

    def startup(self, trainer=None):
        """
        Initializes the vLLM backend with Ray actors and placement groups.
        """
        # Set environment variables for vLLM and Ray
        os.environ.pop("RAY_ADDRESS", None)
        os.environ.pop("RAY_HEAD_IP", None)
        os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
        pass_gpu_fraction = str(self.GPU_FRACTION_VLLM_WORKER)
        #--------------------------------------------------------#
        #                CUSTOM CLASSES DEFINITION               #
        #--------------------------------------------------------#
        class MyLLM(LLM):
            def __init__(self, *args, **kwargs):
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                os.environ["VLLM_RAY_PER_WORKER_GPUS"] = pass_gpu_fraction
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                super().__init__(*args, **kwargs)
        #-----------------------------------------------------#

        print(f"#-- Initializing Backend {self.backend_name} --#")
        print(f"#-- GPUS: {self.NUM_GPUS}, CPUS per GPU: {self.CPUS_PER_GPU}, GPU Fraction VLLM Worker: {self.GPU_FRACTION_VLLM_WORKER} --#")

        print("#-- Spawning Training Actors with vLLM backends --#")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)
        pgs = [placement_group([{"GPU": 1, "CPU": self.CPUS_PER_GPU}]) for _ in range(self.NUM_GPUS)]
        ray.get([pg.ready() for pg in pgs])
        strategies = [PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0) for pg in pgs]

        self.inference_engines = [
            ray.remote(
                num_cpus=0,
                num_gpus=0,
                scheduling_strategy=strategy,
            )(MyLLM).remote(
                model=self.model_name,
                tensor_parallel_size=1,
                distributed_executor_backend="ray",
                worker_extension_cls="libs.backend.vllm_utils.WorkerExtension",
                dtype="float16",
                enable_prefix_caching=False,
                enforce_eager=False,
                gpu_memory_utilization=self.GPU_FRACTION_VLLM_WORKER,
                max_model_len=self.max_model_len
            )
            for strategy in strategies
        ]

        if self.NUM_GPUS > 1:
            print("#-- Initializing Ray Collective group for GPU sync --#")
            master_address = get_ip()
            master_port = get_open_port()
            ray.get([self.inference_engines[i].collective_rpc.remote("init_inter_engine_group", args=(master_address, master_port, i, self.NUM_GPUS)) for i in range(self.NUM_GPUS)])
        else:
            print("#-- Skipping collective group (1 GPU) --#")

        def cleanup():  
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
            ray.shutdown()

        def sig_handler(sig, frame):
            cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

    def evaluate_countdown_handle(self, llm, prompts):
        """Return a generation handle so we can schedule round-robin."""
        start = time.time()
        handle = llm.generate.remote(prompts, self.sampler, use_tqdm=self.use_tqdm)
        return handle, start

    def update(self, optimizer: Optimizer):
        """Update the model permanently with a genome as the source."""
        ray.get([llm.collective_rpc.remote("update_weights", args=(optimizer,)) for llm in self.inference_engines])

        if self.NUM_GPUS > 1:
            ray.get([llm.collective_rpc.remote("broadcast_all_weights", args=(0,)) for llm in self.inference_engines])

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """
        Generate outputs based on the genome and inputs.
        Updates the genomes with their new outputs.
        """
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        prompts = []
        for i in inputs:
            prompt_genome = []
            for j in i:
                s = self.tokenizer.apply_chat_template(j, tokenize=False, add_generation_prompt=True)
                if suffix is not None:
                    s = s + suffix
                prompt_genome.append(s)
            prompts.append(prompt_genome)

        gs = iter(genomes)
        ds = iter(prompts)
        inflight = {}

        for eng_idx, llm in enumerate(self.inference_engines):
            try:
                genome = next(gs)
                prompt_set = next(ds)
            except StopIteration:
                break
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(genome,)))
            handle, start_ts = self.evaluate_countdown_handle(llm, prompt_set)
            inflight[handle] = {"engine": llm, "engine_idx": eng_idx, "genome": genome, "start_ts": start_ts}

        start_time = time.time()
        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            genome = meta["genome"]

            genome.latest_outputs = [o.outputs[0].text for o in outputs]

            llm = meta["engine"]
            ray.get(llm.collective_rpc.remote("restore_self_weights", args=(genome,)))
            try:
                genome = next(gs)
                prompts_set = next(ds)
            except StopIteration:
                continue
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(genome,)))
            handle, start_ts = self.evaluate_countdown_handle(llm, prompts_set)
            inflight[handle] = {"engine": llm, "engine_idx": meta["engine_idx"], "genome": genome, "start_ts": start_ts}
            if self.time_self:
                end_time = time.time()
                print(f"#-- Genome outputs generated in {end_time - start_time:.2f} seconds --#")
                start_time = end_time
                
        if self.NUM_GPUS > 1:
            ray.get([llm.collective_rpc.remote("broadcast_all_weights", args=(0,)) for llm in self.inference_engines])
            
    def save_weights_to_disk(self, filepath: str):
        ray.get(self.inference_engines[0].collective_rpc.remote("save_weights_to_disk", args=(filepath,)))