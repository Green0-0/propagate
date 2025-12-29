# Note: This script is largely the work of https://github.com/dibbla and has been modified from the repo at https://github.com/VsonicV/es-fine-tuning-paper/tree/main
from typing import Dict, List
from propagate.backend.backend_abc import Backend

import signal
import sys
import os
import ray
import time 
import math
from transformers import AutoTokenizer
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_ip, get_open_port

from propagate.genome import Genome
from propagate.optimizers import Optimizer

class VLLMBackend(Backend):
    """The standard vLLM backend. Uses Ray to spawn vLLM workers and distribute inference across them.
    Evolution Strategy (ES) perturbation happens directly on the model weights in this backend.
    
    Attributes
    ----------
    model_name : str
         The name of the model to use.
    NUM_GPUS : int
         The number of GPUs to use.
    CPUS_PER_GPU : int
         The number of CPUs to allocate per GPU.
    GPU_FRACTION_VLLM_WORKER : float
         The fraction of GPU memory to allocate to the vLLM worker.
    sampler : SamplingParams
         The sampling parameters to use for generation.
    use_tqdm : bool
         Whether to use tqdm for progress bars.
    max_model_len : int
         The maximum model length.
    time_self : bool
         Whether to print timing information.
    inference_engines : List[RayActor]
         A list of Ray actors handling the inference.
    """
    def __init__(self, model_name: str, NUM_GPUS: int, CPUS_PER_GPU: int, GPU_FRACTION_VLLM_WORKER: float, sampler: SamplingParams, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False):
        super().__init__(backend_name="Standard vLLM Backend", model_name=model_name, NUM_GPUS=NUM_GPUS, CPUS_PER_GPU=CPUS_PER_GPU, GPU_FRACTION_VLLM_WORKER=GPU_FRACTION_VLLM_WORKER, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
    
    def startup(self, trainer=None):
        """Initializes the vLLM backend with Ray actors and placement groups.
        Automatically detects single-node vs multi-node mode based on environment (multi-node requires a head node IP to be set in the environment).
        Also sets up broadcasting group for weight sync.
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        head_ip = os.environ.get("head_node_ip")
        if head_ip is None:
            print("#-- Single-node mode: starting local Ray instance --#")
            ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)
            pgs = [placement_group([{"GPU": 1, "CPU": self.CPUS_PER_GPU}]) for _ in range(self.NUM_GPUS)]
        else:
            ray_address = f"{head_ip}:6379"
            print(f"Connecting to Ray cluster at {ray_address}...")
            ray.init(address=ray_address, include_dashboard=False, ignore_reinit_error=True)
            
            resources = ray.cluster_resources()
            print(f"Ray cluster resources: {resources}")
            available_gpus = int(resources.get("GPU", 0))
            
            if available_gpus < self.NUM_GPUS:
                raise RuntimeError(f"Requested {self.NUM_GPUS} GPUs but only {available_gpus} available")
            
            print(f"#-- Creating {self.NUM_GPUS} placement groups (engine 0 on head node) --#")
            
            pgs = [placement_group([{"GPU": 1, "CPU": self.CPUS_PER_GPU, f"node:{head_ip}": 0.001}])]
            
            # Spread the rest
            pgs += [
                placement_group([{"GPU": 1, "CPU": self.CPUS_PER_GPU}], strategy="SPREAD") 
                for _ in range(self.NUM_GPUS - 1)
            ]
        ray.get([pg.ready() for pg in pgs])
        print(f"#-- All {self.NUM_GPUS} placement groups ready --#")
        strategies = [PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0) for pg in pgs]
        
        print("#-- Spawning Training Actors with vLLM backends --#")
        self.inference_engines = [
            ray.remote(
                num_cpus=0,
                num_gpus=0,
                scheduling_strategy=strategy,
            )(MyLLM).remote(
                model=self.model_name,
                tensor_parallel_size=1,
                distributed_executor_backend="ray",
                worker_extension_cls="propagate.backend.vllm_utils.WorkerExtension",
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
            ray.get([
                self.inference_engines[i].collective_rpc.remote(
                    "init_inter_engine_group", 
                    args=(master_address, master_port, i, self.NUM_GPUS)
                ) 
                for i in range(self.NUM_GPUS)
            ])
        else:
            print("#-- Skipping collective group (1 GPU) --#")
        # --- Cleanup handlers ---
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
        """Return a generation handle so we can schedule round-robin.
        
        Args:
            llm (RayActor): The Ray actor to schedule the generation on.
            prompts (List[str]): The prompts to generate from.

        Returns:
            Tuple[ObjectRef, float]: A tuple containing the Ray object reference for the generation and the start time.
        """
        start = time.time()
        handle = llm.generate.remote(prompts, self.sampler, use_tqdm=self.use_tqdm)
        return handle, start

    def update(self, optimizer: Optimizer):
        """Update the model permanently with a genome as the source.
        
        Args:
           optimizer (Optimizer): The optimizer containing the update logic and state.
        """
        ray.get([llm.collective_rpc.remote("update_weights", args=(optimizer,)) for llm in self.inference_engines])

        if self.NUM_GPUS > 1:
            ray.get([llm.collective_rpc.remote("broadcast_all_weights", args=(0,)) for llm in self.inference_engines])

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """Generate outputs based on the genome and inputs.
        Updates the genomes with their new outputs.
        
        Args:
            genomes (List[Genome]): The list of genomes to evaluate.
            suffix (str): Only used for debugging / CoT triggering. The suffix to append to the prompt.
            inputs (List[List[List[Dict[str, str]]]]): The inputs to evaluate on. Matches the length of genomes.
        """
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        # Format dataset for generation, cache latest inputs for each genome
        prompts = []
        for idk, i in enumerate(inputs):
            prompt_genome = []
            input_genome_content = []
            for j in i:
                input_genome_content.append(j[-1]['content'])
                s = self.tokenizer.apply_chat_template(j, tokenize=False, add_generation_prompt=True)
                if suffix is not None:
                    s = s + suffix
                prompt_genome.append(s)
            prompts.append(prompt_genome)
            genomes[idk].latest_inputs = input_genome_content

        gs = iter(genomes)
        ds = iter(prompts)
        inflight = {}

        # Schedule generation on each engine
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
            # Wait for a generation to complete
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            # Get outputs and update genome
            outputs = ray.get(h)
            genome = meta["genome"]

            genome.latest_outputs = [o.outputs[0].text for o in outputs]

            # Restore weights and schedule next generation
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
        # Sync weights across engines to prevent floating point error accumulation during perturb-restore steps
        if self.NUM_GPUS > 1:
            ray.get([llm.collective_rpc.remote("broadcast_all_weights", args=(0,)) for llm in self.inference_engines])
            
    def save_weights_to_disk(self, filepath: str):
        """Save the weights of the first inference engine to disk.
        
        Args:
            filepath (str): The path to save the weights to.
        """
        ray.get(self.inference_engines[0].collective_rpc.remote("save_weights_to_disk", args=(filepath,)))

    def load_weights_from_disk(self, filepath: str):
        """Load weights from disk into all inference engines.
        
        Args:
            filepath (str): The path to load the weights from.
        """
        ray.get([llm.collective_rpc.remote("load_weights_from_disk", args=(filepath,)) for llm in self.inference_engines])

    def inference(self, conversations: List[List[Dict[str, str]]]):
        """
        Inference mode: takes a batch of formatted conversations, applies tokenizer, runs inference, and returns the outputs.

        Args:
            conversations (List[List[Dict[str, str]]]): A list of conversations (messages) to generate from.

        Returns:
            List[str]: A list of generated outputs.
        """
        prompts = [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
        
        num_engines = len(self.inference_engines)
        chunk_size = math.ceil(len(prompts) / num_engines)
        
        handles = []
        for i, engine in enumerate(self.inference_engines):
            chunk = prompts[i * chunk_size : (i + 1) * chunk_size]
            if chunk:
                h = engine.generate.remote(chunk, self.sampler, use_tqdm=False)
                handles.append(h)
        
        results_list = ray.get(handles)
        
        final_outputs = []
        for eng_results in results_list:
            for req_out in eng_results:
                final_outputs.append(req_out.outputs[0].text)
                
        return final_outputs