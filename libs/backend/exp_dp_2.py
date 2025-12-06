import os
import sys

# --- CRITICAL FIXES FOR 8x INDEPENDENT TPU MODELS ---

# 1. Stop JAX from grabbing all memory on startup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 2. Force JAX to allocate memory on demand rather than in big slabs
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# 3. Disable V1 multiprocessing (as you did before)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.pop("TPU_MULTIHOST_BACKEND", None)

import time
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import logging

import jax
import jax.numpy as jnp
from flax import nnx
from vllm import LLM, SamplingParams

from libs.backend.backend_abc import Backend
from libs.genome import Genome
from libs.optimizers import Optimizer, SimpleOpt

logging.getLogger("vllm.tpu_inference").setLevel(logging.WARNING)

class VllMTPUIndependentBackend(Backend):
    def __init__(self, model_name: str, sampler: SamplingParams, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False, gpu_memory_utilization: float = 0.85):
        # Increased gpu_memory_utilization default to 0.85. 
        # Since we disabled preallocation, we can safely allow vLLM to use more of the actual chip.
        super().__init__(backend_name="vLLM TPU Independent Backend", model_name=model_name, NUM_GPUS=8, CPUS_PER_GPU=1, GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
        
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_devices = 8
        self.llms = []      
        self.workers = []   
        self.locks = []     

    def startup(self, trainer=None):
        print(f"#-- Initializing vLLM TPU Independent Backend (8x Independent Models) --#")
        print(f"   > Preallocation: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
        
        jax_devices = jax.devices()
        assert len(jax_devices) >= self.num_devices, f"Expected {self.num_devices} TPU devices, found {len(jax_devices)}"

        for i in range(self.num_devices):
            print(f"   > Initializing Model {i+1}/{self.num_devices} on device {jax_devices[i]}...")
            
            # We clear the backend cache to be safe, though mainly helps with compilation cache
            jax.clear_backends()
            gc.collect()

            with jax.default_device(jax_devices[i]):
                llm = LLM(
                    model=self.model_name,
                    tensor_parallel_size=1, 
                    trust_remote_code=True,
                    dtype="bfloat16",
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    # ENFORCE EAGER: Reduces memory overhead by skipping full graph capture
                    enforce_eager=True, 
                    # SWAP SPACE 0: TPUs handle CPU swap poorly, disable it to save RAM
                    swap_space=0,
                )
                
                self.llms.append(llm)
                self.workers.append(self._get_worker(llm))
                self.locks.append(threading.Lock())

        print("#-- vLLM TPU Independent Backend Initialized Successfully --#")

    def _get_worker(self, llm_instance):
        if hasattr(llm_instance.llm_engine, 'model_executor'):
            return llm_instance.llm_engine.model_executor.driver_worker
        elif hasattr(llm_instance.llm_engine, 'engine') and hasattr(llm_instance.llm_engine.engine, 'model_executor'):
            return llm_instance.llm_engine.engine.model_executor.driver_worker
        else:
            raise AttributeError("Could not find model_executor in vLLM engine structure.")

    def _process_weights_chunked(self, worker_idx: int, genome: Genome = None, optimizer: Optimizer = None, mode: str = "perturb"):
        worker = self.workers[worker_idx]
        
        state = worker.model_runner.state
        flat_state = list(state.flat_state())
        # Sorting ensures deterministic parameter indexing across all devices
        flat_state.sort(key=lambda x: str(x[0]))
        
        total_params = len(flat_state)
        chunk_size = 20 # Increased chunk size slightly for speed

        class SimpleParam:
            def __init__(self, value):
                self.value = value

        if mode == "update":
            assert isinstance(optimizer, SimpleOpt)
            genome = optimizer.get_representative()
        
        for i in range(0, total_params, chunk_size):
            current_state = worker.model_runner.state
            chunk_items = flat_state[i : i + chunk_size]
            chunk_paths = [item[0] for item in chunk_items]
            
            chunk_update = {}
            chunk_mappings = {}

            def get_value_by_path(root, path):
                node = root
                for p in path:
                    node = node[p]
                return node

            for chunk_rel_idx, path in enumerate(chunk_paths):
                global_param_index = i + chunk_rel_idx
                
                val = get_value_by_path(current_state, path)
                leaf = val.value if hasattr(val, 'value') else val
                sharding = getattr(val, 'sharding', getattr(leaf, 'sharding', None))

                if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
                    aggregate_delta = jnp.zeros(leaf.shape, dtype=jnp.float32)

                    for seed, weight in zip(genome.seeds, genome.perturb_scales):
                        key = jax.random.PRNGKey(int(seed))
                        key = jax.random.fold_in(key, global_param_index)
                        noise = jax.random.normal(key, leaf.shape, dtype=jnp.float32)
                        aggregate_delta = aggregate_delta + (noise * weight)
                    
                    aggregate_delta = aggregate_delta.astype(leaf.dtype)
                    
                    if mode == "restore":
                        new_val = leaf - aggregate_delta
                    else:
                        new_val = leaf + aggregate_delta
                else:
                    new_val = leaf

                key_str = '.'.join(str(k) for k in path)
                chunk_update[key_str] = SimpleParam(new_val)
                if sharding is not None:
                    chunk_mappings[key_str] = (key_str, sharding)

            chunk_state = nnx.State(chunk_update)
            worker.sync_weights(updated_weights=chunk_state, mappings=chunk_mappings, transpose_keys={}, reshard_fn=None)
            
            # Crucial: Block to ensure memory is freed before next chunk
            arrays_to_sync = [p.value for p in chunk_update.values()]
            jax.block_until_ready(arrays_to_sync)

    def update(self, optimizer: Optimizer):
        """Update ALL model instances simultaneously."""
        print("#-- Updating All Model Weights (Parallel) --#")
        def update_single_worker(idx):
            with self.locks[idx]:
                self._process_weights_chunked(worker_idx=idx, optimizer=optimizer, mode="update")
        
        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            list(executor.map(update_single_worker, range(self.num_devices)))
        gc.collect()

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        assert len(genomes) == len(inputs)
        
        # Use first tokenizer (all are same)
        tokenizer = self.llms[0].get_tokenizer()
        prompts = []
        for i in inputs:
            prompt_genome = []
            for j in i:
                s = tokenizer.apply_chat_template(j, tokenize=False, add_generation_prompt=True)
                if suffix: s += suffix
                prompt_genome.append(s)
            prompts.append(prompt_genome)

        start_time_all = time.time()
        
        import queue
        task_queue = queue.Queue()
        for t in zip(range(len(genomes)), genomes, prompts):
            task_queue.put(t)
            
        def device_worker(device_idx):
            while True:
                try:
                    task = task_queue.get_nowait()
                except queue.Empty:
                    return

                idx, genome, prompt_set = task
                if self.time_self: print(f"[Device {device_idx}] Start Genome {idx}")

                with self.locks[device_idx]:
                    self._process_weights_chunked(worker_idx=device_idx, genome=genome, mode="perturb")
                    
                    outputs = self.llms[device_idx].generate(prompt_set, self.sampler, use_tqdm=False)
                    genome.latest_outputs = [o.outputs[0].text for o in outputs]
                    
                    self._process_weights_chunked(worker_idx=device_idx, genome=genome, mode="restore")

                task_queue.task_done()
                if self.time_self: print(f"[Device {device_idx}] Done Genome {idx}")

        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            futures = [executor.submit(device_worker, i) for i in range(self.num_devices)]
            for f in futures: f.result()

        if self.time_self:
            print(f"#-- All genomes generated in {time.time() - start_time_all:.2f}s --#")