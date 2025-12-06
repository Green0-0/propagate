import os
import sys
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

# Suppress warnings
logging.getLogger("vllm.tpu_inference").setLevel(logging.WARNING)

class VllMTPUIndependentBackend(Backend):
    def __init__(self, model_name: str, sampler: SamplingParams, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False, gpu_memory_utilization: float = 0.6):
        # Note: tensor_parallel_size is set to 1 effectively per model, but we have 8 distinct models.
        # We pass 1 here to the parent, but we manage 8 internal instances.
        super().__init__(backend_name="vLLM TPU Independent Backend", model_name=model_name, NUM_GPUS=8, CPUS_PER_GPU=1, GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
        
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_devices = 8
        self.llms = []      # List of LLM objects
        self.workers = []   # List of Model Workers
        self.locks = []     # Locks per device (just in case)

    def startup(self, trainer=None):
        """Initialize 8 separate vLLM engines, one per TPU device."""
        print(f"#-- Initializing vLLM TPU Independent Backend (8x Independent Models) --#")

        # Crucial: Disable V1 multiprocessing to keep everything in this process
        # so we can hack the weights manually.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.pop("TPU_MULTIHOST_BACKEND", None)

        # Get all JAX devices (should be 8 TPU cores)
        jax_devices = jax.devices()
        assert len(jax_devices) >= self.num_devices, f"Expected {self.num_devices} TPU devices, found {len(jax_devices)}"

        for i in range(self.num_devices):
            print(f"   > Initializing Model {i+1}/{self.num_devices} on device {jax_devices[i]}...")
            
            # Force JAX operations (like weight loading) to happen on this specific device
            # during the initialization of this specific LLM instance.
            with jax.default_device(jax_devices[i]):
                llm = LLM(
                    model=self.model_name,
                    tensor_parallel_size=1, # TP=1 means independent model
                    trust_remote_code=True,
                    dtype="bfloat16",
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    # We rely on JAX context to place it, but vLLM might check visible devices.
                    # Usually vLLM on TPU respects the mesh if provided, or defaults. 
                    # In single-process mode, sequential init with default_device works best.
                )
                
                self.llms.append(llm)
                self.workers.append(self._get_worker(llm))
                self.locks.append(threading.Lock())
                
                # Force garbage collection to ensure we don't peak memory between loads
                gc.collect()

        print("#-- vLLM TPU Independent Backend Initialized Successfully --#")

    def _get_worker(self, llm_instance):
        """Extract the worker from a specific LLM instance."""
        if hasattr(llm_instance.llm_engine, 'model_executor'):
            return llm_instance.llm_engine.model_executor.driver_worker
        elif hasattr(llm_instance.llm_engine, 'engine') and hasattr(llm_instance.llm_engine.engine, 'model_executor'):
            return llm_instance.llm_engine.engine.model_executor.driver_worker
        else:
            raise AttributeError("Could not find model_executor in vLLM engine structure.")

    def _process_weights_chunked(self, worker_idx: int, genome: Genome = None, optimizer: Optimizer = None, mode: str = "perturb"):
        """
        Handles weight perturbation, restoration, and updates for a SPECIFIC worker.
        """
        worker = self.workers[worker_idx]
        
        state = worker.model_runner.state
        flat_state = list(state.flat_state())
        flat_state.sort(key=lambda x: str(x[0]))
        
        total_params = len(flat_state)
        chunk_size = 10 

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
                
                leaf = val
                sharding = None
                if hasattr(val, 'value'):
                    leaf = val.value
                if hasattr(val, 'sharding'):
                    sharding = val.sharding
                elif hasattr(leaf, 'sharding'):
                    sharding = leaf.sharding

                if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
                    # Compute noise
                    # Note: We must ensure the noise calculation is efficient and identical 
                    # regardless of which device runs it, but the seed controls the randomness.
                    
                    aggregate_delta = jnp.zeros(leaf.shape, dtype=jnp.float32)

                    for seed, weight in zip(genome.seeds, genome.perturb_scales):
                        key = jax.random.PRNGKey(int(seed))
                        key = jax.random.fold_in(key, global_param_index)
                        # Generate noise directly on the target device if possible
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
            
            # Block until ready to prevent OOM / queuing up too many async ops
            arrays_to_sync = [p.value for p in chunk_update.values()]
            jax.block_until_ready(arrays_to_sync)

            del chunk_update
            del chunk_mappings
            # Explicit GC per chunk often slows things down too much, 
            # better to do it per batch or rely on Python.
            # gc.collect() 

    def update(self, optimizer: Optimizer):
        """
        Update ALL model instances simultaneously with the optimized genome.
        They must all stay in sync.
        """
        print("#-- Updating All Model Weights (Parallel) --#")
        
        def update_single_worker(idx):
            with self.locks[idx]:
                # We use the existing logic but in 'update' mode
                self._process_weights_chunked(worker_idx=idx, optimizer=optimizer, mode="update")
        
        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            list(executor.map(update_single_worker, range(self.num_devices)))
            
        gc.collect()

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """
        Round-Robin Generation.
        1. Pre-process all prompts.
        2. Create a queue of tasks.
        3. Use a thread pool to assign tasks to the first available TPU device.
        """
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        
        # Pre-process prompts
        prompts = []
        for i in inputs:
            prompt_genome = []
            # We can use the first tokenizer, they are all the same
            tokenizer = self.llms[0].get_tokenizer() 
            for j in i:
                s = tokenizer.apply_chat_template(j, tokenize=False, add_generation_prompt=True)
                if suffix is not None:
                    s = s + suffix
                prompt_genome.append(s)
            prompts.append(prompt_genome)

        start_time_all = time.time()
        
        # Queue of items: (index, genome, specific_prompts)
        tasks = list(zip(range(len(genomes)), genomes, prompts))
        
        # We need a function that consumes a task using a specific device index
        # The ThreadPoolExecutor isn't quite right for "Device Affinity" automatically,
        # so we implement a custom worker loop or map inputs to available workers.
        # Simpler approach: Create a queue, and spawn 8 persistent threads (one per device).
        
        import queue
        task_queue = queue.Queue()
        for t in tasks:
            task_queue.put(t)
            
        results_lock = threading.Lock()
        
        def device_worker(device_idx):
            while True:
                try:
                    # Get a task, non-blocking if possible or timeout
                    task = task_queue.get_nowait()
                except queue.Empty:
                    return

                idx, genome, prompt_set = task
                
                if self.time_self:
                    print(f"[Device {device_idx}] Starting Genome {idx}...")

                with self.locks[device_idx]:
                    # 1. Perturb
                    self._process_weights_chunked(worker_idx=device_idx, genome=genome, mode="perturb")
                    
                    # 2. Generate
                    # We use the specific LLM instance for this device
                    gen_start = time.time()
                    outputs = self.llms[device_idx].generate(prompt_set, self.sampler, use_tqdm=False)
                    
                    # Store results safely
                    text_outputs = [o.outputs[0].text for o in outputs]
                    genome.latest_outputs = text_outputs
                    
                    # 3. Restore
                    self._process_weights_chunked(worker_idx=device_idx, genome=genome, mode="restore")

                task_queue.task_done()
                
                if self.time_self:
                    print(f"[Device {device_idx}] Finished Genome {idx} in {time.time() - gen_start:.2f}s")

        # Spawn 8 threads, one strictly bound to each device index
        with ThreadPoolExecutor(max_workers=self.num_devices) as executor:
            futures = [executor.submit(device_worker, i) for i in range(self.num_devices)]
            # Wait for all to finish
            for f in futures:
                f.result()

        if self.time_self:
            print(f"#-- All genomes generated in {time.time() - start_time_all:.2f}s --#")

    def save_weights_to_disk(self, filepath: str):
        """Save the model weights from the first worker (since they are all synced)."""
        print(f"#-- Saving weights to {filepath} --#")
        # We only need to save from worker 0, as updates are universal
        state = self.workers[0].model_runner.state
        flat_state = state.flat_state()
        
        cpu_state = {}
        for path, val in flat_state:
            key = '.'.join(str(p) for p in path)
            if hasattr(val, 'value'):
                cpu_state[key] =  jax.device_get(val.value)
            else:
                cpu_state[key] = jax.device_get(val)
                
        import torch
        torch.save(cpu_state, filepath)
        print("#-- Weights saved successfully --#")