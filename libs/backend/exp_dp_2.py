import os
import sys
import time
import gc
import logging
from typing import List, Dict, Any
import asyncio

# Ray is required for the multi-process orchestration
import ray

# We keep the environment setup, but note that Ray workers handle their own envs via runtime_env
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 
os.environ.pop("TPU_MULTIHOST_BACKEND", None)

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Param 
from vllm import LLM, SamplingParams

from libs.backend.backend_abc import Backend
from libs.genome import Genome
from libs.optimizers import Optimizer, SimpleOpt

# Suppress logging in main process
logging.getLogger("vllm.tpu_inference").setLevel(logging.WARNING)

@ray.remote(resources={"TPU": 1})
class SingleTPUWorker:
    """
    A Ray Actor that controls a single TPU chip entirely.
    It runs an independent vLLM instance (Tensor Parallel = 1).
    """
    def __init__(self, model_name: str, max_model_len: int, gpu_memory_utilization: float, seed: int):
        # Re-import inside actor to ensure JAX sees the specific TPU assigned by Ray
        import os
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from flax.nnx import Param
        from vllm import LLM
        
        # Enforce the env var for internal access inside this process
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        
        self.model_name = model_name
        
        print(f"DEBUG: Worker initializing on device: {jax.devices()}")

        # Initialize vLLM with TP=1 (Running on single chip)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1, 
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed
        )
        self.worker_internal = self.llm.llm_engine.model_executor.driver_worker

    def is_trainable_param(self, path, node):
        if not isinstance(node, Param):
            return False
        key_str = '.'.join(map(str, path))
        banned = ('rotary', 'kv_cache','inv_freq', 'cos_cached', 'sin_cached')
        if any(b in key_str.lower() for b in banned):
            return False
        return True 

    def _process_weights_chunked(self, genome: Genome = None, optimizer: Optimizer = None, mode: str = "perturb"):
        """
        Adapted from original script to run locally on this worker's model.
        """
        worker = self.worker_internal
        state = worker.model_runner.state
        
        flat_state = list(state.flat_state())
        flat_state.sort(key=lambda x: str(x[0]))
        
        total_params = len(flat_state)
        chunk_size = 10 

        class SimpleParam:
            def __init__(self, value):
                self.value = value

        keys = []
        if mode == "update":
            # For update, we need the optimizer's representative genome
            assert isinstance(optimizer, SimpleOpt)
            genome = optimizer.get_representative()
            
        # Prepare RNG keys
        keys = [(jax.random.PRNGKey(int(seed)), weight) for seed, weight in zip(genome.seeds, genome.perturb_scales)]

        for i in range(0, total_params, chunk_size):
            current_state = worker.model_runner.state
            chunk_paths = [item[0] for item in flat_state[i:i+chunk_size]]
            
            chunk_update = {}
            chunk_mappings = {}

            def get_value_by_path(root, path):
                node = root
                for p in path:
                    node = node[p]
                return node

            for path in chunk_paths:                
                val = get_value_by_path(current_state, path)
                if not self.is_trainable_param(path, val):
                    leaf = val.value if hasattr(val, 'value') else val
                    key_str = '.'.join(str(k) for k in path)
                    chunk_update[key_str] = SimpleParam(leaf)
                    continue
                
                leaf = val
                sharding = None
                if hasattr(val, 'value'):
                    leaf = val.value
                if hasattr(val, 'sharding'):
                    sharding = val.sharding
                elif hasattr(leaf, 'sharding'):
                    sharding = leaf.sharding

                if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
                    # Re-calculate noise locally on this device
                    aggregate_delta = jnp.zeros(leaf.shape, dtype=leaf.dtype)
                    
                    for k, item in enumerate(keys):
                        seed_key, weight = item
                        key, subkey = jax.random.split(seed_key)
                        noise = jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * weight
                        keys[k] = (key, weight)
                        aggregate_delta = aggregate_delta + noise
                    
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
            
            del chunk_update
            del chunk_mappings
            gc.collect()

    def generate_task(self, genome: Genome, prompts: List[str], sampler: SamplingParams):
        """
        Full lifecycle for a batch: Perturb -> Generate -> Restore -> Return Text
        """
        # 1. Perturb
        self._process_weights_chunked(genome=genome, mode="perturb")
        
        # 2. Generate
        # Note: use_tqdm=False to keep logs clean during parallel execution
        outputs = self.llm.generate(prompts, sampler, use_tqdm=True)
        text_outputs = [o.outputs[0].text for o in outputs]
        
        # 3. Restore
        self._process_weights_chunked(genome=genome, mode="restore")
        
        return text_outputs

    def update_weights(self, optimizer: Optimizer):
        """
        Apply a permanent update to this worker's model.
        """
        self._process_weights_chunked(optimizer=optimizer, mode="update")
        return True


class VllMTPUDPBackend(Backend):
    def __init__(self, model_name: str, sampler: SamplingParams, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False, gpu_memory_utilization: float = 0.6, tensor_parallel_size: int = 8):
        # Note: We ignore tensor_parallel_size for the backend logic, as we enforce TP=1 per worker.
        # But we use the value to determine how many separate workers to spawn (assuming 1 worker per TPU chip).
        super().__init__(backend_name="vLLM Ray DataParallel Backend", model_name=model_name, NUM_GPUS=tensor_parallel_size, CPUS_PER_GPU=1, GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
        
        self.num_workers = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.workers = []

    def startup(self, trainer=None):
        """Initialize the Ray cluster and TPU workers."""
        print(f"#-- Initializing Ray DataParallel Backend ({self.num_workers} Workers) --#")
        
        # Initialize Ray if not already running
        if not ray.is_initialized():
            # runtime_env is CRITICAL here to ensure the env var propagates to workers
            ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}})

        # Spawn workers
        print(f"Spawning {self.num_workers} TPU workers...")
        self.workers = [
            SingleTPUWorker.remote(
                model_name=self.model_name,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                seed=42 + i # Different seed per worker initialization just in case, though weights are synced
            )
            for i in range(self.num_workers)
        ]
        
        # Wait for all workers to be ready (force instantiation)
        # We do a cheap call to ensure they are up
        ray.get([w.is_trainable_param.remote([], None) for w in self.workers])
        print("#-- All TPU Workers Initialized Successfully --#")

    def update(self, optimizer: Optimizer):
        """Update ALL models universally."""
        print("#-- Updating All Model Replicas (TPU) --#")
        # Broadcast update to all workers
        futures = [worker.update_weights.remote(optimizer) for worker in self.workers]
        ray.get(futures)
        print("#-- Update Complete --#")

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """
        Round-robin generation:
        1. Create a queue of tasks (Genome + Input Batch).
        2. Assign tasks to free workers.
        3. Collect results.
        """
        assert len(genomes) == len(inputs), "Number of genomes must match input sets."
        
        start_time_all = time.time()
        
        # Pre-process prompts on the head node to save worker time
        # We use a throwaway tokenizer here or the one from the first worker if needed.
        # Ideally, load a tokenizer locally to avoid remote calls for simple string formatting.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        tasks_queue = []
        for i, (genome, input_batch) in enumerate(zip(genomes, inputs)):
            prompt_genome = []
            for chat in input_batch:
                s = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                if suffix:
                    s = s + suffix
                prompt_genome.append(s)
            
            # Store tuple: (Index, GenomeObj, Prompts)
            tasks_queue.append((i, genome, prompt_genome))

        # Result storage
        results = [None] * len(genomes)
        
        # Running map of {Future: (WorkerIndex, TaskIndex, GenomeObj)}
        running_tasks = {}
        
        # Initial fill of workers
        worker_pool = list(range(len(self.workers))) # List of available worker indices
        
        while tasks_queue or running_tasks:
            # Fill available workers
            while worker_pool and tasks_queue:
                worker_idx = worker_pool.pop(0)
                task_idx, genome, prompts = tasks_queue.pop(0)
                
                # Submit task
                worker = self.workers[worker_idx]
                future = worker.generate_task.remote(genome, prompts, self.sampler)
                
                running_tasks[future] = (worker_idx, task_idx, genome)
            
            # Wait for at least one result
            if running_tasks:
                done_futures, _ = ray.wait(list(running_tasks.keys()), num_returns=1)
                
                for future in done_futures:
                    worker_idx, task_idx, genome = running_tasks.pop(future)
                    
                    try:
                        outputs = ray.get(future)
                        genome.latest_outputs = outputs
                        results[task_idx] = outputs # Optional: store in list if needed outside genome
                        
                        if self.time_self:
                             print(f"#-- Batch {task_idx+1} finished --#")
                    except Exception as e:
                        print(f"Error in worker {worker_idx} on task {task_idx}: {e}")
                    
                    # Return worker to pool
                    worker_pool.append(worker_idx)

        if self.time_self:
            print(f"#-- All genomes generated in {time.time() - start_time_all:.2f}s --#")

    def save_weights_to_disk(self, filepath: str):
        """Save weights from the first worker (assuming all are synced)."""
        print(f"#-- Saving weights from Worker 0 to {filepath} --#")
        # We need a method on the worker to export weights
        # Implementing a custom remote method for this:
        
        @ray.remote
        def get_cpu_state(worker):
            # This logic mimics the original save logic but runs inside the worker
            import torch
            import jax
            state = worker.worker_internal.model_runner.state
            flat_state = state.flat_state()
            cpu_state = {}
            for path, val in flat_state:
                key = '.'.join(str(p) for p in path)
                if hasattr(val, 'value'):
                    cpu_state[key] = jax.device_get(val.value)
                else:
                    cpu_state[key] = jax.device_get(val)
            return cpu_state

        # Get state from worker 0
        cpu_state = ray.get(get_cpu_state.remote(self.workers[0]))
        
        import torch
        torch.save(cpu_state, filepath)
        print("#-- Weights saved successfully --#")