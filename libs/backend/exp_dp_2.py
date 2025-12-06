import os
import sys
import time
import gc
import logging
from typing import List, Dict, Any
import multiprocessing
import traceback

# We delay importing jax/vllm until inside the worker processes
# to prevent the parent process from locking TPU resources.
import numpy as np

from libs.backend.backend_abc import Backend
from libs.genome import Genome
from libs.optimizers import Optimizer, SimpleOpt

# -- Worker Function --
def _tpu_worker_entrypoint(
    worker_id: int,
    device_id: str,
    model_name: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    sampler_params: Any,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    barrier: multiprocessing.Barrier
):
    """
    Independent process that owns one TPU device.
    """
    # 1. Device Isolation configuration
    os.environ["TPU_VISIBLE_DEVICES"] = str(device_id)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    
    # Imports must happen AFTER environment setup
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from vllm import LLM
    
    # Suppress logs in workers
    logging.getLogger("vllm").setLevel(logging.ERROR)

    try:
        # 2. Initialize Model
        print(f"[Worker {worker_id}] Initializing LLM on TPU device {device_id}...")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1, # DP means TP=1 per replica
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        # Access internal worker for weights
        if hasattr(llm.llm_engine, 'model_executor'):
            model_worker = llm.llm_engine.model_executor.driver_worker
        elif hasattr(llm.llm_engine, 'engine') and hasattr(llm.llm_engine.engine, 'model_executor'):
            model_worker = llm.llm_engine.engine.model_executor.driver_worker
        else:
            raise AttributeError("Could not find model_executor in vLLM engine.")

        # Helper for weight processing (Copied from your original logic)
        def process_weights(genome_obj, optim_obj, mode):
            state = model_worker.model_runner.state
            flat_state = list(state.flat_state())
            flat_state.sort(key=lambda x: str(x[0])) # Ensure deterministic order
            
            total_params = len(flat_state)
            chunk_size = 10 

            class SimpleParam:
                def __init__(self, value): self.value = value

            if mode == "update":
                # For update, we need the representative genome from the optimizer
                genome_obj = optim_obj.get_representative()

            # Process chunks
            for i in range(0, total_params, chunk_size):
                chunk_items = flat_state[i : i + chunk_size]
                chunk_update = {}
                chunk_mappings = {}

                for chunk_rel_idx, (path, val) in enumerate(chunk_items):
                    global_param_index = i + chunk_rel_idx
                    
                    # Dereference value
                    leaf = val.value if hasattr(val, 'value') else val
                    sharding = getattr(val, 'sharding', getattr(leaf, 'sharding', None))

                    if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
                        aggregate_delta = jnp.zeros(leaf.shape, dtype=jnp.float32)

                        # Apply perturbations
                        if genome_obj:
                            for seed, weight in zip(genome_obj.seeds, genome_obj.perturb_scales):
                                key = jax.random.PRNGKey(int(seed))
                                key = jax.random.fold_in(key, global_param_index)
                                noise = jax.random.normal(key, leaf.shape, dtype=jnp.float32)
                                aggregate_delta = aggregate_delta + (noise * weight)
                        
                        aggregate_delta = aggregate_delta.astype(leaf.dtype)
                        
                        if mode == "restore":
                            new_val = leaf - aggregate_delta
                        elif mode == "perturb":
                            new_val = leaf + aggregate_delta
                        elif mode == "update":
                            # Permanent update logic would go here if optimizer dictates it
                            # For SimpleOpt, it usually implies setting base weights.
                            # Assuming update applies the perturbation permanently:
                            new_val = leaf + aggregate_delta
                        else:
                            new_val = leaf
                    else:
                        new_val = leaf

                    key_str = '.'.join(str(k) for k in path)
                    chunk_update[key_str] = SimpleParam(new_val)
                    if sharding is not None:
                        chunk_mappings[key_str] = (key_str, sharding)

                # Sync chunk
                chunk_state = nnx.State(chunk_update)
                model_worker.sync_weights(updated_weights=chunk_state, mappings=chunk_mappings, transpose_keys={}, reshard_fn=None)
                
                # Block until ready to prevent OOM build up
                arrays = [p.value for p in chunk_update.values()]
                jax.block_until_ready(arrays)
                del chunk_update, chunk_mappings
                gc.collect()

        print(f"[Worker {worker_id}] Ready.")
        
        # 3. Work Loop
        while True:
            task = task_queue.get()
            
            if task is None: # Shutdown signal
                break
                
            task_type, payload = task

            if task_type == "GENERATE":
                genome, prompts = payload
                try:
                    # A. Perturb
                    process_weights(genome, None, "perturb")
                    
                    # B. Generate
                    # Note: We must re-tokenize inside worker because tokenizer is not picklable or valid across processes
                    # We assume 'prompts' passed here are raw strings (processed by backend) 
                    outputs = llm.generate(prompts, sampler_params, use_tqdm=False)
                    text_outputs = [o.outputs[0].text for o in outputs]
                    
                    # C. Restore
                    process_weights(genome, None, "restore")
                    
                    # D. Return result
                    result_queue.put(("SUCCESS", (genome.uid, text_outputs)))
                    
                except Exception as e:
                    print(f"[Worker {worker_id}] Error in generation: {e}")
                    traceback.print_exc()
                    result_queue.put(("ERROR", str(e)))

            elif task_type == "UPDATE":
                optimizer = payload
                try:
                    process_weights(None, optimizer, "update")
                    # Signal we are done with update
                    barrier.wait() 
                except Exception as e:
                    print(f"[Worker {worker_id}] Error in update: {e}")
                    barrier.wait() # Prevent deadlock even on error

    except Exception as e:
        print(f"[Worker {worker_id}] Critical Failure: {e}")
        traceback.print_exc()

# -- Main Backend Class --
class VllMTPUDataParallelBackend(Backend):
    def __init__(self, model_name: str, sampler: Any, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False, gpu_memory_utilization: float = 0.6, num_devices: int = 8):
        super().__init__(backend_name="vLLM TPU DP Backend", model_name=model_name, NUM_GPUS=num_devices, CPUS_PER_GPU=1, GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
        
        self.num_devices = num_devices
        self.gpu_memory_utilization = gpu_memory_utilization
        self.sampler = sampler
        
        # Multiprocessing primitives
        self.ctx = multiprocessing.get_context('spawn')
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.barrier = self.ctx.Barrier(num_devices + 1) # +1 for the main process
        self.processes = []

    def startup(self, trainer=None):
        print(f"#-- Initializing vLLM TPU Data Parallel Backend ({self.num_devices} Replicas) --#")
        
        for i in range(self.num_devices):
            p = self.ctx.Process(
                target=_tpu_worker_entrypoint,
                args=(
                    i, 
                    str(i), # device_id assumes linear mapping 0..7
                    self.model_name,
                    self.max_model_len,
                    self.gpu_memory_utilization,
                    self.sampler,
                    self.task_queue,
                    self.result_queue,
                    self.barrier
                )
            )
            p.start()
            self.processes.append(p)
        
        print("#-- Waiting for workers to initialize is implicit (no ready signal implemented, but generation will block) --#")

    def shutdown(self):
        for _ in range(self.num_devices):
            self.task_queue.put(None)
        for p in self.processes:
            p.join()

    def update(self, optimizer: Optimizer):
        """
        Updates all models universally.
        """
        print("#-- Updating Model Weights (All Replicas) --#")
        
        # Broadcast update command
        for _ in range(self.num_devices):
            self.task_queue.put(("UPDATE", optimizer))
            
        # Wait for all workers to finish updating (Barrier)
        # The barrier in worker waits for other workers. 
        # But we also need the main thread to know when they are done.
        # The logic here: Main thread enters barrier, waits for all N workers to enter barrier.
        self.barrier.wait()
        print("#-- Update Complete --#")

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """
        Dispatches genomes to workers round-robin via Queue.
        """
        assert len(genomes) == len(inputs)
        
        start_time_all = time.time()
        
        # 1. Pre-process prompts (Tokenizer must be handled carefully. 
        # Ideally, use a local tokenizer in main process to convert chat->string)
        # We need a tokenizer instance here just for formatting.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        processed_jobs = []
        for g, i_set in zip(genomes, inputs):
            prompt_strs = []
            for chat in i_set:
                s = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                if suffix: s += suffix
                prompt_strs.append(s)
            processed_jobs.append((g, prompt_strs))

        # 2. Enqueue Tasks
        for job in processed_jobs:
            # We send "GENERATE" command
            self.task_queue.put(("GENERATE", job))

        # 3. Collect Results
        # We expect exactly len(genomes) results
        results_collected = 0
        genome_map = {g.uid: g for g in genomes}
        
        while results_collected < len(genomes):
            res_type, res_data = self.result_queue.get()
            
            if res_type == "SUCCESS":
                uid, outputs = res_data
                if uid in genome_map:
                    genome_map[uid].latest_outputs = outputs
                results_collected += 1
                if self.use_tqdm:
                    print(f"Progress: {results_collected}/{len(genomes)}", end='\r')
            else:
                print(f"Error received from worker: {res_data}")
                # If a worker crashes, this loop might hang. In prod, handle timeouts.
                results_collected += 1

        if self.time_self:
            print(f"#-- All genomes generated in {time.time() - start_time_all:.2f}s --#")