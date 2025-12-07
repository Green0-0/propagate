import os
import sys
import time
import gc
import multiprocessing as mp
import logging
import queue
from typing import List, Dict, Any

# -- Global Environmental Setup (Parent Process) --
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# We keep this 0 to allow the WORKER to access internals, 
# but we manage the actual multiprocessing ourselves.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 
os.environ.pop("TPU_MULTIHOST_BACKEND", None)

import jax
import jax.numpy as jnp
# Note: We import flax/vllm inside the worker to avoid parent-process JAX lock-in
from libs.backend.backend_abc import Backend
from libs.genome import Genome
from libs.optimizers import Optimizer, SimpleOpt

from vllm import SamplingParams

# -- Worker Function --
def _tpu_worker_fn(rank: int, 
                   model_name: str, 
                   max_model_len: int,
                   gpu_memory_utilization: float,
                   input_queue: mp.Queue, 
                   result_queue: mp.Queue):
    """
    Independent worker process that owns one TPU core.
    """
    # 1. Isolate the TPU device for this process
    os.environ["TPU_VISIBLE_DEVICES"] = str(rank)
    
    # 2. Import libraries locally to bind to the specific TPU core
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from flax.nnx import Param 
    from vllm import LLM, SamplingParams

    # Silence logs in workers
    logging.getLogger("vllm").setLevel(logging.ERROR)

    # 3. Initialize Model (TP=1 means full model on this single chip)
    print(f"[Worker {rank}] Initializing Model on TPU Core {rank}...")
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=1,  # TP=1 is crucial here
        trust_remote_code=True, 
        dtype="bfloat16", 
        max_model_len=max_model_len, 
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    # Access the worker guts
    # Since VLLM_ENABLE_V1_MULTIPROCESSING=0, this works directly
    model_worker = llm.llm_engine.model_executor.driver_worker
    
    # --- Internal Helper for Weight Manipulation ---
    def is_trainable_param(path, node):
        if not isinstance(node, Param):
            return False
        key_str = '.'.join(map(str, path))
        banned = ('rotary', 'kv_cache','inv_freq', 'cos_cached', 'sin_cached')
        if any(b in key_str.lower() for b in banned):
            return False
        return True 

    class SimpleParam:
        def __init__(self, value):
            self.value = value

    def process_weights(genome_data, mode="perturb"):
        """
        Modified version of _process_weights_chunked that runs locally.
        genome_data: tuple (seeds, scales) or None
        """
        state = model_worker.model_runner.state
        flat_state = list(state.flat_state())
        flat_state.sort(key=lambda x: str(x[0]))
        
        # Prepare perturbation keys if needed
        keys = []
        if mode != "update" and genome_data:
            seeds, scales = genome_data
            keys = [(jax.random.PRNGKey(int(s)), w) for s, w in zip(seeds, scales)]
        elif mode == "update" and genome_data:
            # genome_data is (seeds, scales) from optimizer
            seeds, scales = genome_data
            keys = [(jax.random.PRNGKey(int(s)), w) for s, w in zip(seeds, scales)]

        chunk_size = 10 
        total_params = len(flat_state)

        for i in range(0, total_params, chunk_size):
            current_state = model_worker.model_runner.state
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
                
                # Filter non-trainables
                if not is_trainable_param(path, val):
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
                    # Re-generate noise locally
                    aggregate_delta = jnp.zeros(leaf.shape, dtype=leaf.dtype)
                    
                    for k, item in enumerate(keys):
                        seed_key, weight = item
                        key, subkey = jax.random.split(seed_key)
                        noise = jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype) * weight
                        keys[k] = (key, weight) # Update key state
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
            
            # Commit chunk
            chunk_state = nnx.State(chunk_update)
            model_worker.sync_weights(updated_weights=chunk_state, mappings=chunk_mappings, transpose_keys={}, reshard_fn=None)
            del chunk_update
            del chunk_mappings
        
        gc.collect()

    print(f"[Worker {rank}] Ready.")
    
    # 4. Event Loop
    while True:
        try:
            task = input_queue.get()
            
            if task['type'] == 'STOP':
                break
            
            elif task['type'] == 'GENERATE':
                # task payload: {'genome_seeds': [...], 'genome_scales': [...], 'prompts': [...], 'genome_id': int, 'sampling_params': dict}
                
                # A. Perturb
                genome_data = (task['genome_seeds'], task['genome_scales'])
                process_weights(genome_data, mode="perturb")
                
                # B. Generate
                # Reconstruct sampling params object
                sp_dict = task['sampling_params']
                sampling_params = SamplingParams(**sp_dict)
                
                outputs = llm.generate(task['prompts'], sampling_params, use_tqdm=False)
                generated_texts = [o.outputs[0].text for o in outputs]
                
                # C. Restore
                process_weights(genome_data, mode="restore")
                
                # D. Send result
                result_queue.put({
                    'type': 'RESULT',
                    'genome_id': task['genome_id'],
                    'outputs': generated_texts
                })
                
            elif task['type'] == 'UPDATE':
                # task payload: {'optimizer_seeds': [...], 'optimizer_scales': [...]}
                # Universally update weights
                optim_data = (task['optimizer_seeds'], task['optimizer_scales'])
                process_weights(optim_data, mode="update")
                
                # Acknowledge update completion
                result_queue.put({'type': 'UPDATE_ACK', 'worker_rank': rank})
                
        except Exception as e:
            print(f"[Worker {rank}] Error: {e}")
            import traceback
            traceback.print_exc()

class VllMTPUDPBackend(Backend):
    def __init__(self, model_name: str, sampler: SamplingParams, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False, gpu_memory_utilization: float = 0.6, tensor_parallel_size: int = 8):
        # Note: tensor_parallel_size passed here is interpreted as "Total Devices" available
        super().__init__(backend_name="vLLM DataParallel Backend", model_name=model_name, NUM_GPUS=tensor_parallel_size, CPUS_PER_GPU=1, GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
        
        self.num_workers = tensor_parallel_size
        self.workers = []
        self.input_queues = []
        self.result_queue = mp.Queue()
        
        # We need the sampler params as a dict to pass to workers
        self.sampler_dict = {
            "n": sampler.n,
            "temperature": sampler.temperature,
            "top_p": sampler.top_p,
            "top_k": sampler.top_k,
            "max_tokens": sampler.max_tokens,
            "stop": sampler.stop,
            # Add other necessary fields from your sampler object
        }
        self.gpu_memory_utilization = gpu_memory_utilization

    def startup(self, trainer=None):
        print(f"#-- Spawning {self.num_workers} vLLM TPU Workers (DP Mode) --#")
        
        # Create a shared input queue or individual queues? 
        # A shared queue allows faster workers to pick up more tasks (Round Robin / Load Balancing)
        self.task_queue = mp.Queue()
        
        for i in range(self.num_workers):
            p = mp.Process(
                target=_tpu_worker_fn,
                args=(
                    i, 
                    self.model_name, 
                    self.max_model_len, 
                    self.gpu_memory_utilization, 
                    self.task_queue, 
                    self.result_queue
                )
            )
            p.start()
            self.workers.append(p)
            
        print("#-- Workers Spawned. Waiting for ready signal (implied by first generation) --#")

    def update(self, optimizer: Optimizer):
        """Update the model permanently across all devices."""
        print("#-- Updating Model Weights (Universal) --#")
        
        assert isinstance(optimizer, SimpleOpt)
        rep = optimizer.get_representative()
        
        # Send update command to ALL workers
        for _ in range(self.num_workers):
            self.task_queue.put({
                'type': 'UPDATE',
                'optimizer_seeds': rep.seeds,
                'optimizer_scales': rep.perturb_scales
            })
            
        # Wait for all workers to ACK
        acks = 0
        while acks < self.num_workers:
            msg = self.result_queue.get()
            if msg['type'] == 'UPDATE_ACK':
                acks += 1
        
        print("#-- All workers updated successfully --#")

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        assert len(genomes) == len(inputs)
        
        # 1. Tokenization (Must happen in main process to avoid loading tokenizer in workers if possible, 
        #    but standard vLLM usage usually does tokenization inside generate. 
        #    Here we prepare string prompts.)
        
        # We need a tokenizer instance in main process just for templating, 
        # or we assume inputs are already chat templates? 
        # The original script does apply_chat_template. We need a tokenizer here.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        start_time = time.time()
        
        # 2. Enqueue Tasks
        for i, (genome, input_batch) in enumerate(zip(genomes, inputs)):
            
            # Prepare prompts
            final_prompts = []
            for chat in input_batch:
                s = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                if suffix:
                    s = s + suffix
                final_prompts.append(s)

            # Push to queue
            # We pass seeds/scales instead of full weights to save bandwidth
            task = {
                'type': 'GENERATE',
                'genome_id': i,
                'genome_seeds': genome.seeds,
                'genome_scales': genome.perturb_scales,
                'prompts': final_prompts,
                'sampling_params': self.sampler_dict
            }
            self.task_queue.put(task)
            
        # 3. Collect Results
        completed = 0
        results_map = {} # genome_id -> outputs
        
        while completed < len(genomes):
            msg = self.result_queue.get()
            if msg['type'] == 'RESULT':
                gid = msg['genome_id']
                results_map[gid] = msg['outputs']
                completed += 1
                if self.time_self:
                    print(f"Completed {completed}/{len(genomes)}")

        # 4. Assign back to genomes
        for i, genome in enumerate(genomes):
            genome.latest_outputs = results_map[i]
            
        if self.time_self:
            print(f"#-- Generation finished in {time.time() - start_time:.2f}s --#")

    def shutdown(self):
        for _ in range(self.num_workers):
            self.task_queue.put({'type': 'STOP'})
        for p in self.workers:
            p.join()

    def save_weights_to_disk(self, filepath: str):
        # We only need to save from ONE worker (Worker 0), as they are synchronized
        # This is complex with MP. Easier way: Update the main process Genome, 
        # then load a temporary model to save? Or implement a SAVE command for worker 0.
        print("Note: Save weights implemented via Worker 0 dump not fully detailed here.")