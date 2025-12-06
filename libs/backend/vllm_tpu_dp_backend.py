import os
import sys
import time
import gc
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import logging

# JAX / Flax imports
import jax
import jax.numpy as jnp
from flax import nnx
from vllm import LLM, SamplingParams

from libs.backend.backend_abc import Backend
from libs.genome import Genome
from libs.optimizers import Optimizer, SimpleOpt

logging.getLogger("vllm.tpu_inference").setLevel(logging.WARNING)

class VllmTPUDPBackend(Backend):
    def __init__(self, model_name: str, sampler: SamplingParams, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False, gpu_memory_utilization: float = 0.6, num_replicas: int = 8):
        # Note: tensor_parallel_size is hardcoded to 1 because we are doing Data Parallelism (1 model per chip)
        super().__init__(backend_name="vLLM TPU DP Backend", model_name=model_name, NUM_GPUS=num_replicas, CPUS_PER_GPU=1, GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
        
        self.num_replicas = num_replicas
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # Lists to hold instances for each device
        self.llms = [] 
        self.model_workers = []
        self.tpu_devices = []

    def startup(self, trainer=None):
        """Initialize one vLLM engine per TPU device."""
        print(f"#-- Initializing vLLM TPU Backend (DP Mode: {self.num_replicas} Replicas) --#")

        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.pop("TPU_MULTIHOST_BACKEND", None)

        # 1. Detect TPU devices
        try:
            self.tpu_devices = jax.devices()
            print(f"Detected JAX devices: {self.tpu_devices}")
            if len(self.tpu_devices) < self.num_replicas:
                print(f"WARNING: Requested {self.num_replicas} replicas but only found {len(self.tpu_devices)} devices.")
                self.num_replicas = len(self.tpu_devices)
        except Exception as e:
            raise RuntimeError(f"Failed to detect TPU devices via JAX: {e}")

        # 2. Instantiate an LLM for each device
        # We assume standard vLLM behavior where TP=1 allocates to available devices.
        # To ensure strict placement in a single process, we use jax.default_device context 
        # hoping vLLM respects the current JAX context or handles sequential allocation.
        
        for i in range(self.num_replicas):
            print(f"Initializing Replica {i} on device {self.tpu_devices[i]}...")
            
            # Force JAX operations for this initialization to target the specific device
            with jax.default_device(self.tpu_devices[i]):
                llm_instance = LLM(
                    model=self.model_name,
                    tensor_parallel_size=1, # No sharding, model fits on one chip
                    trust_remote_code=True,
                    dtype="bfloat16",
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    seed=42 # Ensure deterministic initialization across replicas
                )
                
            self.llms.append(llm_instance)
            self.model_workers.append(self._get_worker(llm_instance))
            
            # Aggressive GC to ensure we don't blow host RAM during initialization loop
            gc.collect()

        print("#-- vLLM TPU DP Backend Initialized Successfully --#")

    def _get_worker(self, llm_instance):
        """Extracts the worker from a specific LLM instance."""
        if hasattr(llm_instance.llm_engine, 'model_executor'):
            return llm_instance.llm_engine.model_executor.driver_worker
        elif hasattr(llm_instance.llm_engine, 'engine') and hasattr(llm_instance.llm_engine.engine, 'model_executor'):
            return llm_instance.llm_engine.engine.model_executor.driver_worker
        else:
            raise AttributeError("Could not find model_executor in vLLM engine structure.")

    def _process_weights_chunked(self, genome: Genome = None, optimizer: Optimizer = None, mode: str = "perturb", replica_idx: int = 0):
        """
        Handles weight operations on a SPECIFIC replica.
        """
        worker = self.model_workers[replica_idx]
        target_device = self.tpu_devices[replica_idx]
        
        state = worker.model_runner.state
        flat_state = list(state.flat_state())
        flat_state.sort(key=lambda x: str(x[0]))
        
        total_params = len(flat_state)
        chunk_size = 15 # Increased chunk size slightly as we are single-device per model

        if mode == "update":
            assert isinstance(optimizer, SimpleOpt)
            genome = optimizer.get_representative()
        
        # Execute operations within the context of the specific device to avoid copy overhead
        with jax.default_device(target_device):
            for i in range(0, total_params, chunk_size):
                current_state = worker.model_runner.state
                chunk_paths = [item[0] for item in flat_state[i : i + chunk_size]]
                
                chunk_update = {}
                chunk_mappings = {}

                def get_value_by_path(root, path):
                    node = root
                    for p in path:
                        node = node[p]
                    return node

                for path in chunk_paths:
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
                        
                        # Generate noise directly on the target device
                        # We use the same seed, but JAX needs the PRNG key on the specific device 
                        # to generate noise on that device without transfer.
                        
                        aggregate_delta = jnp.zeros_like(leaf)

                        for seed, weight in zip(genome.seeds, genome.perturb_scales):
                            # Move key to target device explicitly
                            key = jax.random.PRNGKey(int(seed))
                            key = jax.device_put(key, target_device)
                            
                            path_hash = hash(tuple(path))
                            key = jax.random.fold_in(key, path_hash)
                            
                            noise = jax.random.normal(key, leaf.shape, dtype=leaf.dtype)

                            if mode == "restore":
                                aggregate_delta = aggregate_delta - (noise * weight)
                            else:
                                aggregate_delta = aggregate_delta + (noise * weight)
                        
                        if mode == "restore":
                            new_val = leaf - aggregate_delta
                        else:
                            new_val = leaf + aggregate_delta
                    else:
                        new_val = leaf

                    key_str = '.'.join(str(k) for k in path)
                    
                    class SimpleParam:
                        def __init__(self, value):
                            self.value = value

                    chunk_update[key_str] = SimpleParam(new_val)
                    if sharding is not None:
                        chunk_mappings[key_str] = (key_str, sharding)

                chunk_state = nnx.State(chunk_update)
                worker.sync_weights(updated_weights=chunk_state, mappings=chunk_mappings, transpose_keys={}, reshard_fn=None)
                
                # Block only the specific arrays on this device
                arrays_to_sync = [p.value for p in chunk_update.values()]
                jax.block_until_ready(arrays_to_sync)

                del chunk_update
                del chunk_mappings
                # Local GC within thread usually not effective, but kept for consistency
                
    def update(self, optimizer: Optimizer):
        """
        Update ALL replicas uniformly. 
        This is a blocking operation that runs sequentially or parallel over devices to keep them in sync.
        """
        print("#-- Updating All Model Replicas (Universal) --#")
        
        # Parallel update to save time
        with ThreadPoolExecutor(max_workers=self.num_replicas) as executor:
            futures = [
                executor.submit(self._process_weights_chunked, None, optimizer, "update", i) 
                for i in range(self.num_replicas)
            ]
            for f in futures:
                f.result()
                
        gc.collect()

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """
        Round-robin generation using a Queue and ThreadPool.
        """
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        
        # 1. Prepare Work Queue
        # Each item is a tuple: (genome_index, genome, input_batch)
        work_queue = queue.Queue()
        
        # Pre-process prompts
        processed_prompts = []
        # Use first LLM for tokenizer (assumed identical)
        tokenizer = self.llms[0].get_tokenizer()
        
        for input_batch in inputs:
            batch_prompts = []
            for msg_list in input_batch:
                s = tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=True)
                if suffix:
                    s = s + suffix
                batch_prompts.append(s)
            processed_prompts.append(batch_prompts)

        for i, (genome, prompts) in enumerate(zip(genomes, processed_prompts)):
            work_queue.put((i, genome, prompts))

        start_time_all = time.time()
        
        # 2. Define Worker Function
        def device_worker(replica_idx):
            """
            This function runs in a separate thread, dedicated to one TPU device.
            It pulls from the global queue until empty.
            """
            llm = self.llms[replica_idx]
            
            while not work_queue.empty():
                try:
                    # Non-blocking get with short timeout to handle race conditions at end of queue
                    idx, genome, prompts = work_queue.get(block=False)
                except queue.Empty:
                    break

                # A. Perturb
                self._process_weights_chunked(genome=genome, mode="perturb", replica_idx=replica_idx)
                
                # B. Generate
                # Note: use_tqdm=False to prevent 8 interleaved progress bars
                outputs = llm.generate(prompts, self.sampler, use_tqdm=False)
                
                # Store results
                # Lockless is fine here because each genome is owned by one thread at a time
                genome.latest_outputs = [o.outputs[0].text for o in outputs]
                
                if self.time_self:
                    print(f"[Device {replica_idx}] Finished Genome {idx}")

                # C. Restore
                self._process_weights_chunked(genome=genome, mode="restore", replica_idx=replica_idx)
                
                work_queue.task_done()

        # 3. Launch ThreadPool
        print(f"#-- Starting Parallel Generation on {self.num_replicas} Devices --#")
        with ThreadPoolExecutor(max_workers=self.num_replicas) as executor:
            futures = [executor.submit(device_worker, i) for i in range(self.num_replicas)]
            
            # Wait for all to finish
            for f in futures:
                f.result() # This re-raises exceptions if they occurred in threads

        if self.time_self:
            print(f"#-- All genomes generated in {time.time() - start_time_all:.2f}s --#")
            
    def save_weights_to_disk(self, filepath: str):
        # We only need to save from the first replica, as they are kept in sync
        print(f"#-- Saving weights from Replica 0 to {filepath} --#")
        worker = self.model_workers[0]
        state = worker.model_runner.state
        flat_state = state.flat_state()
        
        cpu_state = {}
        for path, val in flat_state:
            key = '.'.join(str(p) for p in path)
            if hasattr(val, 'value'):
                cpu_state[key] = jax.device_get(val.value)
            else:
                cpu_state[key] = jax.device_get(val)
                
        import torch
        torch.save(cpu_state, filepath)
        print("#-- Weights saved successfully --#")