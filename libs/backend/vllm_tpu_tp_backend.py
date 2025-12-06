import os
import sys
import time
import gc
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

class VllMTPUTPBackend(Backend):
    def __init__(self, model_name: str, sampler: SamplingParams, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False, gpu_memory_utilization: float = 0.6, tensor_parallel_size: int = 8):
        super().__init__(backend_name="vLLM TPU Backend", model_name=model_name, NUM_GPUS=tensor_parallel_size, CPUS_PER_GPU=1, GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
        self.model_worker = None

    def startup(self, trainer=None):
        """Initialize the vLLM TPU engine."""
        print(f"#-- Initializing vLLM TPU Backend (TP={self.tensor_parallel_size}) --#")

        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.pop("TPU_MULTIHOST_BACKEND", None)

        self.llm = LLM(model=self.model_name, tensor_parallel_size=self.tensor_parallel_size, trust_remote_code=True, dtype="bfloat16", max_model_len=self.max_model_len, gpu_memory_utilization=self.gpu_memory_utilization)

        self.model_worker = self._get_worker()
        print("#-- vLLM TPU Backend Initialized Successfully --#")

    def _get_worker(self):
        if hasattr(self.llm.llm_engine, 'model_executor'):
            return self.llm.llm_engine.model_executor.driver_worker
        elif hasattr(self.llm.llm_engine, 'engine') and hasattr(self.llm.llm_engine.engine, 'model_executor'):
            return self.llm.llm_engine.engine.model_executor.driver_worker
        else:
            raise AttributeError("Could not find model_executor in vLLM engine structure.")

    def _process_weights_chunked(self, genome: Genome = None, optimizer: Optimizer = None, mode: str = "perturb"):
        """
        Handles weight perturbation, restoration, and updates in chunks.
        FIXED: Uses parameter index instead of hash(path) for stable RNG.
        """
        worker = self.model_worker
        
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
            arrays_to_sync = [p.value for p in chunk_update.values()]
            jax.block_until_ready(arrays_to_sync)

            del chunk_update
            del chunk_mappings
            gc.collect()

    def update(self, optimizer: Optimizer):
        """Update the model permanently with a genome as the source."""
        print("#-- Updating Model Weights (TPU) --#")
        self._process_weights_chunked(optimizer=optimizer, mode="update")

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """
        Generate outputs serially for each genome on the TPU.
        Perturb -> Generate -> Restore.
        """
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        
        # Pre-process prompts
        prompts = []
        for i in inputs:
            prompt_genome = []
            for j in i:
                s = self.llm.get_tokenizer().apply_chat_template(j, tokenize=False, add_generation_prompt=True)
                if suffix is not None:
                    s = s + suffix
                prompt_genome.append(s)
            prompts.append(prompt_genome)

        start_time_all = time.time()

        for idx, (genome, prompt_set) in enumerate(zip(genomes, prompts)):
            self._process_weights_chunked(genome=genome, mode="perturb")
            
            gen_start = time.time()
            outputs = self.llm.generate(prompt_set, self.sampler, use_tqdm=self.use_tqdm)
            
            genome.latest_outputs = [o.outputs[0].text for o in outputs]
            
            if self.time_self:
                print(f"#-- Genome {idx+1}/{len(genomes)} generated in {time.time() - gen_start:.2f}s --#")

            self._process_weights_chunked(genome=genome, mode="restore")

        if self.time_self:
            print(f"#-- All genomes generated in {time.time() - start_time_all:.2f}s --#")

    def save_weights_to_disk(self, filepath: str):
        """Save the model weights."""
        print("WARNING: MODEL SAVING ALMOST CERTAINLY DOES NOT WORK PROPERLY WITH TPUS. YOU HAVE BEEN WARNED.")
        
        print(f"#-- Saving weights to {filepath} --#")
        state = self.model_worker.model_runner.state
        flat_state = state.flat_state()
        
        # Convert to a dictionary of CPU numpy arrays
        cpu_state = {}
        for path, val in flat_state:
            key = '.'.join(str(p) for p in path)
            if hasattr(val, 'value'):
                cpu_state[key] =  jax.device_get(val.value)
            else:
                cpu_state[key] = jax.device_get(val)
                
        # Save as pickle/msgpack or via torch (common format)
        import torch
        torch.save(cpu_state, filepath)
        print("#-- Weights saved successfully --#")