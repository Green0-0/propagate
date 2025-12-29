import os
from platform import node
import sys
import time
import gc
from typing import List, Dict, Any
import logging

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.pop("TPU_MULTIHOST_BACKEND", None)

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Param 
from vllm import LLM, SamplingParams

from propagate.backend.backend_abc import Backend
from propagate.genome import Genome
from propagate.optimizers import Optimizer, SimpleOpt

logging.getLogger("vllm.tpu_inference").setLevel(logging.WARNING)

class VllMTPUTPBackend(Backend):
    def __init__(self, model_name: str, sampler: SamplingParams, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False, gpu_memory_utilization: float = 0.6, tensor_parallel_size: int = 8):
        super().__init__(backend_name="vLLM TPU Backend", model_name=model_name, NUM_GPUS=tensor_parallel_size, CPUS_PER_GPU=1, GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization, sampler=sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
        self.model_worker = None
        print("### WARNING: EXPERIMENTAL, MAY NOT WORK ###")

    def startup(self, trainer=None):
        """Initialize the vLLM TPU engine."""
        print(f"#-- Initializing vLLM TPU Backend (TP={self.tensor_parallel_size}) --#")

        self.llm = LLM(model=self.model_name, tensor_parallel_size=self.tensor_parallel_size, trust_remote_code=True, dtype="bfloat16", max_model_len=self.max_model_len, gpu_memory_utilization=self.gpu_memory_utilization)

        print("#-- vLLM TPU Backend Initialized Successfully --#")

    def is_trainable_param(self, path, node):
        if not isinstance(node, Param):
            return False
        key_str = '.'.join(map(str, path))
        banned = ('rotary', 'kv_cache','inv_freq', 'cos_cached', 'sin_cached')
        if any(b in key_str.lower() for b in banned):
            return False
        return True 
    
    def _process_weights_chunked(self, genome: Genome = None, optimizer: Optimizer = None, mode: str = "perturb"):
        worker = self.llm.llm_engine.model_executor.driver_worker
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
                    aggregate_delta = jnp.zeros(leaf.shape, dtype=leaf.dtype)
                    #TODO: Future optimization: Use jax jit to speedup and reduce memory here
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
        for idk, i in enumerate(inputs):
            prompt_genome = []
            input_genome_content = []
            for j in i:
                input_genome_content.append(j[-1]['content'])
                s = self.llm.get_tokenizer().apply_chat_template(j, tokenize=False, add_generation_prompt=True)
                if suffix is not None:
                    s = s + suffix
                prompt_genome.append(s)
            prompts.append(prompt_genome)
            genomes[idk].latest_inputs = input_genome_content

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