import os
from platform import node
import sys
import time
import gc
from typing import List, Dict, Any, Optional
import logging

# Environment setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # keep engine internals accessible
os.environ.pop("TPU_MULTIHOST_BACKEND", None)

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Param

from vllm import LLM, SamplingParams

from libs.backend.backend_abc import Backend
from libs.genome import Genome
from libs.optimizers import Optimizer, SimpleOpt

logging.getLogger("vllm.tpu_inference").setLevel(logging.WARNING)


class VllMTPUDPBackend(Backend):
    """
    Data-parallel TPU backend:
    - Spawns one independent vLLM instance per TPU device (tensor_parallel_size=1)
    - Schedules genomes round-robin: perturb -> generate -> restore on first available worker
    - update() is applied universally across all workers
    - Keeps VLLM_ENABLE_V1_MULTIPROCESSING=0 for driver_worker access
    """

    def __init__(
        self,
        model_name: str,
        sampler: SamplingParams,
        use_tqdm: bool = False,
        max_model_len: int = 4096,
        time_self: bool = False,
        gpu_memory_utilization: float = 0.6,
        num_devices: Optional[int] = None,  # number of TPU devices to use; default = all local devices
    ):
        # Detect TPU devices
        self._all_devices = list(jax.local_devices())
        if num_devices is None:
            num_devices = len(self._all_devices)
        else:
            num_devices = min(num_devices, len(self._all_devices))
        self.num_devices = num_devices

        super().__init__(
            backend_name="vLLM TPU Data-Parallel Backend",
            model_name=model_name,
            NUM_GPUS=self.num_devices,
            CPUS_PER_GPU=1,
            GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization,
            sampler=sampler,
            use_tqdm=use_tqdm,
            max_model_len=max_model_len,
            time_self=time_self,
        )

        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm_workers: List[VllMTPUDPBackend._DPWorker] = []
        self._startup_done = False
        self._lock = threading.Lock()

    class _DPWorker:
        def __init__(
            self,
            device_index: int,
            device,
            model_name: str,
            sampler: SamplingParams,
            max_model_len: int,
            gpu_memory_utilization: float,
            use_tqdm: bool,
            time_self: bool,
        ):
            self.device_index = device_index
            self.device = device
            self.model_name = model_name
            self.sampler = sampler
            self.max_model_len = max_model_len
            self.gpu_memory_utilization = gpu_memory_utilization
            self.use_tqdm = use_tqdm
            self.time_self = time_self

            self.llm: Optional[LLM] = None

        def startup(self):
            # Create a single-device vLLM instance on this TPU device.
            # VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps the engine in-process so driver_worker is accessible.
            print(f"#-- [Worker {self.device_index}] Initializing vLLM on {self.device} --#")
            with jax.default_device(self.device):
                self.llm = LLM(
                    model=self.model_name,
                    tensor_parallel_size=1,              # one engine per TPU device, no TP sharding
                    trust_remote_code=True,
                    dtype="bfloat16",
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                )
            print(f"#-- [Worker {self.device_index}] Ready --#")

        @staticmethod
        def is_trainable_param(path, node):
            if not isinstance(node, Param):
                return False
            key_str = '.'.join(map(str, path))
            banned = ('rotary', 'kv_cache', 'inv_freq', 'cos_cached', 'sin_cached')
            if any(b in key_str.lower() for b in banned):
                return False
            return True

        def _process_weights_chunked(self, genome: Genome = None, optimizer: Optimizer = None, mode: str = "perturb"):
            assert self.llm is not None, "Worker not started"
            # Access internal driver worker
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
            # Prepare random keys for deterministic perturb/restore
            keys = [(jax.random.PRNGKey(int(seed)), weight) for seed, weight in zip(genome.seeds, genome.perturb_scales)]

            for i in range(0, total_params, chunk_size):
                current_state = worker.model_runner.state
                chunk_paths = [item[0] for item in flat_state[i:i + chunk_size]]
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
                        # Aggregate Gaussian noise from genome seeds/weights
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
                worker.sync_weights(
                    updated_weights=chunk_state,
                    mappings=chunk_mappings,
                    transpose_keys={},
                    reshard_fn=None
                )
                del chunk_update
                del chunk_mappings
                gc.collect()

        def run_job(self, genome: Genome, prompt_set: List[str]) -> List[str]:
            # perturb -> generate -> restore, all on this worker's LLM
            self._process_weights_chunked(genome=genome, mode="perturb")
            start = time.time()
            outputs = self.llm.generate(prompt_set, self.sampler, use_tqdm=self.use_tqdm)
            texts = [o.outputs[0].text for o in outputs]
            if self.time_self:
                print(f"#-- [Worker {self.device_index}] Generated batch in {time.time() - start:.2f}s --#")
            self._process_weights_chunked(genome=genome, mode="restore")
            gc.collect()
            return texts

        def update(self, optimizer: Optimizer):
            self._process_weights_chunked(optimizer=optimizer, mode="update")

    def startup(self, trainer=None):
        print(f"#-- Initializing vLLM TPU Data-Parallel Backend with {self.num_devices} device(s) --#")
        self.llm_workers = []
        for idx in range(self.num_devices):
            dev = self._all_devices[idx]
            w = VllMTPUDPBackend._DPWorker(
                device_index=idx,
                device=dev,
                model_name=self.model_name,
                sampler=self.sampler,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                use_tqdm=self.use_tqdm,
                time_self=self.time_self,
            )
            w.startup()
            self.llm_workers.append(w)
        self._startup_done = True
        print("#-- vLLM TPU DP Backend Initialized Successfully --#")

    def _ensure_started(self):
        if not self._startup_done:
            raise RuntimeError("Call startup() before using the backend.")

    def update(self, optimizer: Optimizer):
        self._ensure_started()
        print("#-- Updating Model Weights on All Devices (TPU DP) --#")
        for w in self.llm_workers:
            w.update(optimizer)

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """
        Data-parallel generation:
        - Pre-tokenize prompts once (using worker 0's tokenizer)
        - Round-robin schedule genomes to available workers:
          Each worker: perturb -> generate -> restore for its assigned genome.
        """
        self._ensure_started()
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."

        # Pre-process prompts once using worker 0 tokenizer
        tokenizer = self.llm_workers[0].llm.get_tokenizer()
        prompts: List[List[str]] = []
        for i in inputs:
            prompt_genome = []
            for j in i:
                s = tokenizer.apply_chat_template(j, tokenize=False, add_generation_prompt=True)
                if suffix is not None:
                    s = s + suffix
                prompt_genome.append(s)
            prompts.append(prompt_genome)

        start_time_all = time.time()

        # Round-robin scheduling over worker pool with threads
        results = [None] * len(genomes)
        job_indices = list(range(len(genomes)))
        job_lock = threading.Lock()
        next_job_ptr = {"idx": 0}  # mutable box for closure

        def worker_loop(worker: VllMTPUDPBackend._DPWorker):
            while True:
                with job_lock:
                    if next_job_ptr["idx"] >= len(job_indices):
                        return
                    job_id = job_indices[next_job_ptr["idx"]]
                    next_job_ptr["idx"] += 1
                genome = genomes[job_id]
                prompt_set = prompts[job_id]
                gen_start = time.time()
                texts = worker.run_job(genome, prompt_set)
                genome.latest_outputs = texts
                results[job_id] = texts
                if self.time_self:
                    print(f"#-- Genome {job_id+1}/{len(genomes)} completed on worker {worker.device_index} in {time.time() - gen_start:.2f}s --#")

        with ThreadPoolExecutor(max_workers=self.num_devices) as ex:
            futures = [ex.submit(worker_loop, w) for w in self.llm_workers]
            for f in as_completed(futures):
                _ = f.result()

        if self.time_self:
            print(f"#-- All genomes generated in {time.time() - start_time_all:.2f}s --#")

    def save_weights_to_disk(self, filepath: str):
        """
        Save weights from worker 0 (all workers are kept in sync via universal updates).
        Note: Saving on TPU with JAX can be fragile; this mirrors the original warning.
        """
        self._ensure_started()
        print("WARNING: MODEL SAVING ON TPU MAY NOT WORK RELIABLY. PROCEED WITH CAUTION.")
        print(f"#-- Saving weights to {filepath} (from worker 0) --#")

        worker0 = self.llm_workers[0].llm.llm_engine.model_executor.driver_worker
        state = worker0.model_runner.state
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