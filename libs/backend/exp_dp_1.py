import os
from platform import node
import sys
import time
import gc
from typing import List, Dict, Any, Tuple
import logging

# IMPORTANT: Keep this to allow access to internal state
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# TPU/JAX memory behavior
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ.pop("TPU_MULTIHOST_BACKEND", None)

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Param

import ray
from vllm import LLM, SamplingParams

from libs.backend.backend_abc import Backend
from libs.genome import Genome
from libs.optimizers import Optimizer, SimpleOpt

logging.getLogger("vllm.tpu_inference").setLevel(logging.WARNING)


@ray.remote(num_cpus=1, max_concurrency=1)
class TPUWorkerActor:
    def __init__(
        self,
        model_name: str,
        sampler: SamplingParams,
        device_index: int,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.6,
        use_tqdm: bool = False,
        time_self: bool = False,
        dtype: str = "bfloat16",
    ):
        # Re-assert critical env vars in the actor process
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        os.environ.pop("TPU_MULTIHOST_BACKEND", None)

        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_tqdm = use_tqdm
        self.time_self = time_self
        self.sampler = sampler
        self.dtype = dtype
        self.device_index = device_index

        # Pick the TPU device for this actor and create the model on it
        tpu_devices = jax.devices("TPU")
        if not tpu_devices:
            # Fall back to whatever JAX sees
            tpu_devices = jax.devices()
        assert (
            0 <= device_index < len(tpu_devices)
        ), f"Invalid device_index {device_index}, available devices: {len(tpu_devices)}"
        self.device = tpu_devices[device_index]

        # Build the LLM with tensor_parallel_size=1 so it stays on this device
        with jax.default_device(self.device):
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                trust_remote_code=True,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )

    @staticmethod
    def is_trainable_param(path, node):
        if not isinstance(node, Param):
            return False
        key_str = ".".join(map(str, path))
        banned = ("rotary", "kv_cache", "inv_freq", "cos_cached", "sin_cached")
        if any(b in key_str.lower() for b in banned):
            return False
        return True

    def _process_weights_chunked(
        self, genome: Genome = None, optimizer: Optimizer = None, mode: str = "perturb"
    ):
        # Access vLLM internals (requires V1 multiprocessing disabled)
        worker = self.llm.llm_engine.model_executor.driver_worker
        state = worker.model_runner.state
        flat_state = list(state.flat_state())
        flat_state.sort(key=lambda x: str(x[0]))
        total_params = len(flat_state)
        chunk_size = 10

        class SimpleParam:
            def __init__(self, value):
                self.value = value

        # "update" mode uses a representative genome if given optimizer is SimpleOpt
        if mode == "update":
            if isinstance(optimizer, SimpleOpt):
                genome = optimizer.get_representative()
            # else: allow caller to pass genome directly

        # Prepare PRNG keys/weights
        if genome is not None:
            keys = [
                (jax.random.PRNGKey(int(seed)), weight)
                for seed, weight in zip(genome.seeds, genome.perturb_scales)
            ]
        else:
            keys = []

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
                # Non-trainable param passthrough
                if not self.is_trainable_param(path, val):
                    leaf = val.value if hasattr(val, "value") else val
                    key_str = ".".join(str(k) for k in path)
                    chunk_update[key_str] = SimpleParam(leaf)
                    continue

                leaf = val
                sharding = None
                if hasattr(val, "value"):
                    leaf = val.value
                if hasattr(val, "sharding"):
                    sharding = val.sharding
                elif hasattr(leaf, "sharding"):
                    sharding = leaf.sharding

                if (
                    keys
                    and isinstance(leaf, jax.Array)
                    and jnp.issubdtype(leaf.dtype, jnp.floating)
                ):
                    aggregate_delta = jnp.zeros(leaf.shape, dtype=leaf.dtype)
                    # Accumulate perturbations
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

                key_str = ".".join(str(k) for k in path)
                chunk_update[key_str] = SimpleParam(new_val)
                if sharding is not None:
                    chunk_mappings[key_str] = (key_str, sharding)

            chunk_state = nnx.State(chunk_update)
            # Sync chunk into the model
            worker.sync_weights(
                updated_weights=chunk_state,
                mappings=chunk_mappings,
                transpose_keys={},
                reshard_fn=None,
            )
            del chunk_update
            del chunk_mappings
            gc.collect()

    def update_with_genome(self, genome: Genome = None, optimizer: Optimizer = None):
        # Permanently update this actor's model
        self._process_weights_chunked(genome=genome, optimizer=optimizer, mode="update")

    def generate_for_inputs(
        self,
        inputs_for_genome: List[List[Dict[str, str]]],
        suffix: str,
        genome: Genome,
    ) -> List[str]:
        # Build prompts from this actor's tokenizer
        prompts = []
        tok = self.llm.get_tokenizer()
        for conv in inputs_for_genome:
            s = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            if suffix is not None:
                s = s + suffix
            prompts.append(s)

        # Perturb -> Generate -> Restore
        if self.time_self:
            t0 = time.time()
        self._process_weights_chunked(genome=genome, mode="perturb")
        outputs = self.llm.generate(prompts, self.sampler, use_tqdm=self.use_tqdm)
        texts = [o.outputs[0].text for o in outputs]
        self._process_weights_chunked(genome=genome, mode="restore")
        if self.time_self:
            print(
                f"#-- Actor device {self.device_index} generated {len(texts)} samples in {time.time()-t0:.2f}s --#"
            )
        return texts

    def save_weights_to_disk(self, filepath: str):
        print(
            "WARNING: MODEL SAVING ALMOST CERTAINLY DOES NOT WORK PROPERLY WITH TPUS. YOU HAVE BEEN WARNED."
        )
        print(f"#-- Saving weights to {filepath} (actor device {self.device_index}) --#")
        worker = self.llm.llm_engine.model_executor.driver_worker
        state = worker.model_runner.state
        flat_state = state.flat_state()

        cpu_state = {}
        for path, val in flat_state:
            key = ".".join(str(p) for p in path)
            if hasattr(val, "value"):
                cpu_state[key] = jax.device_get(val.value)
            else:
                cpu_state[key] = jax.device_get(val)

        import torch

        torch.save(cpu_state, filepath)
        print("#-- Weights saved successfully --#")


class VllMTPUDPBackend(Backend):
    def __init__(
        self,
        model_name: str,
        sampler: SamplingParams,
        use_tqdm: bool = False,
        max_model_len: int = 4096,
        time_self: bool = False,
        gpu_memory_utilization: float = 0.6,
        num_devices: int = None,  # number of TPU devices (default: all)
        dtype: str = "bfloat16",
    ):
        # NUM_GPUS is informational for your framework; each actor uses 1 TPU device
        # so we report num_devices as NUM_GPUS-like resource.
        tpu_count = len(jax.devices("TPU")) or len(jax.devices())
        if num_devices is None:
            num_devices = tpu_count
        self.num_devices = max(1, min(num_devices, tpu_count))

        super().__init__(
            backend_name="vLLM TPU DP Backend",
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
        self.dtype = dtype

        self.actors: List[ray.actor.ActorHandle] = []

    def startup(self, trainer=None):
        """Initialize Ray and spawn one model per TPU device."""
        print(
            f"#-- Initializing vLLM TPU DP Backend (num_devices={self.num_devices}) --#"
        )
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)

        # Spawn actors, each pinned to a TPU device (index 0..num_devices-1)
        self.actors = []
        for device_idx in range(self.num_devices):
            actor = TPUWorkerActor.remote(
                model_name=self.model_name,
                sampler=self.sampler,
                device_index=device_idx,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                use_tqdm=self.use_tqdm,
                time_self=self.time_self,
                dtype=self.dtype,
            )
            self.actors.append(actor)
        print("#-- vLLM TPU DP Backend Initialized Successfully --#")

    def update(self, optimizer: Optimizer):
        """Broadcast a permanent update to all actor models."""
        print("#-- Updating Model Weights (TPU DP - all devices) --#")
        # Prefer sending the representative genome rather than the whole optimizer state
        genome_rep = None
        if isinstance(optimizer, SimpleOpt):
            genome_rep = optimizer.get_representative()

        # Update on all actors
        futures = []
        for a in self.actors:
            if genome_rep is not None:
                futures.append(a.update_with_genome.remote(genome=genome_rep))
            else:
                futures.append(a.update_with_genome.remote(optimizer=optimizer))
        ray.get(futures)

    def generate_outputs(
        self,
        genomes: List[Genome],
        suffix: str,
        inputs: List[List[List[Dict[str, str]]]],
    ):
        """
        Distribute genomes across per-TPU actors:
        Each actor: Perturb -> Generate -> Restore
        Round-robin scheduling via ray.wait to keep devices busy.
        """
        assert len(genomes) == len(
            inputs
        ), "Number of genomes must match number of input sets."

        start_time_all = time.time()

        # Schedule initial tasks up to num_devices
        next_idx = 0
        in_flight: Dict[ray.ObjectRef, Tuple[int, ray.actor.ActorHandle]] = {}

        def schedule_one(idx: int, actor: ray.actor.ActorHandle):
            g = genomes[idx]
            inp = inputs[idx]
            return actor.generate_for_inputs.remote(inp, suffix, g)

        # Prime the pipeline
        for a in self.actors:
            if next_idx < len(genomes):
                ref = schedule_one(next_idx, a)
                in_flight[ref] = (next_idx, a)
                next_idx += 1

        # Collect and keep scheduling until all done
        finished = 0
        while in_flight:
            done_refs, _ = ray.wait(list(in_flight.keys()), num_returns=1)
            ref = done_refs[0]
            idx, actor = in_flight.pop(ref)
            texts = ray.get(ref)

            # Store outputs on the corresponding genome
            genomes[idx].latest_outputs = texts
            finished += 1
            if self.time_self:
                print(
                    f"#-- Genome {finished}/{len(genomes)} finished on actor --#"
                )

            # Assign next task to the freed actor
            if next_idx < len(genomes):
                new_ref = schedule_one(next_idx, actor)
                in_flight[new_ref] = (next_idx, actor)
                next_idx += 1

        if self.time_self:
            print(
                f"#-- All genomes generated in {time.time() - start_time_all:.2f}s --#"
            )

    def save_weights_to_disk(self, filepath: str, actor_index: int = 0):
        """
        Save the model weights of a single actor (default: first actor).
        WARNING: As with TPUs generally, this may not produce a fully portable or correct file.
        """
        print(
            "WARNING: MODEL SAVING ALMOST CERTAINLY DOES NOT WORK PROPERLY WITH TPUS. YOU HAVE BEEN WARNED."
        )
        assert (
            0 <= actor_index < len(self.actors)
        ), f"actor_index out of range [0, {len(self.actors)-1}]"
        ray.get(self.actors[actor_index].save_weights_to_disk.remote(filepath))
        print("#-- Weights saved successfully --#")