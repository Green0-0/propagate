import os
import sys
import time
import gc
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict, is_dataclass

# External deps (that must exist in your env):
# - vLLM
# - jax/flax (imported lazily inside workers/functions to allow per-process device pinning)
# - torch (for saving weights)

from libs.backend.backend_abc import Backend
from libs.genome import Genome
from libs.optimizers import Optimizer, SimpleOpt

logging.getLogger("vllm.tpu_inference").setLevel(logging.WARNING)


def _serialize_sampler(sampler: Any) -> Dict[str, Any]:
    # vLLM SamplingParams is a dataclass; we serialize to a plain dict for multiprocessing
    if is_dataclass(sampler):
        return asdict(sampler)
    return dict(vars(sampler))


def _get_worker_handle(llm: Any):
    # Same introspection pattern you used; vLLM engine structure can vary by version.
    if hasattr(llm.llm_engine, 'model_executor'):
        return llm.llm_engine.model_executor.driver_worker
    elif hasattr(llm.llm_engine, 'engine') and hasattr(llm.llm_engine.engine, 'model_executor'):
        return llm.llm_engine.engine.model_executor.driver_worker
    else:
        raise AttributeError("Could not find model_executor in vLLM engine structure.")


class _SimpleParam:
    def __init__(self, value):
        self.value = value


def _process_weights_chunked_local(model_worker: Any,
                                   genome_seeds: List[int],
                                   genome_scales: List[float],
                                   mode: str = "perturb",
                                   optimizer_update: bool = False,
                                   chunk_size: int = 10):
    """
    Local version of the weight-chunk processor; runs inside a worker process
    with direct access to model_worker.

    mode: "perturb", "restore", "update" (update adds delta permanently).
    """
    # Lazy imports to ensure env pinning in workers happens first
    import jax
    import jax.numpy as jnp
    from flax import nnx

    state = model_worker.model_runner.state
    flat_state = list(state.flat_state())
    flat_state.sort(key=lambda x: str(x[0]))
    total_params = len(flat_state)

    def get_value_by_path(root, path):
        node = root
        for p in path:
            node = node[p]
        return node

    for i in range(0, total_params, chunk_size):
        current_state = model_worker.model_runner.state
        chunk_items = flat_state[i: i + chunk_size]
        chunk_paths = [item[0] for item in chunk_items]
        chunk_update = {}
        chunk_mappings = {}

        for chunk_rel_idx, path in enumerate(chunk_paths):
            global_param_index = i + chunk_rel_idx
            val = get_value_by_path(current_state, path)

            leaf = val
            if hasattr(val, 'value'):
                leaf = val.value

            sharding = None
            if hasattr(val, 'sharding'):
                sharding = val.sharding
            elif hasattr(leaf, 'sharding'):
                sharding = leaf.sharding

            if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
                aggregate_delta = jnp.zeros(leaf.shape, dtype=jnp.float32)
                for seed, weight in zip(genome_seeds, genome_scales):
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
            chunk_update[key_str] = _SimpleParam(new_val)
            if sharding is not None:
                chunk_mappings[key_str] = (key_str, sharding)

        chunk_state = nnx.State(chunk_update)
        model_worker.sync_weights(
            updated_weights=chunk_state,
            mappings=chunk_mappings,
            transpose_keys={},
            reshard_fn=None
        )
        arrays_to_sync = [p.value for p in chunk_update.values()]
        jax.block_until_ready(arrays_to_sync)

        del chunk_update
        del chunk_mappings
        gc.collect()


def _worker_entry(
    device_idx: int,
    model_name: str,
    sampler_cfg: Dict[str, Any],
    max_model_len: int,
    gpu_memory_utilization: float,
    use_tqdm: bool,
    time_self: bool,
    ctrl_conn: Connection
):
    """
    Worker process entrypoint:
    - Pins to one TPU device (via env) BEFORE importing jax/flax/vllm
    - Builds a vLLM LLM on that device (TP=1)
    - Receives commands: 'generate', 'update', 'save', 'shutdown'
    """
    # Per-process env and pinning
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # keep direct access to internals
    os.environ.pop("TPU_MULTIHOST_BACKEND", None)

    # Pinning to a single TPU chip; try both envs commonly seen.
    os.environ["TPU_VISIBLE_DEVICES"] = str(device_idx)
    os.environ["TPU_VISIBLE_CHIPS"] = str(device_idx)

    # Now safe to import heavy libs
    import jax
    from vllm import LLM, SamplingParams

    sampler = SamplingParams(**sampler_cfg)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    model_worker = _get_worker_handle(llm)

    def _do_generate(genome_seeds: List[int],
                     genome_scales: List[float],
                     inputs_chat: List[List[Dict[str, str]]],
                     suffix: Optional[str]) -> List[str]:
        prompts = []
        tok = llm.get_tokenizer()
        for chat in inputs_chat:
            s = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            if suffix is not None:
                s = s + suffix
            prompts.append(s)

        _process_weights_chunked_local(
            model_worker=model_worker,
            genome_seeds=genome_seeds,
            genome_scales=genome_scales,
            mode="perturb"
        )

        t0 = time.time()
        outs = llm.generate(prompts, sampler, use_tqdm=use_tqdm)
        texts = [o.outputs[0].text for o in outs]
        if time_self:
            print(f"[Worker {device_idx}] Generated batch in {time.time() - t0:.2f}s", flush=True)

        _process_weights_chunked_local(
            model_worker=model_worker,
            genome_seeds=genome_seeds,
            genome_scales=genome_scales,
            mode="restore"
        )
        return texts

    try:
        while True:
            msg = ctrl_conn.recv()
            cmd = msg.get("cmd", None)

            if cmd == "generate":
                job_id = msg["job_id"]
                seeds = msg["seeds"]
                scales = msg["scales"]
                inputs_chat = msg["inputs_chat"]
                suffix = msg.get("suffix", None)
                try:
                    outputs = _do_generate(seeds, scales, inputs_chat, suffix)
                    ctrl_conn.send({"job_id": job_id, "ok": True, "outputs": outputs})
                except Exception as e:
                    ctrl_conn.send({"job_id": job_id, "ok": False, "error": repr(e)})

            elif cmd == "update":
                seeds = msg["seeds"]
                scales = msg["scales"]
                try:
                    _process_weights_chunked_local(
                        model_worker=model_worker,
                        genome_seeds=seeds,
                        genome_scales=scales,
                        mode="update",
                        optimizer_update=True
                    )
                    ctrl_conn.send({"ok": True})
                except Exception as e:
                    ctrl_conn.send({"ok": False, "error": repr(e)})

            elif cmd == "save":
                path = msg["path"]
                try:
                    state = model_worker.model_runner.state
                    flat_state = state.flat_state()
                    cpu_state = {}
                    for path_elems, val in flat_state:
                        key = '.'.join(str(p) for p in path_elems)
                        if hasattr(val, 'value'):
                            cpu_state[key] = jax.device_get(val.value)
                        else:
                            cpu_state[key] = jax.device_get(val)
                    import torch
                    torch.save(cpu_state, path)
                    ctrl_conn.send({"ok": True})
                except Exception as e:
                    ctrl_conn.send({"ok": False, "error": repr(e)})

            elif cmd == "shutdown":
                ctrl_conn.send({"ok": True})
                break

            else:
                ctrl_conn.send({"ok": False, "error": f"Unknown cmd {cmd}"})
    finally:
        try:
            ctrl_conn.close()
        except Exception:
            pass


class VllMTPUPerDeviceBackend(Backend):
    """
    Spawns one vLLM per TPU device (TP=1); schedules genome batches to
    the next available device:
      - Each job: perturb -> generate -> restore on the assigned device
      - Updates are broadcast to all devices (permanent)
    """
    def __init__(
        self,
        model_name: str,
        sampler: Any,  # vLLM SamplingParams or equivalent; serialized to dict
        use_tqdm: bool = False,
        max_model_len: int = 4096,
        time_self: bool = False,
        gpu_memory_utilization: float = 0.6,
        num_devices: Optional[int] = None
    ):
        # Determine device count lazily without importing jax here
        detected_devices = num_devices
        if detected_devices is None:
            # Try reading TPU device count via env hint or default to 8
            detected_devices = int(os.environ.get("TPU_NUM_DEVICES", "8"))

        super().__init__(
            backend_name="vLLM TPU Per-Device Backend",
            model_name=model_name,
            NUM_GPUS=detected_devices,   # naming kept for base class compatibility
            CPUS_PER_GPU=1,
            GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization,
            sampler=sampler,
            use_tqdm=use_tqdm,
            max_model_len=max_model_len,
            time_self=time_self,
        )
        self.num_devices = detected_devices
        self.gpu_memory_utilization = gpu_memory_utilization

        self._ctx = mp.get_context("spawn")
        self._procs: List[mp.Process] = []
        self._conns: List[Connection] = []
        self._sampler_cfg = _serialize_sampler(sampler)

        self._next_job_id = 1

    def startup(self, trainer=None):
        print(f"#-- Initializing per-device vLLM TPU Backend on {self.num_devices} devices --#")

        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.pop("TPU_MULTIHOST_BACKEND", None)

        for dev in range(self.num_devices):
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            p = self._ctx.Process(
                target=_worker_entry,
                args=(
                    dev,
                    self.model_name,
                    self._sampler_cfg,
                    self.max_model_len,
                    self.gpu_memory_utilization,
                    self.use_tqdm,
                    self.time_self,
                    child_conn,
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
            self._conns.append(parent_conn)

        print("#-- vLLM TPU Per-Device Backend Initialized Successfully --#")

    def _broadcast_update(self, seeds: List[int], scales: List[float]):
        for c in self._conns:
            c.send({"cmd": "update", "seeds": seeds, "scales": scales})
        for c in self._conns:
            resp = c.recv()
            if not resp.get("ok", False):
                raise RuntimeError(f"Update failed on a worker: {resp.get('error')}")

    def update(self, optimizer: Optimizer):
        print("#-- Updating Model Weights on all devices (TPU) --#")
        assert isinstance(optimizer, SimpleOpt), "Update expects SimpleOpt optimizer"
        genome = optimizer.get_representative()
        seeds = list(map(int, genome.seeds))
        scales = list(map(float, genome.perturb_scales))
        self._broadcast_update(seeds, scales)

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """
        Dynamic round-robin scheduling:
          - Each job is sent to the first available worker
          - Worker: perturb -> generate -> restore
        """
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        total_jobs = len(genomes)

        next_to_assign = 0
        in_flight: Dict[int, Tuple[int, int]] = {}  # job_id -> (genome_idx, worker_idx)

        t_all = time.time()

        # Prime the workers
        for w_idx in range(self.num_devices):
            if next_to_assign >= total_jobs:
                break
            job_id = self._next_job_id
            self._next_job_id += 1

            g = genomes[next_to_assign]
            inp = inputs[next_to_assign]

            seeds = list(map(int, g.seeds))
            scales = list(map(float, g.perturb_scales))

            self._conns[w_idx].send({
                "cmd": "generate",
                "job_id": job_id,
                "seeds": seeds,
                "scales": scales,
                "inputs_chat": inp,
                "suffix": suffix
            })
            in_flight[job_id] = (next_to_assign, w_idx)
            next_to_assign += 1

        num_done = 0
        while num_done < total_jobs:
            for w_idx, c in enumerate(self._conns):
                if not in_flight:
                    break
                if c.poll(0.01):
                    resp = c.recv()
                    if "job_id" in resp:
                        job_id = resp["job_id"]
                        g_idx, worker_of_resp = in_flight.pop(job_id)
                        if not resp.get("ok", False):
                            raise RuntimeError(f"Worker {worker_of_resp} failed job {job_id}: {resp.get('error')}")
                        genomes[g_idx].latest_outputs = resp["outputs"]
                        num_done += 1

                        if next_to_assign < total_jobs:
                            job_id_new = self._next_job_id
                            self._next_job_id += 1

                            g = genomes[next_to_assign]
                            inp = inputs[next_to_assign]
                            seeds = list(map(int, g.seeds))
                            scales = list(map(float, g.perturb_scales))

                            self._conns[worker_of_resp].send({
                                "cmd": "generate",
                                "job_id": job_id_new,
                                "seeds": seeds,
                                "scales": scales,
                                "inputs_chat": inp,
                                "suffix": suffix
                            })
                            in_flight[job_id_new] = (next_to_assign, worker_of_resp)
                            next_to_assign += 1

        if self.time_self:
            print(f"#-- All genomes generated in {time.time() - t_all:.2f}s --#")

    def save_weights_to_disk(self, filepath: str, worker_index: int = 0):
        """
        Save model weights from one worker (default worker 0).
        """
        print("WARNING: MODEL SAVING ALMOST CERTAINLY DOES NOT WORK PROPERLY WITH TPUS. YOU HAVE BEEN WARNED.")
        print(f"#-- Saving weights from worker {worker_index} to {filepath} --#")
        if worker_index < 0 or worker_index >= len(self._conns):
            raise ValueError("Invalid worker index for save.")
        c = self._conns[worker_index]
        c.send({"cmd": "save", "path": filepath})
        resp = c.recv()
        if not resp.get("ok", False):
            raise RuntimeError(f"Save failed on worker {worker_index}: {resp.get('error')}")
        print("#-- Weights saved successfully --#")

    def shutdown(self):
        for c in self._conns:
            try:
                c.send({"cmd": "shutdown"})
            except Exception:
                pass
        for c in self._conns:
            try:
                if c.poll(2.0):
                    _ = c.recv()
            except Exception:
                pass
            try:
                c.close()
            except Exception:
                pass
        for p in self._procs:
            try:
                p.join(timeout=5.0)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        self._procs.clear()
        self._conns.clear()