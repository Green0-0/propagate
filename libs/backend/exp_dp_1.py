import os
import sys
import time
import gc
import logging
import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait
from typing import List, Dict, Any, Optional

# External deps that should be safe to import in the parent process.
# DO NOT import jax/flax/vllm here to avoid binding devices in the parent.
from libs.backend.backend_abc import Backend
from libs.genome import Genome
from libs.optimizers import Optimizer, SimpleOpt

logging.getLogger("vllm.tpu_inference").setLevel(logging.WARNING)


def _worker_loop(
    conn,
    device_id: int,
    model_name: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    use_tqdm: bool,
    time_self: bool,
    chunk_size: int = 10,
):
    """
    Worker process:
    - Pins itself to a single TPU device via JAX_VISIBLE_DEVICES.
    - Creates a vLLM LLM with tensor_parallel_size=1.
    - Provides commands:
        set_sampler(sampler)
        generate(task_id, genome_dict, inputs, suffix)
        update(genome_dict)
        save_weights(filepath)
        shutdown()
    """
    try:
        # Must set env BEFORE importing JAX/vLLM inside this process
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Prevent multi-host incorrect settings
        os.environ.pop("TPU_MULTIHOST_BACKEND", None)
        # Restrict this process to a single TPU device
        os.environ["JAX_VISIBLE_DEVICES"] = str(device_id)

        # Delayed imports to respect the env above
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from vllm import LLM

        # Initialize vLLM on a single device (no tensor-parallel sharding)
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # Helper to access driver worker
        def _get_worker():
            if hasattr(llm.llm_engine, 'model_executor'):
                return llm.llm_engine.model_executor.driver_worker
            elif hasattr(llm.llm_engine, 'engine') and hasattr(llm.llm_engine.engine, 'model_executor'):
                return llm.llm_engine.engine.model_executor.driver_worker
            else:
                raise AttributeError("Could not find model_executor in vLLM engine structure.")

        model_worker = _get_worker()

        # We keep sampler inside worker; set via 'set_sampler' command
        sampler = None

        # Processing weights in chunks (perturb, restore, update)
        def _process_weights_chunked(genome_dict: Optional[Dict[str, Any]] = None, mode: str = "perturb"):
            """
            Handles weight perturbation, restoration, and permanent updates in chunks.
            Uses parameter index instead of hash(path) for stable RNG.
            genome_dict: {'seeds': List[int], 'perturb_scales': List[float]}
            mode: 'perturb', 'restore', or 'update'
            """
            if genome_dict is None:
                raise ValueError("genome_dict required for _process_weights_chunked")

            seeds = genome_dict.get("seeds", [])
            scales = genome_dict.get("perturb_scales", [])
            if len(seeds) != len(scales):
                raise ValueError("genome_dict seeds and perturb_scales must be same length")

            worker = model_worker
            state = worker.model_runner.state

            flat_state = list(state.flat_state())
            # Stable order
            flat_state.sort(key=lambda x: str(x[0]))
            total_params = len(flat_state)

            class SimpleParam:
                def __init__(self, value):
                    self.value = value

            # For each chunk
            for i in range(0, total_params, chunk_size):
                current_state = worker.model_runner.state
                chunk_items = flat_state[i: i + chunk_size]
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

                    # Only perturb/update floating arrays
                    if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
                        # Accumulate delta
                        aggregate_delta = jnp.zeros(leaf.shape, dtype=jnp.float32)
                        for seed, weight in zip(seeds, scales):
                            key = jax.random.PRNGKey(int(seed))
                            key = jax.random.fold_in(key, global_param_index)
                            noise = jax.random.normal(key, leaf.shape, dtype=jnp.float32)
                            aggregate_delta = aggregate_delta + (noise * weight)

                        aggregate_delta = aggregate_delta.astype(leaf.dtype)
                        if mode == "restore":
                            new_val = leaf - aggregate_delta
                        else:
                            # mode == "perturb" or "update" => add delta
                            new_val = leaf + aggregate_delta
                    else:
                        new_val = leaf

                    key_str = '.'.join(str(k) for k in path)
                    chunk_update[key_str] = SimpleParam(new_val)
                    if sharding is not None:
                        chunk_mappings[key_str] = (key_str, sharding)

                # Apply the chunk update
                chunk_state = nnx.State(chunk_update)
                worker.sync_weights(
                    updated_weights=chunk_state,
                    mappings=chunk_mappings,
                    transpose_keys={},
                    reshard_fn=None
                )

                # Ensure arrays materialize
                arrays_to_sync = [p.value for p in chunk_update.values()]
                for arr in arrays_to_sync:
                    try:
                        arr.block_until_ready()
                    except Exception:
                        # Fallback sync
                        _ = jax.device_get(arr)

                del chunk_update
                del chunk_mappings
                gc.collect()

        # Tokenizer helper
        def _build_prompts_from_chats(raw_chats: List[List[Dict[str, str]]], suffix: Optional[str]):
            tok = llm.get_tokenizer()
            prompts_local = []
            for chat in raw_chats:
                s = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                if suffix is not None:
                    s = s + suffix
                prompts_local.append(s)
            return prompts_local

        # Perform generation for a single task
        def _generate_for_task(genome_dict, raw_inputs: List[List[Dict[str, str]]], suffix: Optional[str]):
            if sampler is None:
                raise RuntimeError("SamplingParams not set in worker. Send 'set_sampler' before 'generate'.")
            # Perturb
            _process_weights_chunked(genome_dict=genome_dict, mode="perturb")
            try:
                prompts = _build_prompts_from_chats(raw_inputs, suffix)
                t0 = time.time()
                outputs = llm.generate(prompts, sampler, use_tqdm=use_tqdm)
                texts = [o.outputs[0].text for o in outputs]
                if time_self:
                    print(f"[Worker {device_id}] Generated {len(texts)} outputs in {time.time() - t0:.2f}s")
                return texts
            finally:
                # Always restore
                _process_weights_chunked(genome_dict=genome_dict, mode="restore")

        # Ready
        conn.send(("ready", {"device_id": device_id, "n_devices": len(jax.devices())}))

        # Command loop
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break

            if not isinstance(msg, (tuple, list)) or len(msg) == 0:
                continue

            cmd = msg[0]

            if cmd == "set_sampler":
                # Set SamplingParams object inside worker
                sampler = msg[1]
                conn.send(("ack", "set_sampler"))
            elif cmd == "generate":
                # msg: ("generate", task_id, genome_dict, inputs_for_genome, suffix)
                _, task_id, genome_dict, raw_inputs, suffix = msg
                try:
                    out_texts = _generate_for_task(genome_dict, raw_inputs, suffix)
                    conn.send(("result", task_id, out_texts))
                except Exception as e:
                    conn.send(("error", task_id, repr(e)))
            elif cmd == "update":
                # msg: ("update", genome_dict)
                _, genome_dict = msg
                try:
                    # Permanent update = add delta once
                    _process_weights_chunked(genome_dict=genome_dict, mode="update")
                    conn.send(("ack", "update"))
                except Exception as e:
                    conn.send(("error", "update", repr(e)))
            elif cmd == "save_weights":
                # msg: ("save_weights", filepath)
                _, filepath = msg
                try:
                    # Save from this worker
                    state = model_worker.model_runner.state
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
                    conn.send(("ack", "save_weights"))
                except Exception as e:
                    conn.send(("error", "save_weights", repr(e)))
            elif cmd == "shutdown":
                conn.send(("ack", "shutdown"))
                break
            else:
                conn.send(("error", "unknown_cmd", str(cmd)))
    except Exception as e:
        try:
            conn.send(("fatal", repr(e)))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class VllMTPUMultiDeviceBackend(Backend):
    """
    Spawns one separate vLLM LLM instance on each TPU device (no tensor-parallel).
    Each worker perturbs itself, generates, restores, and then picks the next unfinished batch.
    Updates are broadcast and applied universally across all workers.
    """

    def __init__(
        self,
        model_name: str,
        sampler,  # vllm.SamplingParams object created by caller
        use_tqdm: bool = False,
        max_model_len: int = 4096,
        time_self: bool = False,
        num_devices: int = 8,
        gpu_memory_utilization: float = 0.6,
        chunk_size: int = 10,
    ):
        super().__init__(
            backend_name="vLLM TPU Backend (Multi-Device, no TP)",
            model_name=model_name,
            NUM_GPUS=num_devices,  # reusing fields for TPU device count
            CPUS_PER_GPU=1,
            GPU_FRACTION_VLLM_WORKER=gpu_memory_utilization,
            sampler=sampler,
            use_tqdm=use_tqdm,
            max_model_len=max_model_len,
            time_self=time_self,
        )
        self.num_devices = num_devices
        self.gpu_memory_utilization = gpu_memory_utilization
        self.chunk_size = chunk_size

        self._workers = []  # list of (Process, Conn)
        self._started = False

    def startup(self, trainer=None):
        print(f"#-- Initializing vLLM TPU Multi-Device Backend (workers={self.num_devices}) --#")
        mp.set_start_method("spawn", force=True)

        # Spawn workers pinned to each device id
        for device_id in range(self.num_devices):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=_worker_loop,
                args=(
                    child_conn,
                    device_id,
                    self.model_name,
                    self.max_model_len,
                    self.gpu_memory_utilization,
                    self.use_tqdm,
                    self.time_self,
                    self.chunk_size,
                ),
                daemon=True,
            )
            p.start()
            self._workers.append((p, parent_conn))

        # Wait for readiness and set sampler
        conns = [c for _, c in self._workers]
        for conn in conns:
            msg = conn.recv()
            if not (isinstance(msg, (tuple, list)) and msg[0] == "ready"):
                raise RuntimeError(f"Worker failed to initialize: {msg}")
        # Send sampler to all workers
        for _, conn in self._workers:
            conn.send(("set_sampler", self.sampler))
        # Await ack
        for _, conn in self._workers:
            msg = conn.recv()
            if not (isinstance(msg, (tuple, list)) and msg[0] == "ack" and msg[1] == "set_sampler"):
                raise RuntimeError(f"Worker failed to set sampler: {msg}")

        self._started = True
        print("#-- vLLM TPU Multi-Device Backend Initialized Successfully --#")

    def _broadcast_update(self, update_genome_dict: Dict[str, Any]):
        # Broadcast permanent update to all workers
        for _, conn in self._workers:
            conn.send(("update", update_genome_dict))
        # Wait for acks
        for _, conn in self._workers:
            msg = conn.recv()
            if not (isinstance(msg, (tuple, list)) and msg[0] == "ack" and msg[1] == "update"):
                raise RuntimeError(f"Update failed on a worker: {msg}")

    def update(self, optimizer: Optimizer):
        """
        Permanently update the model weights on all workers.
        We fetch the representative genome from the optimizer and broadcast.
        """
        assert isinstance(optimizer, SimpleOpt), "Only SimpleOpt is supported for update."
        rep = optimizer.get_representative()
        update_genome_dict = {"seeds": list(rep.seeds), "perturb_scales": list(rep.perturb_scales)}

        print("#-- Broadcasting Model Weights Update (TPU Multi-Device) --#")
        t0 = time.time()
        self._broadcast_update(update_genome_dict)
        if self.time_self:
            print(f"#-- Update broadcast completed in {time.time() - t0:.2f}s --#")

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """
        Generate outputs concurrently across workers. Each worker:
         - Perturbs itself for the assigned genome,
         - Generates,
         - Restores itself,
         - Then gets the next unfinished batch.

        genomes: list of Genome objects (one per batch set)
        inputs: list of per-genome inputs (List[List[Dict[str, str]]]) matching genomes
        suffix: optional suffix string appended to each prompt
        """
        assert self._started, "Call startup() first."
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        num_tasks = len(genomes)
        if num_tasks == 0:
            return

        start_time_all = time.time()

        # Prepare tasks
        tasks = list(range(num_tasks))
        next_task_idx = 0

        # Assign initial tasks to available workers
        # Maintain a map from conn -> task_id
        busy: Dict[Any, int] = {}

        conns = [c for _, c in self._workers]

        def send_task_to_worker(conn, task_id: int):
            genome = genomes[task_id]
            genome_dict = {"seeds": list(genome.seeds), "perturb_scales": list(genome.perturb_scales)}
            conn.send(("generate", task_id, genome_dict, inputs[task_id], suffix))
            busy[conn] = task_id

        # Fill all workers initially
        for conn in conns:
            if next_task_idx < num_tasks:
                send_task_to_worker(conn, next_task_idx)
                next_task_idx += 1

        # Event loop: wait for results and assign next tasks as workers free up
        completed = 0
        while completed < num_tasks:
            ready_conns = mp_wait(list(busy.keys()))
            for conn in ready_conns:
                msg = conn.recv()
                if not isinstance(msg, (tuple, list)):
                    raise RuntimeError(f"Unexpected worker message: {msg}")

                kind = msg[0]
                if kind == "result":
                    _, task_id, out_texts = msg
                    # set outputs on the appropriate genome
                    genomes[task_id].latest_outputs = out_texts
                    completed += 1

                    if self.time_self:
                        print(f"#-- Genome {task_id+1}/{num_tasks} generated --#")

                    # This worker is now free; assign it a new task if any left
                    del busy[conn]
                    if next_task_idx < num_tasks:
                        send_task_to_worker(conn, next_task_idx)
                        next_task_idx += 1
                elif kind == "error":
                    _, task_id_or_cmd, err_str = msg
                    raise RuntimeError(f"Worker error on {task_id_or_cmd}: {err_str}")
                elif kind == "fatal":
                    _, err_str = msg
                    raise RuntimeError(f"Worker fatal error: {err_str}")
                else:
                    # Ignore unknown acks in generation loop
                    pass

        if self.time_self:
            print(f"#-- All genomes generated in {time.time() - start_time_all:.2f}s --#")

    def save_weights_to_disk(self, filepath: str):
        """
        Save weights from worker 0 to disk.
        All workers should be in sync due to global updates, so saving from one is sufficient.
        """
        print("WARNING: MODEL SAVING WITH TPUS MAY BE EXPERIMENTAL. PROCEED WITH CAUTION.")
        print(f"#-- Saving weights to {filepath} --#")
        _, conn0 = self._workers[0]
        conn0.send(("save_weights", filepath))
        msg = conn0.recv()
        if not (isinstance(msg, (tuple, list)) and msg[0] == "ack" and msg[1] == "save_weights"):
            raise RuntimeError(f"Save weights failed: {msg}")
        print("#-- Weights saved successfully --#")

    def shutdown(self):
        if not self._started:
            return
        # Gracefully stop workers
        for p, conn in self._workers:
            try:
                conn.send(("shutdown",))
            except Exception:
                pass
        for p, conn in self._workers:
            try:
                msg = conn.recv()
                # expect ("ack", "shutdown") or EOF
            except EOFError:
                pass
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            try:
                p.join(timeout=5)
            except Exception:
                pass
        self._workers.clear()
        self._started = False


# Example usage pattern (for reference; integrate into your trainer):
# backend = VllMTPUMultiDeviceBackend(
#     model_name="Qwen/Qwen2.5-4B-Instruct",
#     sampler=SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128),
#     num_devices=8,
#     use_tqdm=False,
#     time_self=True,
# )
# backend.startup()
# backend.generate_outputs(genomes, suffix, inputs)
# backend.update(optimizer)
# backend.save_weights_to_disk("/kaggle/working/model.pth")
# backend.shutdown()