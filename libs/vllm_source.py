# SPDX-License-Identifier: Apache-2.0
"""
vLLM + Ray co-located Evolution Strategies (ES) implementation.

This script:
- Starts one Ray placement group PER GPU.
- Starts one training actor and one vLLM engine (tp_size=1) co-located on each GPU.
- Uses seeds-only perturbations and in-place layer perturbation/restoration
  on RayTrainingActor (to minimize memory).
- Evaluates perturbed models in parallel, one seed per GPU.
- Normalizes rewards per-iteration and performs ES update on all actors.
"""

import os
import ray
import torch
import time 
import numpy as np
import math
import logging
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

# ----------------------------
# Config (tune these)
# ----------------------------
NUM_GPUS = int(os.environ.get("NUM_GPUS_FROM_SLURM", 2))
CPUS_PER_GPU = int(os.environ.get("CPUS_PER_GPU_FROM_SLURM", 6))
GPU_FRACTION_TRAINING_ACTOR = 0.3
GPU_FRACTION_VLLM_WORKER = 0.65
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 1000
POPULATION_SIZE = 20        # N
SIGMA = 0.001               # sigma
ALPHA = 0.0005              # learning rate (digests 1/sigma)
NUM_ITERATIONS = 200         # T (kept small for testing)
BATCH_SIZE = 100            # per ES evaluation

# Log file names
REWARD_LOG_FILE = "rewards.log"
OUTPUT_LOG_FILE = "outputs.log"
# ----------------------------


#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(NUM_GPUS)))
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


# --------------------------------------------------------------------------- #
# Helper classes & actors
# --------------------------------------------------------------------------- #

class MyLLM(LLM):
    """Modified LLM class to accept bundle_indices and vllm_gpu_fraction."""
    def __init__(self, *args, bundle_indices: list, vllm_gpu_fraction: float, **kwargs):
        # Worker children should not inherit caller CUDA_VISIBLE_DEVICES
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(vllm_gpu_fraction)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        super().__init__(*args, **kwargs)


@ray.remote
class RayTrainingActor:
    """
    Each RayTrainingActor holds a full pytorch model replica on a single GPU.
    (No changes from original)
    """
    def __init__(self):
        from transformers import AutoModelForCausalLM
        # load model to a specific GPU device (first local CUDA device)
        self.device = torch.device("cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        self.model.to(self.device)
        self.model.eval()
        torch.cuda.synchronize(self.device)
        
        from vllm.platforms import current_platform
        self.device_uuid = current_platform.get_device_uuid(0)

    def report_device_id(self):
        return self.device_uuid

    def get_weight_ipc_handles(self):
        """
        Return CUDA-IPC handles for every parameter using torch.multiprocessing.reductions.reduce_tensor
        Format: {device_uuid: {name: reduced_tensor_handle, ...}}
        """
        from torch.multiprocessing.reductions import reduce_tensor
        return {self.device_uuid: {name: reduce_tensor(p.detach())
                                   for name, p in self.model.named_parameters()}}

    def perturb_with_seed(self, seed: int, sigma: float):
        """
        In-place perturb all parameters layer-by-layer using a deterministic RNG
        seeded by `seed`. The same seed can be used later to restore.
        """
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        for name, p in self.model.named_parameters():
            noise = torch.randn(p.shape, generator=gen, device=self.device, dtype=p.dtype)
            noise.mul_(sigma)
            p.data.add_(noise)
            del noise
        torch.cuda.synchronize(self.device)
        return True

    def restore_with_seed(self, seed: int, sigma: float):
        """
        Restore parameters by subtracting the SAME noise using the same RNG seed.
        """
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        for name, p in self.model.named_parameters():
            noise = torch.randn(p.shape, generator=gen, device=self.device, dtype=p.dtype)
            noise.mul_(sigma)
            p.data.add_(-noise)
            del noise
        torch.cuda.synchronize(self.device)
        return True

    def apply_es_update(self, seeds: list, rewards_normalized: list, alpha: float):
        """
        Apply aggregated ES update in-place:
        For each param: update += alpha * (1/N) * sum_n (Z_n * epsilon_n)
        where epsilon_n is sampled deterministically by seed n.
        This runs on this actor's device and updates its model. Return True on success.
        """
        N = len(seeds)
        rnorm = np.array(rewards_normalized, dtype=np.float32)
        for name, p in self.model.named_parameters():
            update = torch.zeros_like(p, device=self.device, dtype=p.dtype)
            for n_idx, seed in enumerate(seeds):
                gen = torch.Generator(device=self.device)
                gen.manual_seed(int(seed))
                noise = torch.randn(p.shape, generator=gen, device=self.device, dtype=p.dtype)
                noise.mul_(float(rnorm[n_idx]))
                update.add_(noise)
                del noise
            update.div_(float(N))
            p.data.add_(update, alpha=alpha)
            del update
        torch.cuda.synchronize(self.device)
        return True


# --------------------------------------------------------------------------- #
# Ray init & placement groups (NEW)
# --------------------------------------------------------------------------- #

ray.init(ignore_reinit_error=True)

# Create NUM_GPUS placement groups, one for each GPU
print(f"Creating {NUM_GPUS} separate placement groups...")
pgs = [placement_group([{"GPU": 1, "CPU": CPUS_PER_GPU}]) for _ in range(NUM_GPUS)]
ray.get([pg.ready() for pg in pgs])
print("All placement groups ready.")

# Spawn training actors and vLLM engines, 1 pair per GPU/PG
training_actors = []
inference_engines = []

for bidx in range(NUM_GPUS):
    pg = pgs[bidx]
    print(f"Spawning actor/vLLM pair for PG {bidx}...")
    
    # Spawn training actor
    a = RayTrainingActor.options(
        num_cpus=2,
        num_gpus=GPU_FRACTION_TRAINING_ACTOR,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
        ),
    ).remote()
    training_actors.append(a)

    # Spawn a single vLLM engine (tp_size=1) co-located in the same PG
    eng = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True),
    )(MyLLM).remote(
        model=MODEL_NAME,
        enforce_eager=True,
        worker_extension_cls="rlhf_utils.ColocateWorkerExtension",
        tensor_parallel_size=1,  # <-- Each engine is independent
        distributed_executor_backend="ray",
        gpu_memory_utilization=GPU_FRACTION_VLLM_WORKER,
        vllm_gpu_fraction=GPU_FRACTION_VLLM_WORKER, # <-- Pass fraction
        bundle_indices=[0], # <-- Always use bundle 0 of this PG
    )
    inference_engines.append(eng)

# Verify co-location for each pair
for bidx in range(NUM_GPUS):
    actor = training_actors[bidx]
    engine = inference_engines[bidx]
    
    actor_dev_id = ray.get(actor.report_device_id.remote())
    engine_dev_ids = ray.get(engine.collective_rpc.remote("report_device_id", args=tuple()))
    
    print(f"[GPU-{bidx}] train-actor UUID: {actor_dev_id}")
    print(f"[GPU-{bidx}] vLLM-worker UUID(s): {engine_dev_ids}")
    
    assert engine_dev_ids and actor_dev_id == engine_dev_ids[0], \
        f"Actor and vLLM engine on GPU {bidx} are NOT co-located!"

print("All actor/engine pairs are co-located correctly.")

# --------------------------------------------------------------------------- #
# Utilities: pushing weights (NEW)
# --------------------------------------------------------------------------- #

def push_all_weights(training_actors, inference_engines):
    """
    Pushes weights from ALL actors to ALL engines.
    Used after the main ES update to sync the base model.
    """
    print("Collecting IPC handles from all training actors...")
    ipc = {}
    # Get handles from ALL training actors
    ipc_handles_list = ray.get([a.get_weight_ipc_handles.remote() for a in training_actors])
    for handle_dict in ipc_handles_list:
        ipc.update(handle_dict)
    
    # Send the *complete* IPC dict to ALL engines
    # Each engine's worker will pick the handles it needs based on device_uuid
    print(f"Pushing all weights to {len(inference_engines)} inference engines...")
    tasks = []
    for eng in inference_engines:
        tasks.append(eng.collective_rpc.remote("update_weights_from_ipc_handles",
                                              args=(ipc,)))
    ray.get(tasks)
    print("Pushed all weights to all engines.")
    return True

# --------------------------------------------------------------------------- #
# Logging (NEW)
# --------------------------------------------------------------------------- #

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    handler = logging.FileHandler(log_file, mode='w')        
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# --------------------------------------------------------------------------- #
# Dataset helper (Unchanged)
# --------------------------------------------------------------------------- #

# ... (user's dataset loading logic remains here) ...

# --------------------------------------------------------------------------- #
# ES main loop (HEAVILY MODIFIED)
# --------------------------------------------------------------------------- #

def es_loop(dataset, tokenizer, reward_logger, output_logger, 
            num_iterations=NUM_ITERATIONS, pop_size=POPULATION_SIZE,
            sigma=SIGMA, alpha=ALPHA, max_new_tokens=MAX_NEW_TOKENS):
    """
    dataset: instance of your Dataset helper
    tokenizer: transformers tokenizer (must support apply_chat_template)
    reward_logger: logger for iteration-wise reward stats
    output_logger: logger for sample outputs
    """
    # Sampling params for greedy decoding (deterministic)
    sampler = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

    # Sync all engines to the base weights (from all actors)
    push_all_weights(training_actors, inference_engines)

    reward_logger.info(f"Population size: {pop_size}")
    reward_logger.info(f"Sigma: {sigma}")
    reward_logger.info(f"Alpha: {alpha}")
    reward_logger.info(f"Num iterations: {num_iterations}")
    reward_logger.info(f"Max new tokens: {max_new_tokens}")
    reward_logger.info(f"Batch size per evaluation: {dataset.batch_size}")
    for it in range(1, num_iterations + 1):
        print("=" * 20)
        print(f"\n=== ES Iteration {it}/{num_iterations} ===")
        print("=" * 20)

        output_logger.info("\n" + "="*30)
        output_logger.info(f"\n=== ES Iteration {it}/{num_iterations} ===")
        output_logger.info("="*30 + "\n")

        seeds = np.random.randint(0, 2**30, size=pop_size, dtype=np.int64).tolist()
        rewards = [0.0] * pop_size

        prompts_list, reward_fns = dataset.next()  # prompts_list: List[List[Dict[str,str]]]
        
        # Convert prompts to strings ONCE per iteration
        text_prompts = []
        for chat_dict_list in prompts_list:
            s = tokenizer.apply_chat_template(
                chat_dict_list,
                tokenize=False,
                add_generation_prompt=True
            )
            if getattr(dataset, "suffix", None):
                s = s + dataset.suffix
            text_prompts.append(s)

        # Process seeds in parallel chunks of size NUM_GPUS
        num_chunks = math.ceil(pop_size / NUM_GPUS)
        
        for chunk_idx in range(num_chunks):
            start_time = time.monotonic()
            
            start_n_idx = chunk_idx * NUM_GPUS
            end_n_idx = min((chunk_idx + 1) * NUM_GPUS, pop_size)
            
            current_batch_indices = list(range(start_n_idx, end_n_idx))
            if not current_batch_indices:
                continue
            
            num_in_chunk = len(current_batch_indices)
            print(f"Processing chunk {chunk_idx+1}/{num_chunks} (seeds {start_n_idx}-{end_n_idx-1}) on {num_in_chunk} GPUs...")

            # Get the actors, engines, and seeds for this parallel chunk
            current_actors = [training_actors[i] for i in range(num_in_chunk)]
            current_engines = [inference_engines[i] for i in range(num_in_chunk)]
            current_seeds = [seeds[n_idx] for n_idx in current_batch_indices]

            # 1. Perturb all actors in parallel
            perturb_tasks = [
                actor.perturb_with_seed.remote(int(seed), float(sigma)) 
                for actor, seed in zip(current_actors, current_seeds)
            ]
            ray.get(perturb_tasks)

            # 2. Get IPC handles from all perturbed actors in parallel
            ipc_tasks = [actor.get_weight_ipc_handles.remote() for actor in current_actors]
            ipc_handle_list = ray.get(ipc_tasks) # [ipc_dict_gpu_0, ipc_dict_gpu_1, ...]

            # 3. Push each actor's handles to its corresponding engine, in parallel
            # (engine_i receives ipc_dict_gpu_i)
            push_tasks = [
                engine.collective_rpc.remote("update_weights_from_ipc_handles", args=(ipc,))
                for engine, ipc in zip(current_engines, ipc_handle_list)
            ]
            ray.get(push_tasks)

            # 4. Run generation on all engines in parallel
            gen_tasks = [
                engine.generate.remote(text_prompts, sampler, use_tqdm=False)
                for engine in current_engines
            ]
            parallel_gens = ray.get(gen_tasks) # [results_gpu_0, results_gpu_1, ...]

            # 5. Process results and Restore actors (in parallel)
            restore_tasks = []
            for i, n_idx in enumerate(current_batch_indices):
                seed = current_seeds[i]
                gen = parallel_gens[i]
                actor = current_actors[i]
                gpu_id = i # The local index (0 to NUM_GPUS-1) used for this chunk
                
                # --- Reward computation (serial) ---
                batch_texts = []
                for out in gen:
                    text_field = getattr(out, "outputs", None)
                    if text_field and len(text_field) > 0 and hasattr(text_field[0], "text"):
                        batch_texts.append(text_field[0].text)
                    else:
                        batch_texts.append("")
                
                # *** LOGGING: Log first output ***
                first_output = "EMPTY"
                if len(batch_texts) > 0:
                    first_output = batch_texts[0]               

                batch_rewards = []
                for resp, reward_fn in zip(batch_texts, reward_fns):
                    try: r = float(reward_fn(resp))
                    except Exception: r = 0.0
                    batch_rewards.append(r)
                
                total_reward = sum(batch_rewards)
                rewards[n_idx] = float(total_reward) # Update main rewards array
                avg_length = sum(len(text) for text in batch_texts) / max(len(batch_texts), 1)
                output_logger.info(f"SEED={seed} (reward: {total_reward:.4f}, average output length: {avg_length:.2f}):\n{first_output}")
                print(f"Seed {seed} reward: {total_reward:.4f}, average output length: {avg_length:.2f}")
                # --- End reward computation ---

                # Add restore task to run in parallel
                restore_tasks.append(actor.restore_with_seed.remote(int(seed), float(sigma)))
            
            # Wait for all restores in this chunk to complete
            ray.get(restore_tasks)
            torch.cuda.empty_cache()
            
            end_time = time.monotonic()
            print(f"Chunk {chunk_idx+1}/{num_chunks} done in {(end_time - start_time):.2f} sec")

        # --- End of chunk loop ---

        # normalize and apply update (unchanged)
        rewards_arr = np.array(rewards, dtype=np.float32)
        mean = rewards_arr.mean()
        std = rewards_arr.std()
        if std < 1e-8:
            rewards_normalized = np.zeros_like(rewards_arr)
        else:
            rewards_normalized = (rewards_arr - mean) / (std + 1e-12)
        rewards_normalized_list = rewards_normalized.tolist()

        print(f"rewards: mean={mean:.4f}, std={std:.4f}, min={rewards_arr.min():.4f}, max={rewards_arr.max():.4f}")
        
        # *** LOGGING: Log reward stats ***
        reward_logger.info(f"IT={it}, MEAN={mean:.4f}, STD={std:.4f}, MIN={rewards_arr.min():.4f}, MAX={rewards_arr.max():.4f}")

        print("== Applying ES update to all training actors ==")
        update_tasks = [
            a.apply_es_update.remote(seeds, rewards_normalized_list, float(alpha))
            for a in training_actors
        ]
        ray.get(update_tasks)
        
        # Sync all engines to the new base weights
        push_all_weights(training_actors, inference_engines)

    print("\nES loop finished.")
    return True

# --------------------------------------------------------------------------- #
# Example usage (if run as script); user should replace dataset loader
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # --- Setup Loggers ---
    reward_logger = setup_logger('reward_logger', REWARD_LOG_FILE)
    output_logger = setup_logger('output_logger', OUTPUT_LOG_FILE)
    print(f"Logging rewards to {REWARD_LOG_FILE}")
    print(f"Logging outputs to {OUTPUT_LOG_FILE}")

    # Import your dataset loader
    try:
        # v-- REPLACE THIS with your actual dataset helper --v
        from libs.load_countdown_dataset import load_countdown_dataset
        dataset = load_countdown_dataset()
        dataset.batch_size = BATCH_SIZE
        # ^-- REPLACE THIS --^
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure your dataset loader (e.g., libs.load_countdown_dataset) is correct.")
        ray.shutdown()
        exit(1)

    # load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # Call the main loop
    try:
        es_loop(dataset, tokenizer, 
                reward_logger, output_logger,  # Pass loggers
                num_iterations=NUM_ITERATIONS, 
                pop_size=POPULATION_SIZE, 
                sigma=SIGMA, 
                alpha=ALPHA)
    except Exception as e:
        print(f"An error occurred during the ES loop: {e}")
    finally:
        print("Shutting down Ray.")
        ray.shutdown()(base)