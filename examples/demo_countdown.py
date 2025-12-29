from propagate.backend.vllm_backend import VLLMBackend
from propagate.datasets.countdown_dataset import load_countdown_dataset
from propagate.genome import Genome
from propagate.trainer import SimpleTrainer
from propagate.optimizers import SimpleOpt, MomentumOpt, MuonOpt
from vllm import SamplingParams

import gc
import torch
import ray

gc.collect()
torch.cuda.empty_cache()

try:
    dataset = load_countdown_dataset(batch_size=300)
    dataset.generate_test_split(test_fraction=0.1, fold_index=1)

    sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)

    backend = VLLMBackend(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=0.9, sampler=sampler, use_tqdm=False, time_self=True)

    optimizer = SimpleOpt(total_steps=250, learning_rate=0.03, perturb_scale=0.001, norm_by_mean=False, norm_by_stddev=False)

    trainer = SimpleTrainer(population_size=14,
                            mirror=True,
                            optimizer=optimizer,
                            backend=backend,
                            dataset=dataset,
                            wandb_project="propagate_optimizers",
                            validate_every=10,
                            print_samples=True,
                            checkpoint_every=50,
                            checkpoint_path="checkpoints/Qwen_Qwen2_5-3B-Instruct.json"
    )

    trainer.train()

    trainer.backend.save_weights_to_disk("saved_model/Qwen_Qwen2_5-3B-Instruct.pt")

    print("#-- Training complete --#")

finally:
    ray.shutdown()
