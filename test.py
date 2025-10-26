from libs.vllm_backend import VLLMBackend
from libs.countdown_dataset import load_countdown_dataset
from libs.genome import Genome
from libs.trainer import SimpleTrainer
from vllm import SamplingParams

import gc
import torch
gc.collect()
torch.cuda.empty_cache()

dataset = load_countdown_dataset(batch_size=100)
dataset.generate_test_split(test_fraction=0.1, fold_index=1)

sampler = SamplingParams(
    temperature=0.00,
    seed=42,
    max_tokens=1024
)

backend = VLLMBackend(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=0.85, Sampler=sampler)

trainer = SimpleTrainer(
    population_size=12,
    learning_rate=0.0005,
    seed_weight=0.001,
    backend=backend,
    dataset=dataset,
    wandb_project="propagate_tests",
    validate_every=10,
    print_samples=True,
)

for i in range(250):
    trainer.train_step()

trainer.save_model_seeds("saved_model/saved_model_seeds.json")
