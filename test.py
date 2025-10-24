from libs.vllm_backend_multi import VLLMBackendMulti
from libs.vllm_backend_multi2 import VLLMBackendMulti2
from libs.countdown_dataset import load_countdown_dataset
from libs.genome import Genome
from libs.trainer import SimpleTrainer
from vllm import SamplingParams

import gc
import torch
gc.collect()
torch.cuda.empty_cache()

dataset = load_countdown_dataset(batch_size=50, pairs_loaded=50)

sampler = SamplingParams(
    temperature=0.00,
    seed=42,
    max_tokens=1024
)
backend = VLLMBackendMulti2(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=0.85, Sampler=sampler)
trainer = SimpleTrainer(
    population_size=12,
    learning_rate=0.0005,
    seed_weight=0.001,
    backend=backend,
    dataset=dataset
)

for i in range(250):
    trainer.train_step()
