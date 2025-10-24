from libs.vllm_backend_tp import VLLMBackendTP
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

dataset = load_countdown_dataset()
dataset.batch_size = 100

sampler = SamplingParams(
    temperature=0.01,
    top_p=0.99,
    max_tokens=1024
)
backend = VLLMBackendMulti2(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_TRAINING_ACTOR=0.35, GPU_FRACTION_VLLM_WORKER=0.6, Sampler=sampler, output_log_file="logs/output_fullrun.log", full_output_log_file="logs/full_output_fullrun.log")
trainer = SimpleTrainer(
    population_size=28,
    learning_rate=0.0005,
    seed_weight=0.001,
    backend=backend,
    dataset=dataset,
    output_log_file="logs/output_fullrun.log",
    full_output_log_file="logs/full_output_fullrun.log",
    reward_log_file="logs/reward_fullrun.log"
)

for i in range(250):
    trainer.train_step()
