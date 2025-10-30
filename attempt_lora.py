import re
from typing import Optional
from libs.vllm_backend import VLLMBackend
from libs.countdown_dataset import load_countdown_dataset
from libs.oreal_math_dataset import load_oreal_rl_prompts_dataset
from libs.genome import Genome
from libs.trainer import SimpleTrainer
from libs.optimizers import SimpleOptimizer, MomentumOptimizer, TestMaxOptimizer
from vllm import SamplingParams

import gc
import torch
import ray

sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)
backend = VLLMBackend(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=1, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=0.85, Sampler=sampler)
print(backend.report_lora_params(expected_adapter_names=["lora_adapter1", "lora_adapter2"]))