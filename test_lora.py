from libs.backend.vllm_lorabackend import VLLMBackendLoRA
from libs.datasets.countdown_dataset import load_countdown_dataset
from libs.datasets.oreal_math_dataset import load_oreal_rl_prompts_dataset
from libs.genome import Genome
from libs.trainer import SimpleTrainer
from libs.optimizers import SimpleOptimizer, MomentumOptimizer, TestMaxOptimizer
from vllm import SamplingParams

import gc
import torch
import ray

gc.collect()
torch.cuda.empty_cache()

try:
    dataset = load_countdown_dataset(batch_size=300)
    dataset.generate_test_split(test_fraction=0.1, fold_index=1)
    #dataset = load_oreal_rl_prompts_dataset(batch_size=300)
    #dataset.generate_test_split(test_fraction=0.05, fold_index=1)

    sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)

    backend = VLLMBackendLoRA(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=1, CPUS_PER_GPU=12, GPU_FRACTION_VLLM_WORKER=0.85, Sampler=sampler, population_size=2, lora_rank=16)
    
    """
    optimizer = SimpleOptimizer(total_steps=250, learning_rate=0.0005, seed_weight=0.001)
    #optimizer = MomentumOptimizer(total_steps=250, learning_rate=0.0005, seed_weight=0.001, warmup_steps=10, scheduler="cosine", momentum=0.5)
    #optimizer = TestMaxOptimizer(total_steps=250, learning_rate=0.0005, seed_weight=0.001, warmup_steps=0, scheduler="none")
    
    trainer = SimpleTrainer(population_size=28,
                            mirror=False,
                            optimizer=optimizer,
                            backend=backend,
                            dataset=dataset,
                            wandb_project="propagate_tests",
                            validate_every=10,
                            print_samples=True,
    )
    
    trainer.train()

    trainer.save_model_seeds("saved_model/saved_model_seeds.json")
    """
    print("#-- Training complete --#")

finally:
    ray.shutdown()