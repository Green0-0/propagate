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

gc.collect()
torch.cuda.empty_cache()

try:
    dataset = load_oreal_rl_prompts_dataset(batch_size=300)
    dataset.generate_test_split(test_fraction=0.05, fold_index=1)

    sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)

    backend = VLLMBackend(model_name="Qwen/Qwen3-4B-Base", NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=0.85, Sampler=sampler)

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

    print("#-- Training complete --#")

finally:
    ray.shutdown()
