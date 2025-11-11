from libs.backend.vllm_lorabackend import VLLMBackendLoRA
from libs.datasets.countdown_dataset import load_countdown_dataset
from libs.datasets.oreal_math_dataset import load_oreal_rl_prompts_dataset
from libs.genome import Genome
from libs.trainer import SimpleTrainer
from libs.optimizers import SimpleOpt, MomentumOpt
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

    backend = VLLMBackendLoRA(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=0.7, Sampler=sampler, lora_rank=8, use_tqdm=True, time_self=False, lora_perturb_target="b-")
    
    optimizer = SimpleOpt(total_steps=75, learning_rate=0.015, seed_weight=0.02)
    #optimizer = MomentumOpt(total_steps=75, learning_rate=0.015, seed_weight=0.02)
    #optimizer = MuonOpt(total_steps=75, learning_rate=0.015, seed_weight=0.02)

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