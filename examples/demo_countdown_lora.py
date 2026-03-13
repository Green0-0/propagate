from propagate.backend.vllm_lorabackend import VLLMBackendLoRA
from propagate.datasets.countdown_dataset import load_countdown_dataset
from propagate.genome import Genome
from propagate.trainer import SimpleTrainer
from propagate.training_config import TrainingConfig
from propagate.optimizers.optimizer import Optimizer
from propagate.optimizers import example_chains
from vllm import SamplingParams

import gc
import torch
import ray

gc.collect()
torch.cuda.empty_cache()

try:
    dataset = load_countdown_dataset(batch_size=50)
    dataset.generate_test_split(test_fraction=0.1, fold_index=1)
    sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)
    backend = VLLMBackendLoRA(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=0.75, sampler=sampler, lora_rank=8, use_tqdm=False, time_self=True, lora_perturb_target="b-", norm_scale_update=True)
    config = TrainingConfig(
        total_steps=250, 
        learning_rate=3, 
        perturb_scale=0.06,
        population_size=28,
        mirror=True,
        rank_norm_rewards=False
    )
    optimizer = Optimizer(
        optimizer_name="SimpleOpt",
        config=config,
        perturb_chain=example_chains.STANDARD_GAUSSIAN_PERTURB,
        update_chain=example_chains.STANDARD_GAUSSIAN_UPDATE
    )
    trainer = SimpleTrainer(
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