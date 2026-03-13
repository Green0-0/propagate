from propagate.backend.vllm_backend import VLLMBackend
from propagate.datasets.countdown_dataset import load_countdown_dataset
from propagate.genome import Genome
from propagate.trainer import SimpleTrainer
from propagate.training_config import TrainingConfig
from propagate.optimizers.optimizer import Optimizer
from propagate.optimizers import chain, chain_adam, chain_adam_seeded, chain_log, chain_misc
from vllm import SamplingParams

import gc
import torch
import ray

gc.collect()
torch.cuda.empty_cache()

try:
    dataset = load_countdown_dataset(batch_size=300, force_reuse_batches=True)
    dataset.generate_test_split(test_fraction=0.1, fold_index=1)
    sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)
    backend = VLLMBackend(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=0.9, sampler=sampler, use_tqdm=False, time_self=True)
    perturb_chain = [
        chain.Init_Perturbation_Bernoulli(fp32_accumulate=True), 
        chain.Scale_Perturbation(mul_by_std=True, mul_by_lr_scalar=True), 
        chain.Add_Perturb_Buffer(), 
        chain.Delete_Perturb_Buffer()
    ]
    update_chain = [
        chain.Init_Perturbation_Bernoulli(fp32_accumulate=True), 
        chain.Scale_Perturbation(div_by_pop=True, mul_by_lr=True, mul_by_std=True, div_by_rstd=True, mul_by_lr_scalar=True), 
        chain.Add_Perturb_Buffer(), 
        chain.Delete_Perturb_Buffer()
    ]
    config = TrainingConfig(
        total_steps=200,
        learning_rate=25,
        perturb_scale=0.001,
        population_size=28,
        mirror=False,
        rank_norm_rewards=False,
        centered_eval=True,
        dynamic_perturb_smoothing_factor=0.2
    )
    optimizer = Optimizer(
        optimizer_name="Test Optimizer",
        config=config,
        perturb_chain=perturb_chain,
        update_chain=update_chain
    )
    trainer = SimpleTrainer(optimizer=optimizer,
                            backend=backend,
                            dataset=dataset,
                            wandb_project="propagate_v2_optimizers",
                            wandb_project_name="bern_nomirror_centered_fp32_lr25_std0.001",
                            validate_every=10,
                            print_samples=True,
                            checkpoint_every=1000,
                            checkpoint_path="checkpoints/Qwen_Qwen2_5-3B-Instruct.json"
    )
    trainer.train()
    trainer.backend.save_weights_to_disk("saved_model/Qwen_Qwen2_5-3B-Instruct.pt")
    print("#-- Training complete --#")
finally:
    ray.shutdown()
