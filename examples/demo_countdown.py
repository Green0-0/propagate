from propagate.backend.vllm_backend import VLLMBackend
from propagate.datasets.countdown_dataset import load_countdown_dataset
from propagate.trainer import SimpleTrainer
from propagate.training_config import TrainingConfig
from propagate.optimizers.optimizer import Optimizer
from propagate.optimizers import chain
from vllm import SamplingParams

import gc
import torch
import ray

gc.collect()
torch.cuda.empty_cache()

try:
    dataset = load_countdown_dataset(batch_size=200, force_reuse_batches=False)
    dataset.generate_test_split(test_fraction=0.2, fold_index=1)
    
    sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)
    
    backend = VLLMBackend(
        model_name="Qwen/Qwen2.5-3B-Instruct", 
        NUM_GPUS=4, 
        CPUS_PER_GPU=6, 
        GPU_FRACTION_VLLM_WORKER=0.9, 
        sampler=sampler, 
        use_tqdm=False, 
        time_self=True
    )
    
    perturb_chain = [
        chain.Init_Perturbation_Gaussian(fp32_accumulate=True), 
        chain.Scale_Perturbation(mul_by_std=True, mul_by_lr_scalar=True), 
        chain.Add_Perturb_Buffer(), 
        chain.Delete_Perturb_Buffer()
    ]
    
    update_chain = [
        chain.Init_Perturbation_Gaussian(fp32_accumulate=True), 
        chain.Scale_Perturbation(div_by_pop=True, mul_by_lr=True, div_by_rstd=False, mul_by_std=True, mul_by_lr_scalar=True), 
        chain.Add_Perturb_Buffer(), 
        chain.Delete_Perturb_Buffer()
    ]
    
    config = TrainingConfig(
        total_steps=200,
        learning_rate=3.0,
        population_size=14,
        perturb_scale=0.001,
        mirror=True,
        rank_norm_rewards=True,
        centered_eval=True
    )
    
    optimizer = Optimizer(
        optimizer_name="SimpleOpt",
        config=config,
        perturb_chain=perturb_chain,
        update_chain=update_chain
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
    print("#-- Training complete --#")
finally:
    ray.shutdown()
