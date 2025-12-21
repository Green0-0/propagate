from libs.backend.vllm_lorabackend import VLLMBackendLoRA
from libs.datasets.countdown_dataset import load_countdown_dataset
from libs.genome import Genome
from libs.trainer import SimpleTrainer
from libs.optimizers import SimpleOpt, MomentumOpt, MuonOpt, AdamOpt
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

    backend = VLLMBackendLoRA(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=0.75, Sampler=sampler, lora_rank=8, use_tqdm=False, time_self=True, lora_perturb_target="b-", norm_scale_update=True)
    
    optimizer = SimpleOpt(total_steps=250, learning_rate=3, perturb_scale=0.06, norm_by_mean=False, norm_by_stddev=False)

    trainer = SimpleTrainer(population_size=28,
                            mirror=True,
                            optimizer=optimizer,
                            backend=backend,
                            dataset=dataset,
                            wandb_project="propagate_optimizers",
                            validate_every=10,
                            print_samples=True,
    )
    
    trainer.train()

    trainer.save_model_seeds("saved_model/saved_model_seeds.json")
    
    print("#-- Training complete --#")

finally:
    ray.shutdown()