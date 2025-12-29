import argparse
import sys
import os

# Ensure the current directory is in the python path so we can import from scripts
sys.path.append(os.getcwd())

from train import do_train

def main():
    parser = argparse.ArgumentParser(description="Quick Start for Propagate Training")

    # Model Configuration
    parser.add_argument("--model_source", type=str, default="Qwen/Qwen3-8B-Base",
                        help="Path or HuggingFace ID of the base model.")
    parser.add_argument("--lora_model_source", type=str, default="Qwen/Qwen3-8B-Base",
                        help="Path or HuggingFace ID for LoRA initialization (usually same as base).")
    parser.add_argument("--gpu_fraction", type=float, default=0.5,
                        help="Fraction of GPU memory to allocate per vLLM worker.")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="Rank of the LoRA adapter.")
    parser.add_argument("--ctx_len", type=int, default=4096,
                        help="Context length for the model.")

    # Training Configuration
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for training pairs.")
    parser.add_argument("--population_size", type=int, default=30,
                        help="Size of the population for evolution/optimization. This is doubled for mirroring.")
    parser.add_argument("--total_steps", type=int, default=500,
                        help="Total number of optimization steps.")
    
    # Optimizer Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3.0,
                        help="Learning rate (step size) for the optimizer.")
    parser.add_argument("--sigma", type=float, default=0.06,
                        help="Sigma (noise standard deviation) for perturbation.")
    parser.add_argument("--momentum", type=float, default=0.6,
                        help="Momentum factor for relevant optimizers.")
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="Beta2 parameter for relevant optimizers.")
    parser.add_argument("--optimizer_name", type=str, default="none",
                        choices=["none", "momentum", "muon", "adam", "two_halves", "stein"],
                        help="Type of optimizer to use.")

    # Misc
    parser.add_argument("--wandb_project", type=str, default="propagate_optimizers",
                        help="Weights & Biases project name.")
    parser.add_argument("--target_dataset", type=str, default="merged",
                        choices=["merged", "oreal", "mmlu", "megascience"],
                        help="Which dataset to use for training.")

    args = parser.parse_args()

    print(f"Starting training with optimizer: {args.optimizer_name}")
    print(f"Dataset: {args.target_dataset} | Population: {args.population_size} | Steps: {args.total_steps}")

    do_train(
        model_source=args.model_source,
        lora_model_source=args.lora_model_source,
        gpu_fraction=args.gpu_fraction,
        lora_rank=args.lora_rank,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        population_size=args.population_size,
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        perturb_scale=args.sigma,
        momentum=args.momentum,
        beta2=args.beta2,
        optimizer_name=args.optimizer_name,
        wandb_project=args.wandb_project,
        target_dataset=args.target_dataset
    )

if __name__ == "__main__":
    main()