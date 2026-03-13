import argparse
import optuna
import multiprocessing as mp
import os
import time
import traceback

def worker_process(storage_url, study_name, mode):
    """
    Runs in total isolation. Connects to the DB, runs 1 trial, and dies cleanly.
    """
    try:
        # 1. Heavy imports happen HERE to prevent CUDA context clashes
        from propagate.backend.vllm_backend import VLLMBackend
        from propagate.backend.vllm_lorabackend import VLLMBackendLoRA
        from propagate.datasets.countdown_dataset import load_countdown_dataset
        from propagate.trainer_optuna import OptunaTrainer
        from propagate.training_config import TrainingConfig
        from propagate.optimizers.optimizer import Optimizer
        from propagate.optimizers import chain, chain_adam, chain_adam_seeded, chain_log, chain_misc
        from vllm import SamplingParams
        import torch
        import ray
        import gc
        import wandb

        gc.collect()
        torch.cuda.empty_cache()

        # 2. Connect to the existing study database
        study = optuna.load_study(study_name=study_name, storage=storage_url)

        def objective(trial):
            # --- HYPERPARAMETER SEARCH SPACE ---
            # Independent parameters (Flat)
            noise_type = trial.suggest_categorical("noise_type", ['bern', 'gaus'])
            mirror = trial.suggest_categorical("mirror", [True, False])
            reuse_batches = trial.suggest_categorical("reuse_batches", [True, False])
            norm_type = trial.suggest_categorical("norm_type", ["rank_norm", "std_norm"])
            scheduler = trial.suggest_categorical("scheduler", ['constant', 'linear', 'cosine', 'exponential'])
            
            if reuse_batches:
                if not mirror:
                    centered_eval_norm = trial.suggest_categorical("centered_eval_norm", [True, False])
                else:
                    centered_eval_norm = False
                dynamic_perturbation_smoothing_factor = trial.suggest_float("dynamic_perturbation_smoothing_factor", 0.01, 1)
                dynamic_perturbation_target = trial.suggest_float("dynamic_perturbation_target", 0.05, 1)
            else:
                centered_eval_norm = False
                dynamic_perturbation_smoothing_factor = 0
                dynamic_perturbation_target = 0.1
                
            lr = trial.suggest_float("lr", 0.01, 100, log=True)
            
            # Branched parameters based on architecture (LoRA vs Full)
            sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)
            if mode == "lora":
                batch_size = 100
                population_size = 56 // (2 if mirror == True else 1)
                
                perturb_scale = trial.suggest_float("std_lora", 0.005, 0.5)
                perturb_target = trial.suggest_categorical("perturb_target", ["ab", "b-"])
                lora_normscale = trial.suggest_categorical("lora_normscale", [True, False])
                run_name = f"{perturb_target}_{noise_type}_t{trial.number}_lr{lr}_std{perturb_scale}_{norm_type}"
                backend = VLLMBackendLoRA(
                    model_name="Qwen/Qwen2.5-3B-Instruct", 
                    NUM_GPUS=4, 
                    CPUS_PER_GPU=6, 
                    GPU_FRACTION_VLLM_WORKER=0.75, 
                    sampler=sampler, 
                    use_tqdm=False, 
                    time_self=True,
                    lora_perturb_target=perturb_target,
                    norm_scale_update=lora_normscale
                )
            elif mode == "non_lora":
                batch_size = 200
                population_size = 28 // (2 if mirror == True else 1)
                
                perturb_scale = trial.suggest_float("std_full", 0.0001, 0.01, log=True)
                run_name = f"{noise_type}_t{trial.number}_{lr}_std{perturb_scale}_{norm_type}"
                backend = VLLMBackend(
                    model_name="Qwen/Qwen2.5-3B-Instruct", 
                    NUM_GPUS=4, 
                    CPUS_PER_GPU=6, 
                    GPU_FRACTION_VLLM_WORKER=0.75, 
                    sampler=sampler, 
                    use_tqdm=False, 
                    time_self=True
                )
            # --- SETUP COMPUTE & DATA ---
            dataset = load_countdown_dataset(batch_size=batch_size, force_reuse_batches=reuse_batches)
            dataset.generate_test_split(test_fraction=0.2, fold_index=1)

            # --- SETUP CHAINS ---
            # (Assuming standard chains for this example, branch if needed)
            perturb_chain = [
                chain.Init_Perturbation_Bernoulli(fp32_accumulate=True) if noise_type == 'bern' else chain.Init_Perturbation_Gaussian(fp32_accumulate=True), 
                chain.Scale_Perturbation(mul_by_std=True, mul_by_lr_scalar=True), 
                chain.Add_Perturb_Buffer(), 
                chain.Delete_Perturb_Buffer()
            ]
            update_chain = [
                chain.Init_Perturbation_Bernoulli(fp32_accumulate=True) if noise_type == 'bern' else chain.Init_Perturbation_Gaussian(fp32_accumulate=True), 
                chain.Scale_Perturbation(div_by_pop=True, mul_by_lr=True, div_by_rstd=(norm_type == "std_norm"), mul_by_std=True, mul_by_lr_scalar=True), 
                chain.Add_Perturb_Buffer(), 
                chain.Delete_Perturb_Buffer()
            ]

            config = TrainingConfig(
                total_steps=200,
                learning_rate=lr,
                perturb_scale=perturb_scale,
                mirror=mirror,
                population_size=population_size,
                rank_norm_rewards=(norm_type == "rank_norm"),
                lr_scheduler=scheduler,
                centered_eval=reuse_batches,
                pass_true_mean=centered_eval_norm,
                dynamic_perturb_target=dynamic_perturbation_target,
                dynamic_perturb_smoothing_factor=dynamic_perturbation_smoothing_factor
            )

            optimizer = Optimizer(
                optimizer_name="Test Optimizer", 
                config=config, 
                perturb_chain=perturb_chain, 
                update_chain=update_chain
            )

            # --- SETUP TRAINER ---
            trainer = OptunaTrainer(
                optimizer=optimizer,
                backend=backend,
                dataset=dataset,
                wandb_project="propagate_lora_basic_sweeps" if mode == 'lora' else "propagate_basic_sweeps",
                wandb_project_name=run_name,
                validate_every=10,
                print_samples=False,
                checkpoint_every=1000,
                checkpoint_path=f"checkpoints/Qwen_{run_name}.json"
            )

            try:
                final_score = trainer.train(optuna_trial=trial)
                return final_score
            except optuna.TrialPruned:
                print(f"Trial {trial.number} pruned by Hybrid Pruner!")
                raise
            finally:
                # Force W&B to sync before the OS kills the process
                if wandb.run is not None:
                    wandb.finish(quiet=True)

        # Run exactly ONE trial in this child process
        study.optimize(objective, n_trials=1)

    except Exception as e:
        print(f"Worker process encountered an error: {e}")
        traceback.print_exc()
    finally:
        # Failsafe cleanup
        if 'ray' in locals():
            ray.shutdown()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Sweep for Propagate")
    parser.add_argument("--mode", type=str, choices=["lora", "non_lora"], required=True, help="Which architecture to sweep")
    parser.add_argument("--trials", type=int, default=50, help="Total trials to run on this node")
    args = parser.parse_args()

    STORAGE_URL = f"sqlite:///sweep_{args.mode}.db"
    STUDY_NAME = f"study_{args.mode}"

    pruner = optuna.pruners.PercentilePruner(
        percentile=60.0,
        n_warmup_steps=100,
        n_min_trials=5 
    )
    
    study = optuna.create_study(
        direction="maximize", 
        study_name=STUDY_NAME, 
        storage=STORAGE_URL, 
        load_if_exists=True,
        pruner=pruner
    )

    print(f"\n#--- Starting {args.mode.upper()} Sweep ({args.trials} trials) ---#")
    
    ctx = mp.get_context('spawn')
    
    for i in range(args.trials):
        print(f"\n--- Initiating Trial {i+1}/{args.trials} ---")
        
        p = ctx.Process(target=worker_process, args=(STORAGE_URL, STUDY_NAME, args.mode))
        p.start()
        p.join()
        
        print("Scrubbing Ray daemon processes...")
        os.system("ray stop --force > /dev/null 2>&1")
        time.sleep(3) 

    print("\n#--- Sweep Complete ---#")