import argparse
import optuna
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileOpenLock
import multiprocessing as mp
import os
import time
import traceback

def get_or_create_study(study_name, journal_path, direction="maximize"):
    directory = os.path.dirname(journal_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    lock = JournalFileOpenLock(f"{journal_path}.lock")
    storage = JournalStorage(JournalFileStorage(journal_path, lock_obj=lock))
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        multivariate=True,
        constant_liar=True,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=10,
        reduction_factor=4,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

def worker_process(journal_path, study_name):
    """
    Runs in total isolation. Connects to the DB, runs 1 trial, and dies cleanly.
    """
    try:
        from propagate.backend.vllm_lorabackend import VLLMBackendLoRA
        from propagate.datasets.countdown_dataset import load_countdown_dataset
        from propagate.trainer_optuna import OptunaTrainer
        from propagate.training_config import TrainingConfig
        from propagate.optimizers.optimizer import Optimizer
        from propagate.optimizers import chain, chain_misc
        from vllm import SamplingParams
        import torch
        import ray
        import gc
        import wandb
        import optuna

        gc.collect()
        torch.cuda.empty_cache()
        
        study = get_or_create_study(study_name, journal_path)

        def objective(trial):            
            # --- SWEPT HYPERPARAMETERS ---
            lr = trial.suggest_float("lr", 0.5, 10.0)
            lambda_val = trial.suggest_float("lambda_val", 1e-6, 1, log=True)
            exponent = trial.suggest_float("exponent", 0.01, 10.0, log=True)
            
            sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)
                        
            run_name = f"wd_t{trial.number}_lr{lr:.2f}_lam{lambda_val:.1e}_exp{exponent:.2f}"
            
            backend = VLLMBackendLoRA(
                model_name="Qwen/Qwen2.5-3B-Instruct", 
                NUM_GPUS=4, 
                CPUS_PER_GPU=6, 
                GPU_FRACTION_VLLM_WORKER=0.75, 
                sampler=sampler, 
                use_tqdm=False, 
                time_self=True,
                lora_perturb_target="b-",
                norm_scale_update=True
            )
            
            # --- SETUP COMPUTE & DATA ---
            dataset = load_countdown_dataset(batch_size=100, force_reuse_batches=False)
            dataset.generate_test_split(test_fraction=0.2, fold_index=1)

            # --- SETUP CHAINS ---
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
                chain.Delete_Perturb_Buffer(),
                chain_misc.Direct_Weight_Decay(lambda_val=lambda_val, exponent=exponent)
            ]
            config = TrainingConfig(
                total_steps=200,
                learning_rate=lr,
                population_size=28,
                perturb_scale=0.1,
                mirror=True,
                rank_norm_rewards=True,
                centered_eval=True,
            )
            optimizer = Optimizer(
                optimizer_name="Weight Decay Optimizer", 
                config=config, 
                perturb_chain=perturb_chain, 
                update_chain=update_chain
            )
            trainer = OptunaTrainer(
                optimizer=optimizer,
                backend=backend,
                dataset=dataset,
                wandb_project="propagate_lora_wd_sweeps",
                wandb_project_name=run_name,
                validate_every=10,
                print_samples=False,
                min_val_reward=0.25,
                start_prune_min_reward_iter=10
            )
            try:
                final_score = trainer.train(optuna_trial=trial)
                return final_score
            except optuna.TrialPruned:
                print(f"Trial {trial.number} pruned by Hybrid Pruner!")
                raise
            finally:
                if wandb.run is not None:
                    wandb.finish(quiet=True)

        study.optimize(objective, n_trials=1)

    except Exception as e:
        print(f"Worker process encountered an error: {e}")
        traceback.print_exc()
    finally:
        if 'ray' in locals():
            ray.shutdown()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Sweep for Propagate Weight Decay")
    parser.add_argument("--trials", type=int, default=100, help="Total trials to run on this node")
    args = parser.parse_args()

    JOURNAL_PATH = "sweep_lora_wd.journal"
    STUDY_NAME = "study_lora_wd"

    study = get_or_create_study(STUDY_NAME, JOURNAL_PATH)

    print(f"\n#--- Starting Weight Decay Sweep ({args.trials} trials) ---#")
    
    ctx = mp.get_context('spawn')
    
    for i in range(args.trials):
        print(f"\n--- Initiating Trial {i+1}/{args.trials} ---")
        
        p = ctx.Process(target=worker_process, args=(JOURNAL_PATH, STUDY_NAME))
        p.start()
        p.join()
        
        print("Scrubbing Ray daemon processes...")
        os.system("ray stop --force > /dev/null 2>&1")
        time.sleep(3) 

    print("\n#--- Sweep Complete ---#")
