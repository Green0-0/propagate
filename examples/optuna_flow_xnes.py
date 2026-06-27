# flow_xnes_entrypoint.py
import argparse
import optuna
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileOpenLock
import multiprocessing as mp
import os
import time
import traceback

def get_or_create_study(study_name, journal_path, direction="maximize"):
    directory = os.path.dirname(journal_path)
    if directory: os.makedirs(directory, exist_ok=True)
    lock = JournalFileOpenLock(f"{journal_path}.lock")
    storage = JournalStorage(JournalFileStorage(journal_path, lock_obj=lock))
    sampler = optuna.samplers.TPESampler(n_startup_trials=30, multivariate=True, constant_liar=True)
    pruner = optuna.pruners.HyperbandPruner(min_resource=30, max_resource=200, reduction_factor=2)
    return optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction=direction, sampler=sampler, pruner=pruner)

def worker_process(journal_path, study_name):
    try:
        from propagate.backend.experimental.vllm_flow_backend import VLLMFlowBackendLoRA
        from propagate.datasets.countdown_dataset import load_countdown_dataset
        from propagate.experimental.flow_trainer import XNESTrainer
        from propagate.training_config import TrainingConfig
        from vllm import SamplingParams
        import torch, ray, gc, wandb

        gc.collect()
        torch.cuda.empty_cache()
        study = get_or_create_study(study_name, journal_path)

        def objective(trial):            
            lora_perturb_target = trial.suggest_categorical("lora_perturb_target", ["b", "ab"])
            if lora_perturb_target == "b":
                target_sigma = trial.suggest_float("target_sigma_b", 0.01, 0.2)
            else:
                target_sigma = trial.suggest_float("target_sigma_ab", 0.001, 0.04)
            
            mu_lr = trial.suggest_float("mu_lr", 0.01, 6.0)
            sigma_lr = trial.suggest_float("sigma_lr", 0.01, 1.0)
            cov_lr = trial.suggest_float("cov_lr", 0.01, 1.0)
            
            num_subspaces = trial.suggest_categorical("num_subspaces", [1, 18, 36, 72, 252])
            subspace_k = trial.suggest_categorical("subspace_k", [1, 4, 8, 16])
            latent_dim = num_subspaces * subspace_k
            
            sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)
            run_name = f"xnes_t{trial.number}_ld{latent_dim}_ns{num_subspaces}_mu{mu_lr:.1e}"
            
            backend = VLLMFlowBackendLoRA(
                model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=4, CPUS_PER_GPU=6, 
                GPU_FRACTION_VLLM_WORKER=0.75, sampler=sampler, use_tqdm=False, time_self=True,
                lora_perturb_target=lora_perturb_target, norm_scale_update=False
            )
            
            dataset = load_countdown_dataset(batch_size=100, force_reuse_batches=False)
            dataset.generate_test_split(test_fraction=0.2, fold_index=1)

            config = TrainingConfig(
                total_steps=200, learning_rate=1.0, population_size=28, perturb_scale=0.1,
                mirror=True, rank_norm_rewards=True, centered_eval=False,
            )
            
            trainer = XNESTrainer(
                config=config, backend=backend, dataset=dataset,
                mu_lr=mu_lr, sigma_lr=sigma_lr, cov_lr=cov_lr,
                target_sigma=target_sigma, latent_dim=latent_dim, num_subspaces=num_subspaces,
                wandb_project="propagate_xnes_sweeps", wandb_project_name=run_name,
                validate_every=10, print_samples=False, min_val_reward=0.25, start_prune_min_reward_iter=10
            )
            try:
                return trainer.train(optuna_trial=trial)
            except optuna.TrialPruned:
                raise
            finally:
                if wandb.run is not None: wandb.finish(quiet=True)

        study.optimize(objective, n_trials=1)
    except Exception as e:
        print(f"Worker error: {e}"); traceback.print_exc()
    finally:
        ray.shutdown(); gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    JOURNAL_PATH = "sweep_flow_xnes.journal"
    STUDY_NAME = "study_flow_xnes"
    study = get_or_create_study(STUDY_NAME, JOURNAL_PATH)
    print(f"\n#--- Starting xNES Sweep ({args.trials} trials) ---#")
    ctx = mp.get_context('spawn')
    for i in range(args.trials):
        print(f"\n--- Initiating Trial {i+1}/{args.trials} ---")
        p = ctx.Process(target=worker_process, args=(JOURNAL_PATH, STUDY_NAME))
        p.start(); p.join()
        os.system("ray stop --force > /dev/null 2>&1")
        time.sleep(3)
    print("\n#--- Sweep Complete ---#")