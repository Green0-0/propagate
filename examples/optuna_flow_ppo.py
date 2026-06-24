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
        n_startup_trials=30,
        multivariate=True,
        constant_liar=True,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=30,
        max_resource=200,
        reduction_factor=2,
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
        from propagate.backend.experimental.vllm_flow_backend import VLLMFlowBackendLoRA
        from propagate.datasets.countdown_dataset import load_countdown_dataset
        from propagate.experimental.flow_trainer import OptunaFlowTrainer
        from propagate.training_config import TrainingConfig
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
            flow_lr = trial.suggest_float("flow_lr", 1e-4, 5e-2, log=True)
            alpha_entropy = trial.suggest_float("alpha_entropy", 1e-4, 0.5, log=True)
            lora_perturb_target = trial.suggest_categorical("lora_perturb_target", ["b", "ab"])
            
            if lora_perturb_target == "b":
                target_sigma = trial.suggest_float("target_sigma_b", 0.01, 0.2)
            else:
                target_sigma = trial.suggest_float("target_sigma_ab", 0.001, 0.04)
            
            mu_lr = trial.suggest_float("mu_lr", 0.01, 6.0)
            adam_beta1 = trial.suggest_float("adam_beta1", 0.1, 0.99)
            adam_beta2 = trial.suggest_float("adam_beta2", 0.8, 0.999)
            
            flow_hidden_layers = trial.suggest_int("flow_hidden_layers", 1, 4)
            flow_hidden_dim = trial.suggest_int("flow_hidden_dim", 4, 16, step=2)
            
            ppo_epochs = trial.suggest_int("ppo_epochs", 2, 8)
            ppo_minibatches = trial.suggest_categorical("ppo_minibatches", [1, 2, 4])
            clip_eps = trial.suggest_float("clip_eps", 0.1, 0.4)
            grad_clip = trial.suggest_float("grad_clip", 0.1, 1.0)
            
            sampler = SamplingParams(temperature=0.00, seed=42, max_tokens=1024)
                        
            run_name = f"flow_ppo_t{trial.number}_tgt{lora_perturb_target}_lr{flow_lr:.1e}_ent{alpha_entropy:.1e}_sig{target_sigma:.2f}_ep{ppo_epochs}_mb{ppo_minibatches}"
            
            backend = VLLMFlowBackendLoRA(
                model_name="Qwen/Qwen2.5-3B-Instruct", 
                NUM_GPUS=4, 
                CPUS_PER_GPU=6, 
                GPU_FRACTION_VLLM_WORKER=0.75, 
                sampler=sampler, 
                use_tqdm=False, 
                time_self=True,
                lora_perturb_target=lora_perturb_target,
                norm_scale_update=False
            )
            
            # --- SETUP COMPUTE & DATA ---
            dataset = load_countdown_dataset(batch_size=100, force_reuse_batches=False)
            dataset.generate_test_split(test_fraction=0.2, fold_index=1)

            config = TrainingConfig(
                total_steps=200,
                learning_rate=1.0, # Unused, just to satisfy assertion
                population_size=28,
                perturb_scale=0.1, # Unused
                mirror=True, # Handled implicitly by antithetic sampling in trainer
                rank_norm_rewards=True,
                centered_eval=False,
            )
            
            trainer = OptunaFlowTrainer(
                config=config,
                backend=backend,
                dataset=dataset,
                flow_lr=flow_lr,
                alpha_entropy=alpha_entropy,
                target_sigma=target_sigma,
                mu_lr=mu_lr,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                flow_hidden_layers=flow_hidden_layers,
                flow_hidden_dim=flow_hidden_dim,
                ppo_mode=True,
                ppo_epochs=ppo_epochs,
                ppo_minibatches=ppo_minibatches,
                clip_eps=clip_eps,
                grad_clip=grad_clip,
                wandb_project="propagate_flow_ppo_sweeps",
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
                print(f"Trial {trial.number} pruned by Hyperband Pruner!")
                raise
            finally:
                if wandb.run is not None:
                    wandb.finish(quiet=True)

        study.optimize(objective, n_trials=1)

    except Exception as e:
        print(f"Worker process encountered an error: {e}")
        traceback.print_exc()
    finally:
        ray.shutdown()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Sweep for FlowES (PPO)")
    parser.add_argument("--trials", type=int, default=5, help="Total trials to run on this node")
    args = parser.parse_args()

    JOURNAL_PATH = "sweep_flow_ppo.journal"
    STUDY_NAME = "study_flow_ppo"

    study = get_or_create_study(STUDY_NAME, JOURNAL_PATH)

    print(f"\n#--- Starting FlowES PPO Sweep ({args.trials} trials) ---#")
    
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
