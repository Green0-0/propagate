import copy
import json
import math
import os
from typing import Any, List
from propagate.backend.backend_abc import Backend
from propagate.datasets.dataset import Dataset
from propagate.optimizers.optimizer import Optimizer
import time
import wandb

from propagate.genome import Genome

class SimpleTrainer:
    """The standard ES trainer. Orchestrates the training process, logging statistics, and model saving.
        
    Attributes
    ----------
    optimizer : Optimizer
        The optimizer to use for training.
    backend : Backend
        The backend to use for generating outputs.
    dataset : Dataset
        The dataset to use for training.
    wandb_project : str, optional
        The name of the project to use for logging. Defaults to None.
    validate_every : int, optional
        The number of iterations between validation. Defaults to 0.
    print_samples : bool, optional
        Whether to print samples. Defaults to False.
    checkpoint_every : int, optional
        The number of iterations between checkpoints. Defaults to 0.
    checkpoint_path : str, optional
        The path to save checkpoints to. Defaults to "checkpoints/model.json".
    """
    optimizer: Optimizer
    backend: Backend
    dataset: Dataset
    
    genomes: List[Genome]

    iteration_count: int

    wandb_project: str

    def __init__(self, optimizer: Optimizer, backend: Backend, dataset: Dataset, wandb_project: str = None, wandb_project_name: str = None, validate_every: int = 0, print_samples: bool = False, checkpoint_every: int = 0, checkpoint_path: str = "checkpoints/model.json"):
        self.config = optimizer.config
        backend.startup(self.config)
        
        print("#-- Initializing Trainer [SimpleTrainer] --#")
        print(f"#-- Population Size: {self.config.population_size}, Learning Rate: {self.config.learning_rate}, Weight: {self.config.perturb_scale} --#")
        if self.config.mirror:
            print("#-- Mirror mode enabled: population size doubled. --#")
        self.optimizer = optimizer
        self.backend = backend
        self.dataset = dataset
        
        self.genomes = [Genome() for _ in range(self.config.population_size)]
        for genome in self.genomes:
            genome.mutate_seed(1)
        if self.config.mirror:
            mirrored_genomes = []
            for genome in self.genomes:
                mirrored_genomes.append(genome.get_mirrored())
            self.genomes.extend(mirrored_genomes)

        self.iteration_count = 0
        self.update_history = {"genomes": [], "step": [], "lr": [], "std": [], "rstd": []}
        
        if self.config.centered_eval:
            self.center = Genome()
            self.genomes.append(self.center)
            if dataset.force_reuse_batches == False:
                print("WARNING: CENTERED EVAL WON'T WORK WELL UNLESS YOU REUSE BATCHES!")

        self.wandb_project = wandb_project
        self.validate_every = validate_every
        self.print_samples = print_samples
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

        if self.wandb_project is not None and self.wandb_project != "":
            try:
                wandb.login()
                config = {
                    "perturb_chain": ", ".join([obj.__class__.__name__ for obj in optimizer.perturb_chain]),
                    "inverted_perturb_chain": ", ".join([obj.__class__.__name__ for obj in optimizer.inverted_perturb_chain]),
                    "update_chain": ", ".join([obj.__class__.__name__ for obj in optimizer.update_chain]),
                    "population_size": self.config.population_size,
                    "mirror": self.config.mirror,
                    "total_steps": self.config.total_steps,
                    "learning_rate": self.config.learning_rate,
                    "perturb_scale": self.config.perturb_scale,
                    "do_centered_eval": self.config.centered_eval,
                    "pass_true_mean": self.config.pass_true_mean,
                    "dynamic_perturbation_target": self.config.dynamic_perturb_target,
                    "dynamic_perturbation_smoothing_factor": self.config.dynamic_perturb_smoothing_factor,
                    "optimizer": optimizer.optimizer_name,
                    "warmup_steps": optimizer.config.warmup_steps,
                    "lr_scheduler": optimizer.config.lr_scheduler,
                    "batch_size": dataset.batch_size,
                    "dataset_train_len": len(dataset.pairs_train),
                    "dataset_val_len": len(dataset.pairs_test),
                    "dataset_suffix": dataset.suffix,
                    "pass@k": dataset.passk,
                    "pass@k_proportion": dataset.passk_proportion,
                    "pass@k_minimum": dataset.passk_minimum,
                    "force_reuse_batches": dataset.force_reuse_batches,
                    "reward_func_ratio": dataset.reward_func_ratio,
                    "backend": backend.backend_name,
                    "num_gpus": backend.NUM_GPUS,
                    "cpus_per_gpu": backend.CPUS_PER_GPU,
                    "gpu_fraction_worker": backend.GPU_FRACTION_VLLM_WORKER,
                    "max_ctx_len": backend.max_model_len,
                    "checkpoint_every": checkpoint_every,
                }
                wandb.init(project=self.wandb_project, config=config, name=wandb_project_name)
                wandb.define_metric("iteration_count")
                wandb.define_metric("train/*", step_metric="iteration_count")
                wandb.define_metric("misc/*", step_metric="iteration_count")
                wandb.define_metric("val/*", step_metric="iteration_count")
                print(f"#-- WandB logging initialized for project: {self.wandb_project} --#")
            except Exception as e:
                print(f"#-- WandB logging initialization failed: {e} --#")
                self.wandb_project = None
            
        print("#-- Trainer initialized. --#")

    def train(self):
        """Trains the model for the specified number of iterations.
        Beguns by yielding the next set of data points, which are sent to the backend alongside the genomes for evaluation.
        Then, after evaluation, the dataset is used to score the genomes, which are passed to the optimizer to perform the gradient update.
        Lastly, a new generation is created, and statistics are logged (including val, if applicable)."""
        while self.iteration_count < self.config.total_steps:
            self.iteration_count += 1

            start_time = time.time()

            # Evaluation
            inputs = self.dataset.next(population_size=self.config.population_size, mirror=self.config.mirror, center=self.config.centered_eval)
            self.backend.generate_outputs(self.genomes, self.optimizer, self.dataset.suffix, inputs)
            self.dataset.score_all(self.genomes)
            if self.config.centered_eval:
                self.genomes.remove(self.center)
            end_time = time.time()

            print(f"#-- Iteration {self.iteration_count} completed in {end_time - start_time:.2f} seconds --#")
            genome_rewards = [g.historical_rewards[-1] for g in self.genomes]
            reward_mean = sum(genome_rewards) / len(self.genomes)
            reward_std = (sum([(r - reward_mean) ** 2 for r in genome_rewards]) / len(self.genomes)) ** 0.5
            self.log_train_stats(self.genomes, end_time - start_time, reward_mean, reward_std)
            
            # Gradient update
            if self.config.centered_eval:
                center_reward = self.center.historical_rewards[-1]
                self.optimizer.update_self(self.genomes, self.iteration_count, true_reward_mean=center_reward if self.config.pass_true_mean else None)
                    
                degradation = (center_reward - reward_mean) / (reward_std + 1e-8)
                std_scale_factor = math.exp(self.config.dynamic_perturb_smoothing_factor * math.tanh(self.config.dynamic_perturb_target - degradation))
                self.config.perturb_scale *= std_scale_factor
            else:
                self.optimizer.update_self(self.genomes, self.iteration_count)
            self.backend.update(self.optimizer)
            self.update_history['genomes'].append(self.optimizer.get_representative().get_data())
            self.update_history['step'].append(self.optimizer.last_step)
            self.update_history['lr'].append(self.optimizer.last_lr)
            self.update_history['std'].append(self.config.perturb_scale)
            self.update_history['rstd'].append(self.optimizer.last_rstd)
            
            # Validate
            if self.validate_every > 0 and self.iteration_count % self.validate_every == 0:
                new_genome = Genome()
                start_time = time.time()
                prompts = self.dataset.get_test_set()
                self.backend.generate_outputs([new_genome], self.optimizer, self.dataset.suffix, prompts)
                self.dataset.score_all([new_genome])
                end_time = time.time()
                print(f"#-- Validation for iteration {self.iteration_count} completed in {end_time - start_time:.2f} seconds --#")
                self.log_val_stats(new_genome, end_time - start_time)
            
            # Save checkpoint
            if self.checkpoint_every > 0 and self.iteration_count > 0 and self.iteration_count % self.checkpoint_every == 0:
                base, ext = os.path.splitext(self.checkpoint_path)
                path = f"{base}_step_{self.iteration_count}{ext}"
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                self.save_history(path)
            
            # Create next generation of genomes
            self.genomes = [Genome() for _ in range(self.config.population_size)]
            for genome in self.genomes:
                genome.mutate_seed(1)
            if self.config.mirror:
                mirrored_genomes = []
                for genome in self.genomes:
                    mirrored_genomes.append(genome.get_mirrored())
                self.genomes.extend(mirrored_genomes)
            if self.config.centered_eval:
                self.center = Genome()
                self.genomes.append(self.center)

    def save_history(self, filepath: str):
        """Get the update history from the optimizer, convert the genomes to a json-serializable format, and save it to a file."""
        with open(filepath, "w") as f:
            json.dump(self.update_history, f, indent=4)
        print(f"#-- Successfully saved model seeds to {filepath} --#")

    def restore_model(self, filepath: str):
        """Load the optimizer history from a file, convert the genomes back to their original format, and restore the optimizer."""
        if self.iteration_count != 0:
            raise RuntimeError("The trainer, optimizer, and backend must be in a blank state to properly restore the model!")
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.update_history = data
        
        for step_genome, step, lr, std, rstd in zip([Genome().from_data(g) for g in data['genomes']], data['step'], data['lr'], data['std'], data['rstd']):
            self.optimizer.rep_genome = step_genome            
            self.optimizer.last_step = step
            self.optimizer.last_lr = lr
            self.config.perturb_scale = std
            self.optimizer.last_rstd = rstd
            self.backend.update(self.optimizer)
            
        self.optimizer.rep_genome = Genome()
        self.iteration_count = self.update_history['step'][-1] if self.update_history['step'] else 0
        
        print(f"#-- Successfully loaded model seeds from {filepath}. Resuming from iteration {self.iteration_count} --#")
        
    def log_train_stats(self, genomes: List[Genome], time_taken: float, reward_mean: float, reward_std: float):
        """Log the training statistics for the current iteration."""
        average_response_length = sum([sum(len(response.split()) for response in genome.latest_outputs) for genome in genomes]) / len(genomes) / len(genomes[0].latest_outputs)

        best_genome = max(genomes, key=lambda g: g.historical_rewards[-1])
        worst_genome = min(genomes, key=lambda g: g.historical_rewards[-1])
        if self.wandb_project is not None and self.wandb_project != "":
            try:
                sample_table = wandb.Table(columns=["type", "reward", "response"])
                sample_table.add_data(
                    "best",
                    best_genome.historical_rewards[-1],
                    best_genome.latest_outputs[0]
                )
                sample_table.add_data(
                    "worst",
                    worst_genome.historical_rewards[-1],
                    worst_genome.latest_outputs[0]
                )
                log_data = {
                    f"train/average_reward": reward_mean,
                    f"train/min_reward": worst_genome.historical_rewards[-1],
                    f"train/max_reward": best_genome.historical_rewards[-1],
                    f"train/stddev_reward": reward_std,
                    f"train/time_seconds": time_taken,
                    f"train/average_response_length": average_response_length,
                    f"train/samples": sample_table,
                    f"train/learning_rate": self.optimizer.get_lr(self.iteration_count),
                    f"train/perturbation_scale": self.config.perturb_scale,
                    f"iteration_count": self.iteration_count
                }
                if self.config.centered_eval:
                    center_reward = self.center.historical_rewards[-1]
                    better_count = sum(1 for genome in self.genomes if genome.historical_rewards[-1] > center_reward)
                    proportion_better = better_count / len(self.genomes)
                    
                    degradation = (center_reward - reward_mean) / (reward_std + 1e-8)
                    std_scale_factor = math.exp(self.config.dynamic_perturb_smoothing_factor * math.tanh(self.config.dynamic_perturb_target - degradation))
                
                    log_data[f"train/centered_reward"] = center_reward
                    log_data[f"train/proportion_better"] = proportion_better
                    log_data[f"train/std_scale_factor"] = std_scale_factor
                    
                wandb.log(log_data, step=self.iteration_count)
            except Exception as e:
                print(f"#-- WandB logging failed: {e} --#")
        print(f"#-- Stats: average: {reward_mean}, min: {worst_genome.historical_rewards[-1]}, max: {best_genome.historical_rewards[-1]}, stddev: {reward_std}, average response length: {average_response_length} --#")
        if self.print_samples:
            print(f"#-- SAMPLE RESPONSE BEST GENOME: --#\n{best_genome.latest_outputs[0]}\n")
            print(f"#-- SAMPLE RESPONSE WORST GENOME: --#\n{worst_genome.latest_outputs[0]}\n") 

    def log_val_stats(self, genome: Genome, time_taken: float):
        """Log the validation statistics for the current iteration."""
        score = genome.historical_rewards[-1]
        score_stddev = (sum((genome.latest_rewards[i] - score) ** 2 for i in range(len(genome.latest_rewards))) / (len(genome.latest_rewards)-1)) ** 0.5 if len(genome.latest_rewards) > 1 else 0
        average_response_length = sum(len(response.split()) for response in genome.latest_outputs) / len(genome.latest_outputs)
        sample_response = genome.latest_outputs[0]
        if self.wandb_project is not None and self.wandb_project != "":
            try:
                wandb.log({
                    f"val/validation_score": score,
                    f"val/validation_stddev": score_stddev,
                    f"val/time_seconds": time_taken,
                    f"val/average_response_length": average_response_length,
                    f"val/sample_response": wandb.Table(data=[[sample_response]], columns=["response"]),
                    f"iteration_count": self.iteration_count
                }, step=self.iteration_count)
            except Exception as e:
                print(f"#-- WandB logging failed: {e} --#")
        print(f"#-- Stats: reward: {score}, response length: {average_response_length} --#")
        if self.print_samples:
            print(f"#-- SAMPLE RESPONSE: --#\n{sample_response}\n")