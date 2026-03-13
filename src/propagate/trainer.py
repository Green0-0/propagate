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
    do_centered_eval : bool, optional
        Whether or not to do one extra eval on the unperturbed gradient. Enables centered eval and dyanmic perturbation scales. Defaults to False.
    pass_true_mean : bool, optional
        Whether or not to pass the centered eval mean to the optimizer as the true mean for gradient calculation when norm_by_mean is set to true. Defaults to True.
    dynamic_perturbation_target : float, optional
        The amount we want our perturbed rewards to drift from the center. Defaults to 0.1.
    dynamic_perturbation_smoothing_factor : float, optional 
        Whether or not to dynamically adjust the perturbation scale using the centered eval and a slightly modified version of the PSR rule that works with mirroring. Defaults to 0, which disables it, recommend <0.5.
    skip_step_ratio : float, optional
        Skips steps where the proportion of genomes with a reward greater than the centered eval is less than this variable. Defaults to 0.05.
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

    def __init__(self, optimizer: Optimizer, backend: Backend, dataset: Dataset, do_centered_eval: bool = False, pass_true_mean: bool = True, dynamic_perturbation_target: float = 0.1, dynamic_perturbation_smoothing_factor: float = 0, skip_step_ratio: float = 0.05, wandb_project: str = None, wandb_project_name: str = None, validate_every: int = 0, print_samples: bool = False, checkpoint_every: int = 0, checkpoint_path: str = "checkpoints/model.json"):
        print("#-- Initializing Trainer [SimpleTrainer] --#")
        print(f"#-- Population Size: {optimizer.population_size}, Learning Rate: {optimizer.learning_rate}, Weight: {optimizer.perturb_scale} --#")
        self.optimizer = optimizer
        self.backend = backend
        self.dataset = dataset
        
        backend.startup(self)
        
        self.genomes = [Genome() for _ in range(optimizer.population_size)]
        for genome in self.genomes:
            genome.mutate_seed(1)
        if optimizer.mirror:
            mirrored_genomes = []
            for genome in self.genomes:
                mirrored_genomes.append(genome.get_mirrored())
            self.genomes.extend(mirrored_genomes)

        self.iteration_count = 0
        
        self.do_centered_eval = do_centered_eval
        self.pass_true_mean = pass_true_mean
        self.dynamic_perturbation_target = dynamic_perturbation_target
        self.dynamic_perturbation_smoothing_factor = dynamic_perturbation_smoothing_factor
        self.skip_step_ratio = skip_step_ratio
        
        if dataset.force_reuse_batches == False and self.do_centered_eval:
            # TODO: Consider bagging style sampling from the batch instead of assigning a seperate batch to the centered eval
            print("WARNING: CENTERED EVAL WON'T WORK WELL UNLESS YOU REUSE BATCHES!")
        
        if self.do_centered_eval:
            self.center = Genome()
            self.genomes.append(self.center)

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
                    "population_size": optimizer.population_size,
                    "mirror": optimizer.mirror,
                    "total_steps": optimizer.total_steps,
                    "learning_rate": optimizer.learning_rate,
                    "perturb_scale": optimizer.perturb_scale,
                    "do_centered_eval": self.do_centered_eval,
                    "pass_true_mean": self.pass_true_mean,
                    "dynamic_perturbation_target": self.dynamic_perturbation_target,
                    "dynamic_perturbation_smoothing_factor": self.dynamic_perturbation_smoothing_factor,
                    "skip_step_ratio": self.skip_step_ratio,
                    "optimizer": optimizer.optimizer_name,
                    "optimizer_mean_norm": optimizer.norm_by_mean,
                    "warmup_steps": optimizer.warmup_steps,
                    "scheduler": optimizer.scheduler,
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
        while self.iteration_count < self.optimizer.total_steps:
            self.iteration_count += 1

            start_time = time.time()

            # Evaluation
            inputs = self.dataset.next(population_size=self.optimizer.population_size, mirror=self.optimizer.mirror, center=self.do_centered_eval)
            self.backend.generate_outputs(self.genomes, self.optimizer, self.dataset.suffix, inputs)
            self.dataset.score_all(self.genomes)
            if self.do_centered_eval:
                self.genomes.remove(self.center)
            end_time = time.time()

            print(f"#-- Iteration {self.iteration_count} completed in {end_time - start_time:.2f} seconds --#")
            genome_rewards = [g.historical_rewards[-1] for g in self.genomes]
            reward_mean = sum(genome_rewards) / len(self.genomes)
            reward_std = (sum([(r - reward_mean) ** 2 for r in genome_rewards]) / len(self.genomes)) ** 0.5
        
            # Gradient update
            skip = False
            if self.do_centered_eval:
                center_reward = self.center.historical_rewards[-1]
                better_count = sum(1 for genome in self.genomes if genome.historical_rewards[-1] > center_reward)
                proportion_better = better_count / len(self.genomes)

                # Skip the update if the ratio is too low
                if proportion_better >= self.skip_step_ratio:
                    self.log_train_stats(self.genomes, end_time - start_time, reward_mean, reward_std)
                    if self.pass_true_mean:
                        self.optimizer.update_self(self.genomes, self.iteration_count, true_reward_mean=center_reward)
                    else:
                        self.optimizer.update_self(self.genomes, self.iteration_count)
                    self.backend.update(self.optimizer)
                else:
                    print(f"#-- Skipping update: better ratio ({proportion_better:.2f}) < threshold ({self.skip_step_ratio}) --#")
                    self.iteration_count -= 1
                    skip = True
                    
                degradation = (center_reward - reward_mean) / (reward_std + 1e-8)
                std_scale_factor = math.exp(self.dynamic_perturbation_smoothing_factor * math.tanh(self.dynamic_perturbation_target - degradation))
                self.optimizer.perturb_scale *= std_scale_factor
            else:
                self.log_train_stats(self.genomes, end_time - start_time, reward_mean, reward_std)
                self.optimizer.update_self(self.genomes, self.iteration_count)
                self.backend.update(self.optimizer)
            if not skip:
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
                    self.save_model_seeds(path)
            
            # Create next generation of genomes
            self.genomes = [Genome() for _ in range(self.optimizer.population_size)]
            for genome in self.genomes:
                genome.mutate_seed(1)
            if self.optimizer.mirror:
                mirrored_genomes = []
                for genome in self.genomes:
                    mirrored_genomes.append(genome.get_mirrored())
                self.genomes.extend(mirrored_genomes)
            if self.do_centered_eval:
                self.center = Genome()
                self.genomes.append(self.center)

    def save_model_seeds(self, filepath: str):
        """Get the actual optimizer history from the optimizer (which has some structure and contains genomes), convert the genomes to a json-serializable format, and save it to a file."""
        history = self.optimizer.get_update_history()

        def serialize_structure(obj):
            """Recursively convert an arbitrary structure to a json-serializable format."""
            if isinstance(obj, Genome):
                return {
                    "seeds": obj.seeds,
                    "perturb_scales": obj.perturb_scales,
                    "historical_rewards": obj.historical_rewards,
                    "starting_index": obj.starting_index,
                    "__is_genome__": True
                }
            elif isinstance(obj, list):
                return [serialize_structure(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(serialize_structure(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: serialize_structure(v) for k, v in obj.items()}
            else:
                return obj

        serializable_history = serialize_structure(history)

        with open(filepath, "w") as f:
            json.dump(serializable_history, f, indent=4)
        print(f"#-- Successfully saved model seeds to {filepath} --#")

    def restore_model(self, filepath: str):
        """Load the optimizer history from a file, convert the genomes back to their original format, and restore the optimizer."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        def reconstruct_structure(obj):
            """Recursively convert a json-serializable format back to its original format."""
            if isinstance(obj, dict):
                if obj.get("__is_genome__") is True or ("seeds" in obj and "perturb_scales" in obj):
                    genome = Genome()
                    genome.seeds = obj.get("seeds", [])
                    genome.perturb_scales = obj.get("perturb_scales", [])
                    genome.historical_rewards = obj.get("historical_rewards", [float('-inf')] * len(genome.seeds))
                    genome.starting_index = obj.get("starting_index", 0)
                    return genome
                return {k: reconstruct_structure(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [reconstruct_structure(item) for item in obj]
            else:
                return obj

        history = reconstruct_structure(data)

        self.optimizer.restore_from_history(history, self.backend)
        self.iteration_count = len(history)
        
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
                    f"train/perturbation_scale": self.optimizer.perturb_scale,
                    f"iteration_count": self.iteration_count
                }
                if self.do_centered_eval:
                    center_reward = self.center.historical_rewards[-1]
                    better_count = sum(1 for genome in self.genomes if genome.historical_rewards[-1] > center_reward)
                    proportion_better = better_count / len(self.genomes)
                    
                    degradation = (center_reward - reward_mean) / (reward_std + 1e-8)
                    std_scale_factor = math.exp(self.dynamic_perturbation_smoothing_factor * math.tanh(self.dynamic_perturbation_target - degradation))
                
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
        score_stddev = (sum((genome.latest_rewards[i] - score) ** 2 for i in range(len(genome.latest_rewards))) / (len(genome.latest_rewards)-1)) ** 0.5
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