import json
from typing import Any, List
from libs.backend.backend_abc import Backend
from libs.datasets.dataset import Dataset
from libs.optimizers import Optimizer
import time
import wandb

from libs.genome import Genome

class SimpleTrainer:
    population_size: int
    optimizer: Optimizer
    backend: Backend
    dataset: Dataset
    mirror: bool

    genomes: List[Genome]

    iteration_count: int

    wandb_project: str

    def __init__(self, population_size: int, optimizer: Optimizer, backend: Backend, dataset: Dataset, mirror: bool = False, wandb_project: str = None, validate_every: int = 0, print_samples: bool = False):
        print("#-- Initializing Trainer [SimpleTrainer] --#")
        print(f"#-- Population Size: {population_size}, Learning Rate: {optimizer.learning_rate}, Weight: {optimizer.perturb_scale} --#")
        self.optimizer = optimizer
        self.backend = backend
        self.dataset = dataset
        self.population_size = population_size
        self.mirror = mirror
        if mirror:
            print("#-- Mirror mode enabled: population size doubled. --#")
        
        backend.startup(self)
        
        self.genomes = [Genome() for _ in range(population_size)]
        for genome in self.genomes:
            genome.mutate_seed(optimizer.perturb_scale)
        if mirror:
            mirrored_genomes = []
            for genome in self.genomes:
                mirrored_genomes.append(genome.get_mirrored())
            self.genomes.extend(mirrored_genomes)

        self.iteration_count = 0

        self.wandb_project = wandb_project
        self.validate_every = validate_every
        self.print_samples = print_samples

        if self.wandb_project is not None and self.wandb_project != "":
            try:
                wandb.login()
                config = {
                    "population_size": population_size,
                    "mirror": mirror,
                    "total_steps": optimizer.total_steps,
                    "learning_rate": optimizer.learning_rate,
                    "perturb_scale": optimizer.perturb_scale,
                    "optimizer": optimizer.optimizer_name,
                    "optimizer_mean_norm": optimizer.norm_by_mean,
                    "optimizer_stddev_norm": optimizer.norm_by_stddev,
                    "optimizer_force_lora_alternating": optimizer.force_lora_alternating,
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
                    "suffix": dataset.suffix,
                    "backend": backend.backend_name,
                    "num_gpus": backend.NUM_GPUS,
                    "cpus_per_gpu": backend.CPUS_PER_GPU,
                    "gpu_fraction_worker": backend.GPU_FRACTION_VLLM_WORKER,
                    "max_ctx_len": backend.max_model_len,
                }
                wandb.init(project=self.wandb_project, config=config)
                wandb.define_metric("iteration_count")
                wandb.define_metric("train/*", step_metric="iteration_count")
                wandb.define_metric("val/*", step_metric="iteration_count")
                print(f"#-- WandB logging initialized for project: {self.wandb_project} --#")
            except Exception as e:
                print(f"#-- WandB logging initialization failed: {e} --#")
                self.wandb_project = None
            
        print("#-- Trainer initialized. --#")

    def log_train_stats(self, genomes: List[Genome], time_taken: float):
        sum_scores = sum(genome.historical_rewards[-1] for genome in genomes)
        average = sum_scores / len(genomes)
        stddev = (sum((genome.historical_rewards[-1] - average) ** 2 for genome in genomes) / (len(genomes))) ** 0.5
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
                wandb.log({
                    f"train/average_reward": average,
                    f"train/min_reward": worst_genome.historical_rewards[-1],
                    f"train/max_reward": best_genome.historical_rewards[-1],
                    f"train/stddev_reward": stddev,
                    f"train/time_seconds": time_taken,
                    f"train/average_response_length": average_response_length,
                    f"train/samples": sample_table,
                    f"train/learning_rate": self.optimizer.get_lr(self.iteration_count),
                    f"iteration_count": self.iteration_count
                }, step=self.iteration_count)
            except Exception as e:
                print(f"#-- WandB logging failed: {e} --#")
        print(f"#-- Stats: average: {average}, min: {worst_genome.historical_rewards[-1]}, max: {best_genome.historical_rewards[-1]}, stddev: {stddev}, average response length: {average_response_length} --#")
        if self.print_samples:
            print(f"#-- SAMPLE RESPONSE BEST GENOME: --#\n{best_genome.latest_outputs[0]}\n")
            print(f"#-- SAMPLE RESPONSE WORST GENOME: --#\n{worst_genome.latest_outputs[0]}\n") 

    def log_val_stats(self, genome: Genome, time_taken: float):
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

    def train(self):
        while self.iteration_count < self.optimizer.total_steps:
            self.iteration_count += 1

            start_time = time.time()
            inputs = self.dataset.next(population_size=self.population_size, mirror=self.mirror)
            self.backend.generate_outputs(self.genomes, self.dataset.suffix, inputs)

            self.dataset.score_all(self.genomes)

            end_time = time.time()

            print(f"#-- Iteration {self.iteration_count} completed in {end_time - start_time:.2f} seconds --#")
            self.log_train_stats(self.genomes, end_time - start_time)

            self.optimizer.update_self(self.genomes, self.iteration_count)
            self.backend.update(self.optimizer)

            new_genome = Genome()

            if self.validate_every > 0 and self.iteration_count % self.validate_every == 0:
                start_time = time.time()
                prompts = self.dataset.get_test_set()
                self.backend.generate_outputs([new_genome], self.dataset.suffix, prompts)
                self.dataset.score_all([new_genome])
                end_time = time.time()
                print(f"#-- Validation for iteration {self.iteration_count} completed in {end_time - start_time:.2f} seconds --#")
                self.log_val_stats(new_genome, end_time - start_time)
            
            self.genomes = [Genome() for _ in range(self.population_size)]
            for genome in self.genomes:
                genome.mutate_seed(self.optimizer.perturb_scale)
            if self.mirror:
                mirrored_genomes = []
                for genome in self.genomes:
                    mirrored_genomes.append(genome.get_mirrored())
                self.genomes.extend(mirrored_genomes)

    def save_model_seeds(self, filepath: str):
        history = self.optimizer.get_update_history()

        def serialize_structure(obj):
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

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_history, f, indent=4)
            print(f"#-- Successfully saved model seeds to {filepath} --#")
        except Exception as e:
            print(f"#-- Error saving model seeds: {e} --#")

    def restore_model(self, filepath: str):        
        try:
            history = self.load_genome_history(filepath)
            if not history:
                print(f"#-- No history found in {filepath} or file is empty. --#")
                return

            self.optimizer.restore_from_history(history, self.backend)
            
            if isinstance(history, list):
                 self.iteration_count = len(history)
            
            print(f"#-- Successfully loaded model seeds from {filepath}. Resuming from iteration {self.iteration_count} --#")
        except Exception as e:
            print(f"#-- Error loading model seeds: {e} --#")

    def load_genome_history(self, filepath: str) -> Any:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"#-- File {filepath} not found. --#")
            return []

        def reconstruct_structure(obj):
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
        return history