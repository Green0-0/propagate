from typing import List
from libs.backend import Backend
from libs.dataset import Dataset
from libs.genome import Genome, merge_genomes
import time
import wandb

class SimpleTrainer:
    population_size: int
    learning_rate: float
    seed_weight: float
    backend: Backend
    dataset: Dataset

    genomes: List[Genome]

    iteration_count: int

    wandb_project: str

    def __init__(self, population_size: int, learning_rate: float, seed_weight: float, backend: Backend, dataset: Dataset, wandb_project: str = None, validate_every: int = 0, print_samples: bool = False):
        print("#-- Initializing Trainer [SimpleTrainer] --#")
        print(f"#-- Population Size: {population_size}, Learning Rate: {learning_rate}, Weight: {seed_weight} --#")
        self.learning_rate = learning_rate
        self.seed_weight = seed_weight
        self.backend = backend
        self.dataset = dataset
        self.population_size = population_size

        self.genomes = [Genome() for _ in range(population_size)]
        for genome in self.genomes:
            genome.mutate_seed(seed_weight)

        self.iteration_count = 0

        self.wandb_project = wandb_project
        self.validate_every = validate_every
        self.print_samples = print_samples

        if self.wandb_project is not None and self.wandb_project != "":
            wandb.login()
            config = {
                "population_size": population_size,
                "learning_rate": learning_rate,
                "seed_weight": seed_weight,
                "batch_size": dataset.batch_size
            }
            if hasattr(backend, "NUM_GPUS"):
                config["NUM_GPUS"] = backend.NUM_GPUS
            if hasattr(backend, "CPUS_PER_GPU"):
                config["CPUS_PER_GPU"] = backend.CPUS_PER_GPU
            if hasattr(backend, "GPU_FRACTION_VLLM_WORKER"):
                config["GPU_FRACTION_VLLM_WORKER"] = backend.GPU_FRACTION_VLLM_WORKER
            wandb.init(project=self.wandb_project, config=config)
            wandb.define_metric("iteration_count", step_metric=True)
            wandb.define_metric("train/*", step_metric="iteration_count")
            wandb.define_metric("val/*", step_metric="iteration_count")
            print(f"#-- WandB logging initialized for project: {self.wandb_project} --#")
            
        print("#-- Trainer initialized. --#")

    def log_train_stats(self, genomes: List[Genome], time_taken: float):
        sum_scores = sum(genome.historical_rewards[-1] for genome in genomes)
        average = sum_scores / len(genomes)
        stddev = (sum((genome.historical_rewards[-1] - average) ** 2 for genome in genomes) / len(genomes)) ** 0.5
        average_response_length = sum([sum(len(response.split()) for response in genome.latest_outputs) for genome in genomes]) / len(genomes) / len(genomes[0].latest_outputs)

        best_genome = max(genomes, key=lambda g: g.historical_rewards[-1])
        worst_genome = min(genomes, key=lambda g: g.historical_rewards[-1])
        if self.wandb_project is not None and self.wandb_project != "":
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
                f"iteration_count": self.iteration_count
            }, step=self.iteration_count)
        print(f"#-- Stats: average: {average}, min: {worst_genome.historical_rewards[-1]}, max: {best_genome.historical_rewards[-1]}, stddev: {stddev}, average response length: {average_response_length} --#")
        if self.print_samples:
            print(f"#-- SAMPLE RESPONSE BEST GENOME: --#\n{best_genome.latest_outputs[0]}\n")
            print(f"#-- SAMPLE RESPONSE WORST GENOME: --#\n{worst_genome.latest_outputs[0]}\n") 

    def log_val_stats(self, genome: Genome, time_taken: float):
        score = genome.historical_rewards[-1]
        score_stddev = sum((genome.latest_rewards[i] - score) ** 2 for i in range(len(genome.latest_rewards))) / len(genome.latest_rewards)
        average_response_length = sum(len(response.split()) for response in genome.latest_outputs) / len(genome.latest_outputs)
        sample_response = genome.latest_outputs[0]
        if self.wandb_project is not None and self.wandb_project != "":
            wandb.log({
                f"val/validation_score": score,
                f"val/validation_stddev": score_stddev,
                f"val/time_seconds": time_taken,
                f"val/average_response_length": average_response_length,
                f"val/sample_response": wandb.Table(data=[[sample_response]], columns=["response"]),
                f"iteration_count": self.iteration_count
            }, step=self.iteration_count)
        print(f"#-- Stats: reward: {score}, response length: {average_response_length} --#")
        if self.print_samples:
            print(f"#-- SAMPLE RESPONSE: --#\n{sample_response}\n")

    def train_step(self):
        self.iteration_count += 1

        start_time = time.time()
        inputs = self.dataset.next()
        self.backend.generate_outputs(self.genomes, self.dataset.suffix, inputs)

        for genome in self.genomes:
            self.dataset.score(genome)

        end_time = time.time()

        print(f"#-- Iteration {self.iteration_count} completed in {end_time - start_time:.2f} seconds --#")
        self.log_train_stats(self.genomes, end_time - start_time)

        new_genome = merge_genomes(self.genomes, self.learning_rate)

        if self.validate_every > 0 and self.iteration_count % self.validate_every == 0:
            start_time = time.time()
            prompts = self.dataset.get_test_set()
            self.backend.generate_outputs([new_genome], self.dataset.suffix, prompts)
            self.dataset.score(new_genome)
            end_time = time.time()
            print(f"#-- Validation for iteration {self.iteration_count} completed in {end_time - start_time:.2f} seconds --#")
            self.log_val_stats(new_genome, end_time - start_time)
        self.backend.update(new_genome)
        
        self.genomes = [Genome() for _ in range(self.population_size)]
        for genome in self.genomes:
            genome.mutate_seed(self.seed_weight)