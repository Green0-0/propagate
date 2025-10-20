from typing import List
from libs.backend import Backend
from libs.dataset import Dataset
from libs.genome import Genome, merge_genomes
import time

class SimpleTrainer:
    population_size: int
    learning_rate: float
    seed_weight: float
    backend: Backend
    dataset: Dataset

    genomes: List[Genome]

    iteration_count: int

    def __init__(self, population_size: int, learning_rate: float, seed_weight: float, backend: Backend, dataset: Dataset, output_log_file: str = "logs/output.log", full_output_log_file: str = "logs/full_output.log", reward_log_file: str = "logs/reward.log"):
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

        self.output_log_file = output_log_file
        self.full_output_log_file = full_output_log_file
        self.reward_log_file = reward_log_file

        with open(self.reward_log_file, "w", encoding="utf-8") as f:
            f.write("=== Reward Log Started ===\n")
            f.write("Population Size: {}\n".format(population_size))
            f.write("Learning Rate: {}\n".format(learning_rate))
            f.write("Seed Weight: {}\n".format(seed_weight))
            f.write("Batch Size: {}\n".format(dataset.batch_size))

        print("#-- Trainer initialized. --#")

    def train_step(self):
        self.iteration_count += 1

        print(f"#-- Starting Training Iteration {self.iteration_count} --#\n")
        with open(self.output_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Iteration {self.iteration_count} ===\n\n")
        with open(self.full_output_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Iteration {self.iteration_count} ===\n\n")

        start_time = time.time()
        inputs = self.dataset.next()
        self.backend.generate_outputs(self.genomes, self.dataset.suffix, inputs)

        for genome in self.genomes:
            self.dataset.score(genome)

        new_genome = merge_genomes(
            self.genomes,
            self.learning_rate
        )

        self.backend.update(new_genome)

        self.genomes = [Genome() for _ in range(self.population_size)]
        for genome in self.genomes:
            genome.mutate_seed(self.seed_weight)

        end_time = time.time()

        sum_scores = sum(genome.historical_rewards[-1] for genome in self.genomes)
        average = sum_scores / self.population_size
        stddev = (sum((genome.historical_rewards[-1] - average) ** 2 for genome in self.genomes) / self.population_size) ** 0.5
        min_score = min(genome.historical_rewards[-1] for genome in self.genomes)
        max_score = max(genome.historical_rewards[-1] for genome in self.genomes)
        with open(self.reward_log_file, "a", encoding="utf-8") as f:
            f.write(f"IT {self.iteration_count}: Average {average}, Min {min_score}, Max {max_score}, Stddev {stddev} in {end_time - start_time:.2f} seconds\n")
        print(f"\n\n#-- Iteration average: {average}, min: {min_score}, max: {max_score}, stddev: {stddev} --#")
        print(f"#-- Completed in {end_time - start_time:.2f} seconds --#")