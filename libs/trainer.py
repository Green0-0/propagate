from typing import List
from libs.backend import Backend
from libs.dataset import Dataset
from libs.genome import Genome, merge_genomes

class SimpleTrainer:
    population_size: int
    learning_rate: float
    weight: float
    backend: Backend
    dataset: Dataset

    genomes: List[Genome]

    iteration_count: int

    def __init__(self, population_size: int, learning_rate: float, weight: float, backend: Backend, dataset: Dataset, output_log_file: str = "logs/output.log", full_output_log_file: str = "logs/full_output.log", reward_log_file: str = "logs/reward.log"):
        self.learning_rate = learning_rate
        self.weight = weight
        self.backend = backend
        self.dataset = dataset
        self.population_size = population_size

        self.genomes = [Genome() for _ in range(population_size)]
        for genome in self.genomes:
            genome.mutate_seed(weight)

        self.iteration_count = 0
        print("Trainer initialized with population size:", population_size)

        self.output_log_file = output_log_file
        self.full_output_log_file = full_output_log_file
        self.reward_log_file = reward_log_file

        with open(self.reward_log_file, "a", encoding="utf-8") as f:
            f.write("=== Reward Log Started ===\n")
            f.write("Population Size: {}\n".format(population_size))
            f.write("Learning Rate: {}\n".format(learning_rate))
            f.write("Weight: {}\n".format(weight))
            f.write("Batch Size: {}\n".format(dataset.batch_size))

    def train_step(self):
        with open(self.output_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Iteration {self.iteration_count + 1} ===\n\n")
        with open(self.full_output_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== Iteration {self.iteration_count + 1} ===\n\n")

        self.iteration_count += 1
        print(f"Starting training iteration {self.iteration_count}")

        inputs = self.dataset.next()
        self.backend.generate_outputs(self.genomes, self.dataset.suffix, inputs)

        sum_scores = 0.0
        for genome in self.genomes:
            self.dataset.score(genome)
            sum_scores += genome.historical_rewards[-1]

        print("Average score:", sum_scores / self.population_size)
        with open(self.reward_log_file, "a", encoding="utf-8") as f:
            f.write(f"Iteration {self.iteration_count} Average Score: {sum_scores / self.population_size}\n")
        new_genome = merge_genomes(
            self.genomes,
            self.learning_rate
        )

        self.backend.update(new_genome)

        self.genomes = [Genome() for _ in range(self.population_size)]
        for genome in self.genomes:
            genome.mutate_seed(self.weight)