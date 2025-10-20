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

    def __init__(self, population_size: int, learning_rate: float, weight: float, backend: Backend, dataset: Dataset):
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

    def train_step(self):
        self.iteration_count += 1
        print(f"Starting training iteration {self.iteration_count}")

        inputs = self.dataset.next()
        self.backend.generate_outputs(self.genomes, self.dataset.suffix, inputs)

        sum_scores = 0.0
        for genome in self.genomes:
            self.dataset.score(genome)
            sum_scores += genome.historical_rewards[-1]

        print("Average score:", sum_scores / self.population_size)
        new_genome = merge_genomes(
            self.genomes,
            self.learning_rate
        )

        self.backend.update(new_genome)

        self.genomes = [Genome() for _ in range(self.population_size)]
        for genome in self.genomes:
            genome.mutate_seed(self.weight)