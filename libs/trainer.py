from typing import List
from libs.backend import Backend
from libs.dataset import Dataset
from libs.genome import Genome, merge_genomes

class SimpleTrainer:
    population_size: int
    learning_rate: float
    weight_decay: float
    backend: Backend
    dataset: Dataset

    genomes: List[Genome]

    def __init__(self, learning_rate: float, weight_decay: float, backend: Backend, dataset: Dataset, population_size: int = 20):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.backend = backend
        self.dataset = dataset
        self.population_size = population_size

        self.genomes = [Genome() for _ in range(population_size)]
        for genome in self.genomes:
            genome.mutate_seed()

    def train_step(self):
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
            genome.mutate_seed()