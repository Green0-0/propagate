from abc import ABC, abstractmethod
from typing import List, Dict

from libs.genome import Genome

class Backend(ABC):
    @abstractmethod
    def update(self, genome: Genome):
        """Update the model permanently with a genome as the source."""
        pass

    @abstractmethod
    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[Dict[str, str]]]) -> List[List[str]]:
        """Generate outputs based on the genome and inputs."""
        pass

