from abc import ABC, abstractmethod
from typing import List, Dict

from libs.genome import Genome

class Backend(ABC):
    def __init__(self, backend_name: str, NUM_GPUS: int = 1, CPUS_PER_GPU: int = 4, GPU_FRACTION_VLLM_WORKER: float = 0.5, sampler: object = None, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False):
        self.backend_name = backend_name
        self.NUM_GPUS = NUM_GPUS
        self.CPUS_PER_GPU = CPUS_PER_GPU
        self.GPU_FRACTION_VLLM_WORKER = GPU_FRACTION_VLLM_WORKER
        self.sampler = sampler
        self.use_tqdm = use_tqdm
        self.max_model_len = max_model_len
        self.time_self = time_self

    @abstractmethod
    def update(self, genome: Genome):
        """Update the model permanently with a genome as the source."""
        pass

    @abstractmethod
    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[Dict[str, str]]]) -> List[List[str]]:
        """Generate outputs based on the genome and inputs."""
        pass

    @abstractmethod
    def save_weights_to_disk(self, filepath: str):
        """Save the current model weights to disk."""
        pass
