from abc import ABC, abstractmethod
from typing import List, Dict

from propagate.genome import Genome
from propagate.optimizers import Optimizer

class Backend(ABC):
    """Abstract base class for execution backends.

    A backend manages an LLM inference engine (e.g., vLLM), and performs inference on batches of inputs given to it from the trainer.
    
    The backend also manages model perturbation to evaluate a seed, and handles weight updates by taking in as input the optimizer.
    """
    def __init__(self, backend_name: str, model_name: str, NUM_GPUS: int = 1, CPUS_PER_GPU: int = 4, GPU_FRACTION_VLLM_WORKER: float = 0.5, sampler: object = None, use_tqdm: bool = False, max_model_len: int = 4096, time_self: bool = False):
        self.backend_name = backend_name
        self.model_name = model_name
        self.NUM_GPUS = NUM_GPUS
        self.CPUS_PER_GPU = CPUS_PER_GPU
        self.GPU_FRACTION_VLLM_WORKER = GPU_FRACTION_VLLM_WORKER
        self.sampler = sampler
        self.use_tqdm = use_tqdm
        self.max_model_len = max_model_len
        self.time_self = time_self

    @abstractmethod
    def startup(self, trainer=None):
        """Load the model and prepare the backend for inference. Called before training starts."""
        pass

    @abstractmethod
    def update(self, optimizer: Optimizer):
        """Update the model permanently with the optimizer."""
        pass

    @abstractmethod
    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[List[Dict[str, str]]]]):
        """Generate outputs based on the genome and inputs. Note that the input is a list of batches (one per genome) of sharegpt data"""
        pass

    @abstractmethod
    def save_weights_to_disk(self, filepath: str):
        """Save the current model weights to disk."""
        pass
