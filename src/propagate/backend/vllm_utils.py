import gc
import time
import torch

from propagate.genome import Genome

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from propagate.optimizers import Optimizer

def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

class WorkerExtension:
    """A Ray actor extension that runs on vLLM workers. 
    It handles weight updates, perturbations, and weight synchronization across workers.
    
    Attributes
    ----------
    optimizer_state : Dict[str, Any]
        A dictionary storing the optimizer state for each parameter.
    inter_pg : StatelessProcessGroup
        The process group for inter-engine communication.
    """
    @torch.inference_mode()
    def update_weights(self, optimizer: Optimizer):
        """Update the model's weights using the provided optimizer.
        
        Args:
            optimizer (Optimizer): The optimizer to use for the update.
        """
        # check if the worker has an optimizer state
        if not hasattr(self, 'optimizer_state'):
            self.optimizer_state = {}
        rand_counter = 0
        for id, p in self.model_runner.model.named_parameters():
            optimizer.step_update(p.data, rand_counter, id, lr_scalar=float(1.0), state=self.optimizer_state)
            rand_counter += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def perturb_self_weights(self, genome: Genome):
        """Perturb the model's weights using the provided genome.
        Iterates through the genome's seeds and perturb scales, generating noise and adding it to the weights.
        
        Args:
           genome (Genome): The genome containing the seeds and scales for perturbation.
        """
        for seed, weight in zip(genome.seeds, genome.perturb_scales):
            rand_counter = 0
            for _, p in self.model_runner.model.named_parameters():
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed) + rand_counter)
                rand_counter += 1

                noise = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
                p.data.add_(noise, alpha=weight)
                del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def restore_self_weights(self, genome: Genome):
        """Restore the model's weights by removing the perturbations introduced by the genome.
        Essentially replicates the perturbation process but subtracts the noise instead of adding it.
        
        Args:
            genome (Genome): The genome containing the seeds and scales to reverse.
        """
        for seed, weight in zip(genome.seeds, genome.perturb_scales):
            rand_counter = 0
            for _, p in self.model_runner.model.named_parameters():
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed) + rand_counter)
                rand_counter += 1
                
                noise = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
                p.data.sub_(noise, alpha=weight)
                del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        """Initialize the process group for inter-engine communication (NCCL).
        
        Args:
            master_address (str): The IP address of the master node.
            master_port (int): The port of the master node.
            rank (int): The rank of this worker.
            world_size (int): The total number of workers.
            
        Returns:
            bool: True if initialization was successful.
        """
        self.inter_pg = _stateless_init_process_group(master_address, master_port, rank, world_size, self.device)
        return True

    def broadcast_all_weights(self, src_rank: int):
        """Broadcast all model weights from the source rank to all other workers.
        
        Args:
            src_rank (int): The rank of the worker to broadcast from.
        
        Returns:
            bool: True if broadcast was successful.
        """
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_weights_to_disk(self, filepath):
        """Save the model's weights to disk.
        
        Args:
            filepath (str): The path to save the state dict to.
        
        Returns:
            bool: True if save was successful.
        """
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True

    def load_weights_from_disk(self, filepath):
        """Load the model's weights from disk.
        
        Args:
            filepath (str): The path to load the state dict from.
            
        Returns:
            bool: True if load was successful.
        """
        state_dict = torch.load(filepath, map_location=self.device)
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(state_dict[name].to(self.device))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True