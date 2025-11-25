import gc
import time
import torch

from libs.genome import Genome

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from libs.optimizers import Optimizer

def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

class WorkerExtension:
    @torch.inference_mode()
    def update_weights(self, optimizer: Optimizer):
        """Update the model's weights using the provided optimizer."""
        rand_counter = 0
        for id, p in self.model_runner.model.named_parameters():
            optimizer.step_update(p.data, rand_counter, id)
            rand_counter += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def perturb_self_weights(self, genome: Genome):
        """Perturb the model's weights using the provided genome."""
        for seed, weight in zip(genome.seeds, genome.seed_weights):
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
        """Restore the model's weights by removing the perturbations introduced by the genome."""
        for seed, weight in zip(genome.seeds, genome.seed_weights):
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
        self.inter_pg = _stateless_init_process_group(master_address, master_port, rank, world_size, self.device)
        return True

    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_weights_to_disk(self, filepath):
        """Save the model's weights to disk."""
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True