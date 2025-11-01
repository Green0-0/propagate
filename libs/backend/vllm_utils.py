import gc
import re
import time
import torch
from ray.util import collective

from libs.genome import Genome

class WorkerExtension:
    def init_collective_group(self, world_size: int, rank: int, backend: str = "nccl"):
        self.collective_group_name = "weight_sync_group"
        self.world_size = world_size
        self.rank = rank
        
        if collective.is_group_initialized(self.collective_group_name):
            collective.destroy_collective_group(self.collective_group_name)
        
        collective.init_collective_group(
            world_size=world_size,
            rank=rank,
            backend=backend,
            group_name=self.collective_group_name,
        )
        print(f"#-- Worker {rank} collective group initialized. --#")
        return True

    def perform_all_reduce_sync(self):
        """Performs AllReduce to average weights across all workers."""
        if not collective.is_group_initialized(self.collective_group_name) or self.world_size <= 1:
            return True
            
        with torch.no_grad():
            for name, p in self.model_runner.model.named_parameters():
                if p.requires_grad:
                    collective.allreduce(
                        p.data, 
                        group_name=self.collective_group_name
                    )
                    p.data.div_(self.world_size)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def destroy_collective_group(self):
        if collective.is_group_initialized(self.collective_group_name):
            collective.destroy_collective_group(self.collective_group_name)
        return True

    def __del__(self):
        self.destroy_collective_group()

    @torch.inference_mode()
    def perturb_self_weights(self, genome: Genome):
        device = next(self.model_runner.model.parameters()).device
        for seed, weight in zip(genome.seeds, genome.seed_weights):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
            for _, p in self.model_runner.model.named_parameters():
                noise = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
                noise.mul_(weight)
                p.data.add_(noise)
                del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def restore_self_weights(self, genome: Genome):
        device = next(self.model_runner.model.parameters()).device
        for seed, weight in zip(genome.seeds, genome.seed_weights):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
            for _, p in self.model_runner.model.named_parameters():
                noise = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
                noise.mul_(weight)
                p.data.sub_(noise)
                del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def save_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True