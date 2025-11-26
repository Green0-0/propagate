from abc import ABC, abstractmethod
import copy
from typing import Any, List, Dict, Tuple
from collections import OrderedDict

import torch

from libs.genome import Genome
from libs.optimizers import Optimizer

import math

class SteinOpt(Optimizer):
    def __init__(self, total_steps: int, learning_rate: float, seed_weight: float, warmup_steps: int = 0, scheduler: str = "none", norm_by_mean : bool = True, norm_by_stddev : bool = True, optimizer_name: str = "SimpleOptimizer", force_lora_alternating: bool = False):
        super().__init__(optimizer_name, total_steps, learning_rate, seed_weight, warmup_steps, scheduler, norm_by_mean=norm_by_mean, norm_by_stddev=norm_by_stddev, force_lora_alternating=force_lora_alternating)

    def update_self(self, genomes: List[Genome], current_step: int):
        self.rep_genome = Genome()
        self.last_lr = self.get_lr(current_step)
        reward_mean = sum([g.historical_rewards[-1] for g in genomes]) / len(genomes)
        reward_stddev = (sum([(g.historical_rewards[-1] - reward_mean) ** 2 for g in genomes]) / len(genomes)) ** 0.5
        new_seeds = {}
        old_seeds_count = {}
        for g in genomes:
            added_old_seed = False
            for i in range(len(g.seeds)):
                seed = g.seeds[i]
                weight = g.seed_weights[i]
                if seed not in new_seeds:
                    if i < g.starting_index:
                        added_old_seed = True
                        old_seeds_count[seed] = 1
                        new_seeds[seed] = weight
                    else:
                        update_value = 0.0
                        if self.norm_by_mean:
                            update_value = math.copysign(1, weight) * (1/len(genomes)) * (g.historical_rewards[-1] - reward_mean)
                        else:
                            update_value = math.copysign(1, weight) * (1/len(genomes)) * g.historical_rewards[-1]
                        if self.norm_by_stddev:
                            update_value /= (reward_stddev + 1e-8)
                        new_seeds[seed] = update_value
                else:
                    if i < g.starting_index:
                        if not added_old_seed:
                            added_old_seed = True
                            old_seeds_count[seed] += 1
                        new_seeds[seed] += weight
                    else:
                        update_value = 0.0
                        if self.norm_by_mean:
                            update_value = math.copysign(1, weight) * (1/len(genomes)) * (g.historical_rewards[-1] - reward_mean)
                        else:
                            update_value = math.copysign(1, weight) * (1/len(genomes)) * g.historical_rewards[-1]
                        if self.norm_by_stddev:
                            update_value /= (reward_stddev + 1e-8)
                        new_seeds[seed] += update_value
        for seed, count in old_seeds_count.items():
            new_seeds[seed] /= count

        for seed, weight in new_seeds.items():
            self.rep_genome.seeds.append(seed)
            self.rep_genome.seed_weights.append(weight)
            self.rep_genome.historical_rewards.append(float('-inf'))
        self.rep_genome.starting_index = len(self.rep_genome.seeds)
        self.update_history.append(copy.deepcopy(self.rep_genome))

    def step_update(self, tensor: torch.Tensor, random_offset: int, parameter_id, lr_scalar: float = 1, state: Dict = None):
        """Apply a single optimization step to the given tensor."""
        gen = torch.Generator(device=tensor.device)
        ghat = torch.zeros_like(tensor)
        noise = torch.empty_like(tensor)
        for seed, weight in zip(self.rep_genome.seeds, self.rep_genome.seed_weights):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
            ghat.add_(noise, alpha=float(weight))
        
        norm = torch.norm(ghat)
        uhat = ghat / (norm + 1e-8)
        s1 = 0
        s2 = 0
        for seed, weight in zip(self.rep_genome.seeds, self.rep_genome.seed_weights):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
            t = torch.dot(uhat.view(-1), noise.view(-1))
            s1 += weight * t
            s2 += weight * (t ** 2 - 1) 
        s1 /= len(self.rep_genome.seeds)
        s2 /= len(self.rep_genome.seeds)
        k = max(s2, 1e-6)
        n = s1/(k*norm)
        for seed, weight in zip(self.rep_genome.seeds, self.rep_genome.seed_weights):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
            tensor.data.add_(noise, alpha=n * lr_scalar * float(weight) * self.last_lr)
        del ghat
        del uhat
        del noise

    def get_representative(self) -> Genome:
        raise NotImplementedError("SteinOpt does not support a single representative genome.")
    
    def get_update_history(self) -> List[Genome]:
        return self.update_history
    
    def restore_from_history(self, history, backend):
        for step_genome in history:
            self.rep_genome = step_genome
            backend.update(self)
        self.rep_genome = Genome()