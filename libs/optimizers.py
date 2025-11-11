from abc import ABC, abstractmethod
import copy
from typing import List, Dict
from collections import OrderedDict

import torch

from libs.genome import Genome

import math

class Optimizer(ABC):
    def __init__(self, optimizer_name, total_steps: int, learning_rate: float, seed_weight: float, warmup_steps: int = 0, scheduler: str = "none", norm_by_mean : bool = True, norm_by_stddev : bool = True):
        self.optimizer_name = optimizer_name
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.seed_weight = seed_weight
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.norm_by_mean = norm_by_mean
        self.norm_by_stddev = norm_by_stddev
        self.rep_genome = Genome()
        self.update_history = []

    @abstractmethod
    def update_self(self, genomes: List[Genome], current_step: int):
        """Updates the optimizer's internal state based on the provided genomes and current step."""
        pass

    @abstractmethod
    def step_update(self, tensor: torch.Tensor, random_offset: int):
        """Performs a step update on the provided tensor."""
        pass

    @abstractmethod
    def get_representative(self) -> Genome:
        """Returns a representative genome for the current step."""
        pass

    @abstractmethod
    def get_update_history(self):
        """Returns a list of lists of genomes representing the history of updates."""
        pass

    @abstractmethod
    def restore_from_history(self, history, backend):
        """Restores the optimizer's state from the provided history of updates."""
        pass

    def get_lr(self, current_step: int) -> float:
        """Returns the learning rate based on the current step, applying the warmup and scheduler."""
        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            return self.learning_rate * (current_step / self.warmup_steps)

        t = max(0.0, min(1.0, (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))

        sched = (self.scheduler or "none").lower()
        if sched in ("none", "constant"):
            return self.learning_rate
        if sched.startswith("linear"):
            return self.learning_rate * (1.0 - t)
        if sched.startswith("cosine"):
            return self.learning_rate * 0.5 * (1.0 + math.cos(math.pi * t))
        if sched.startswith("exponential"):
            return self.learning_rate * math.exp(-2 * math.sqrt(t/(self.total_steps - self.warmup_steps)))
        raise ValueError(f"Unknown scheduler: {self.scheduler}")
    
class SimpleOpt(Optimizer):
    def __init__(self, total_steps: int, learning_rate: float, seed_weight: float, warmup_steps: int = 0, scheduler: str = "none", norm_by_mean : bool = True, norm_by_stddev : bool = True, optimizer_name: str = "SimpleOptimizer"):
        super().__init__(optimizer_name, total_steps, learning_rate, seed_weight, warmup_steps, scheduler, norm_by_mean=norm_by_mean, norm_by_stddev=norm_by_stddev)

    def update_self(self, genomes: List[Genome], current_step: int):
        self.rep_genome = Genome()
        lr = self.get_lr(current_step)
        reward_mean = sum([g.historical_rewards[-1] for g in genomes]) / len(genomes)
        reward_stddev = (sum([(g.historical_rewards[-1] - reward_mean) ** 2 for g in genomes]) / len(genomes)) ** 0.5
        new_seeds = {}
        old_seeds_count = {}
        for g in genomes:
            for i in range(len(g.seeds)):
                seed = g.seeds[i]
                weight = g.seed_weights[i]
                if seed not in new_seeds:
                    if i < g.starting_index:
                        old_seeds_count[seed] = 1
                        new_seeds[seed] = weight
                    else:
                        update_value = 0.0
                        if self.norm_by_mean:
                            update_value = math.copysign(1, weight) * lr * (1/len(genomes)) * (g.historical_rewards[-1] - reward_mean)
                        else:
                            update_value = math.copysign(1, weight) * lr * (1/len(genomes)) * g.historical_rewards[-1]
                        if self.norm_by_stddev:
                            update_value /= (reward_stddev + 1e-8)
                        new_seeds[seed] = update_value
                else:
                    if i < g.starting_index:
                        old_seeds_count[seed] += 1
                        new_seeds[seed] += weight
                    else:
                        update_value = 0.0
                        if self.norm_by_mean:
                            update_value = math.copysign(1, weight) * lr * (1/len(genomes)) * (g.historical_rewards[-1] - reward_mean)
                        else:
                            update_value = math.copysign(1, weight) * lr * (1/len(genomes)) * g.historical_rewards[-1]
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

    def step_update(self, tensor: torch.Tensor, random_offset: int):
        """Apply a single optimization step to the given tensor."""
        gen = torch.Generator(device=tensor.device)
        noise = torch.empty_like(tensor)
        for seed, weight in zip(self.rep_genome.seeds, self.rep_genome.seed_weights):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
            tensor.add_(noise, alpha=weight)
        del noise

    def get_representative(self) -> Genome:
        return self.rep_genome
    
    def get_update_history(self) -> List[List[Genome]]:
        return self.update_history
    
    def restore_from_history(self, history, backend):
        for step_genome in history:
            self.rep_genome = step_genome
            backend.update(self)
        self.rep_genome = Genome()

class MomentumOpt(Optimizer):
    velocity_seeds: OrderedDict
    cutoff_seeds: int

    def __init__(self, total_steps: int, learning_rate: float, seed_weight: float, warmup_steps: int = 0, scheduler: str = "none", momentum: float = 0.6, cutoff_seeds = 300, norm_by_mean : bool = True, norm_by_stddev : bool = True, optimizer_name: str = "MomentumOptimizer"):
        super().__init__(optimizer_name, total_steps, learning_rate, seed_weight, warmup_steps, scheduler, norm_by_mean=norm_by_mean, norm_by_stddev=norm_by_stddev)
        self.momentum = momentum
        self.velocity_seeds = OrderedDict()
        self.cutoff_seeds = cutoff_seeds

    def update_self(self, genomes: List[Genome], current_step: int):
        self.rep_genome = Genome()
        lr = self.get_lr(current_step)
        for seed in self.velocity_seeds:
            self.velocity_seeds[seed] *= self.momentum
       
        reward_mean = sum([g.historical_rewards[-1] for g in genomes]) / len(genomes)
        reward_stddev = (sum([(g.historical_rewards[-1] - reward_mean) ** 2 for g in genomes]) / len(genomes)) ** 0.5
        old_seeds = {}
        old_seeds_count = {}
        for g in genomes:
            for i in range(len(g.seeds)):
                # If this is a new seed (it's unlikely we have seen it before so we do not optimize for duplicates), add it to the velocity. If this is an old seed (generated by a gradient step), ignore it because we keep a running history of all old seeds.
                seed = g.seeds[i]
                weight = g.seed_weights[i]
                if i < g.starting_index:
                    if seed not in old_seeds:
                        old_seeds_count[seed] = 1
                        old_seeds[seed] = weight                        
                    else:
                        old_seeds_count[seed] += 1
                        old_seeds[seed] += weight
                else:
                    new_seed_value = 0
                    if self.norm_by_mean:
                        new_seed_value = math.copysign(1, weight) * lr * (1/len(genomes)) * (g.historical_rewards[-1] - reward_mean)
                    else:
                        new_seed_value = math.copysign(1, weight) * lr * (1/len(genomes)) * g.historical_rewards[-1]
                    if self.norm_by_stddev:
                        new_seed_value /= (reward_stddev + 1e-8)
                    if seed in self.velocity_seeds:
                        self.velocity_seeds[seed] += new_seed_value
                    else:
                        self.velocity_seeds[seed] = new_seed_value

        self.velocity_seeds = OrderedDict(list(self.velocity_seeds.items())[-self.cutoff_seeds:])

        for seed, count in old_seeds_count.items():
            old_seeds[seed] /= count

        for seed, weight in old_seeds.items():
            self.rep_genome.seeds.append(seed)
            self.rep_genome.seed_weights.append(weight)
            self.rep_genome.historical_rewards.append(float('-inf'))
        for seed, weight in self.velocity_seeds.items():
            self.rep_genome.seeds.append(seed)
            self.rep_genome.seed_weights.append(weight)
            self.rep_genome.historical_rewards.append(float('-inf'))
        self.rep_genome.starting_index = len(self.rep_genome.seeds)
        self.update_history.append(copy.deepcopy(self.rep_genome))

    def step_update(self, tensor: torch.Tensor, random_offset: int):
        gen = torch.Generator(device=tensor.device)
        noise = torch.empty_like(tensor)
        for seed, weight in zip(self.rep_genome.seeds, self.rep_genome.seed_weights):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
            tensor.add_(noise, alpha=weight)
        del noise

    def get_representative(self) -> Genome:
        return self.rep_genome
    
    def get_update_history(self) -> List[List[Genome]]:
        return self.update_history
    
    def restore_from_history(self, history, backend):
        for step_genome in history:
            self.rep_genome = step_genome
            backend.update(self)
        self.rep_genome = Genome()

class MuonOpt(MomentumOpt):
    def __init__(self, total_steps: int, learning_rate: float, seed_weight: float, warmup_steps: int = 0, scheduler: str = "none", momentum: float = 0.6, cutoff_seeds = 2000, norm_by_mean : bool = True, norm_by_stddev : bool = True, optimizer_name: str = "MuonOptimizer"):
        super().__init__(total_steps, learning_rate, seed_weight, warmup_steps, scheduler, norm_by_mean=norm_by_mean, norm_by_stddev=norm_by_stddev, optimizer_name=optimizer_name)
        self.momentum = momentum
        self.velocity_seeds = OrderedDict()
        self.cutoff_seeds = cutoff_seeds

    def step_update(self, tensor: torch.Tensor, random_offset: int):
        gen = torch.Generator(device=tensor.device)
        noise = torch.empty_like(tensor)
        total_noise = torch.empty_like(tensor)
        for seed, weight in zip(self.rep_genome.seeds, self.rep_genome.seed_weights):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
            total_noise.add_(noise, alpha=weight)
        tensor.add_(self.newtonschulz5(total_noise), alpha=weight)
        del noise
        del total_noise

    def newtonschulz5(self, G: torch.Tensor, steps=5, eps=1e-7):
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        X /= (X.norm() + eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X