from abc import ABC, abstractmethod
import copy
from typing import Any, List, Dict, Tuple
from collections import OrderedDict

import torch

from libs.genome import Genome

import math

class Optimizer(ABC):
    def __init__(self, optimizer_name, total_steps: int, learning_rate: float, perturb_scale: float, warmup_steps: int = 0, scheduler: str = "none", norm_by_mean : bool = True, norm_by_stddev : bool = True, force_lora_alternating: bool = False):
        self.optimizer_name = optimizer_name
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.perturb_scale = perturb_scale
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.norm_by_mean = norm_by_mean
        self.norm_by_stddev = norm_by_stddev
        self.force_lora_alternating = force_lora_alternating
        self.rep_genome = Genome()
        self.update_history = []

    @abstractmethod
    def update_self(self, genomes: List[Genome], current_step: int):
        """Updates the optimizer's internal state based on the provided genomes and current step."""
        pass

    @abstractmethod
    def step_update(self, tensor: torch.Tensor, random_offset: int, parameter_id, lr_scalar: float = 1, state: Dict = None):
        """Performs a step update on the provided tensor."""
        pass

    @abstractmethod
    def get_representative(self) -> Genome:
        """Returns a representative genome for the current step."""
        pass

    @abstractmethod
    def get_update_history(self) -> Any:
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
            return self.learning_rate * math.exp(-2 * math.sqrt(t))
        raise ValueError(f"Unknown scheduler: {self.scheduler}")
    
class SimpleOpt(Optimizer):
    def __init__(self, total_steps: int, learning_rate: float, perturb_scale: float, warmup_steps: int = 0, scheduler: str = "none", norm_by_mean : bool = True, norm_by_stddev : bool = True, force_lora_alternating: bool = False):
        super().__init__("SimpleOptimizer", total_steps, learning_rate, perturb_scale, warmup_steps, scheduler, norm_by_mean=norm_by_mean, norm_by_stddev=norm_by_stddev, force_lora_alternating=force_lora_alternating)

    def update_self(self, genomes: List[Genome], current_step: int):
        self.rep_genome = Genome()
        lr = self.get_lr(current_step)
        reward_mean = sum([g.historical_rewards[-1] for g in genomes]) / len(genomes)
        reward_stddev = (sum([(g.historical_rewards[-1] - reward_mean) ** 2 for g in genomes]) / len(genomes)) ** 0.5
        new_seeds = {}
        old_seeds_count = {}
        for g in genomes:
            added_old_seed = False
            for i in range(len(g.seeds)):
                seed = g.seeds[i]
                weight = g.perturb_scales[i]
                if seed not in new_seeds:
                    if i < g.starting_index:
                        added_old_seed = True
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
                        if not added_old_seed:
                            added_old_seed = True
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
            self.rep_genome.perturb_scales.append(weight)
            self.rep_genome.historical_rewards.append(float('-inf'))
        self.rep_genome.starting_index = len(self.rep_genome.seeds)
        self.update_history.append(copy.deepcopy(self.rep_genome))

    def step_update(self, tensor: torch.Tensor, random_offset: int, parameter_id, lr_scalar: float = 1, state: Dict = None):
        """Apply a single optimization step to the given tensor."""
        gen = torch.Generator(device=tensor.device)
        noise = torch.empty_like(tensor)
        for seed, weight in zip(self.rep_genome.seeds, self.rep_genome.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
            tensor.add_(noise, alpha=float(weight * lr_scalar))
        del noise

    def get_representative(self) -> Genome:
        return self.rep_genome
    
    def get_update_history(self) -> List[Genome]:
        return self.update_history
    
    def restore_from_history(self, history, backend):
        for step_genome in history:
            self.rep_genome = step_genome
            backend.update(self)
        self.rep_genome = Genome()

class MomentumOpt(Optimizer):
    velocity_seeds_steps: List[List[Tuple[int, float]]]
    cutoff_steps: int

    def __init__(self, total_steps: int, learning_rate: float, perturb_scale: float, warmup_steps: int = 0, scheduler: str = "none", momentum: float = 0.6, cutoff_steps = 30, norm_by_mean : bool = True, norm_by_stddev : bool = True, force_lora_alternating: bool = False, optimizer_name: str = "MomentumOptimizer"):
        super().__init__(f"{optimizer_name} (momentum={momentum})", total_steps, learning_rate, perturb_scale, warmup_steps, scheduler, norm_by_mean=norm_by_mean, norm_by_stddev=norm_by_stddev, force_lora_alternating=force_lora_alternating)
        self.momentum = momentum
        self.velocity_seeds_steps = []
        self.cutoff_steps = cutoff_steps
        self.last_lr = 0
        self.last_step = 0
        self.force_disable_lr = False

    def update_self(self, genomes: List[Genome], current_step: int):
        self.rep_genome = Genome()
        self.last_lr = self.get_lr(current_step)
        self.last_step = current_step
        lr_used = self.last_lr if not self.force_disable_lr else 1.0
        
        reward_mean = sum([g.historical_rewards[-1] for g in genomes]) / len(genomes)
        reward_stddev = (sum([(g.historical_rewards[-1] - reward_mean) ** 2 for g in genomes]) / (len(genomes))) ** 0.5

        new_seeds = {}
        old_seeds = {}
        old_seeds_count = {}
        for g in genomes:
            added_old_seed = False
            for i in range(len(g.seeds)):
                # If this is a new seed (it's unlikely we have seen it before so we do not optimize for duplicates), add it to the velocity. If this is an old seed (generated by a gradient step), ignore it because we keep a running history of all old seeds.
                seed = g.seeds[i]
                weight = g.perturb_scales[i]
                if i < g.starting_index:
                    if seed not in old_seeds:
                        added_old_seed = True
                        old_seeds_count[seed] = 1
                        old_seeds[seed] = weight                        
                    else:
                        if not added_old_seed:
                            added_old_seed = True
                            old_seeds_count[seed] += 1
                        old_seeds[seed] += weight
                else:
                    new_seed_value = 0
                    if self.norm_by_mean:
                        new_seed_value = math.copysign(1, weight) * (1/len(genomes)) * (g.historical_rewards[-1] - reward_mean)
                    else:
                        new_seed_value = math.copysign(1, weight) * (1/len(genomes)) * g.historical_rewards[-1]
                    if self.norm_by_stddev:
                        new_seed_value /= (reward_stddev + 1e-8)
                    if seed in new_seeds:
                        new_seeds[seed] += new_seed_value
                    else:
                        new_seeds[seed] = new_seed_value
        
        for seed, count in old_seeds_count.items():
            old_seeds[seed] /= count

        self.velocity_seeds_steps.append([(seed, new_seeds[seed]) for seed in new_seeds.keys()])
        if len(self.velocity_seeds_steps) > self.cutoff_steps:
            self.velocity_seeds_steps.pop(0)
        for seed, weight in old_seeds.items():
            self.rep_genome.seeds.append(seed)
            self.rep_genome.perturb_scales.append(weight)
            self.rep_genome.historical_rewards.append(float('-inf'))

        # Traverse velocity seeds in reverse
        accumulated_coefficient = 1
        current_head_idx = len(self.velocity_seeds_steps) - 1
        for step_idx in reversed(range(len(self.velocity_seeds_steps))):
            if step_idx % 2 != current_head_idx % 2 and self.force_lora_alternating:
                continue
            for idx in range(len(self.velocity_seeds_steps[step_idx])):
                seed, weight = self.velocity_seeds_steps[step_idx][idx]
                self.rep_genome.seeds.append(seed)
                self.rep_genome.perturb_scales.append(weight * accumulated_coefficient * lr_used)
                self.rep_genome.historical_rewards.append(float('-inf'))
            accumulated_coefficient *= self.momentum
        self.rep_genome.starting_index = len(self.rep_genome.seeds)
        self.update_history.append(copy.deepcopy(self.rep_genome))

    def step_update(self, tensor: torch.Tensor, random_offset: int, parameter_id, lr_scalar: float = 1, state: Dict = None):
        gen = torch.Generator(device=tensor.device)
        noise = torch.empty_like(tensor)
        for seed, weight in zip(self.rep_genome.seeds, self.rep_genome.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
            tensor.add_(noise, alpha=float(weight * lr_scalar))
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
    def __init__(self, total_steps: int, learning_rate: float, perturb_scale: float, warmup_steps: int = 0, scheduler: str = "none", momentum: float = 0.6, cutoff_steps = 30, norm_by_mean : bool = True, norm_by_stddev : bool = True, force_lora_alternating: bool = False):
        super().__init__(total_steps, learning_rate, perturb_scale, warmup_steps, scheduler, momentum=momentum, cutoff_steps=cutoff_steps, norm_by_mean=norm_by_mean, norm_by_stddev=norm_by_stddev, force_lora_alternating=force_lora_alternating, optimizer_name="MuonOptimizer")
        self.force_disable_lr = True

    def step_update(self, tensor: torch.Tensor, random_offset: int, parameter_id, lr_scalar: float = 1):
        gen = torch.Generator(device=tensor.device)
        noise = torch.empty_like(tensor)
        total_noise = torch.zeros_like(tensor)
        for seed, weight in zip(self.rep_genome.seeds, self.rep_genome.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
            total_noise.add_(noise, alpha=weight)
        if tensor.ndim == 2:
            tensor.add_(self.newtonschulz5(total_noise), alpha=float(self.last_lr * lr_scalar))
        else:
            tensor.add_(total_noise, alpha=float(self.last_lr * lr_scalar))
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
    
    def get_representative(self) -> Genome:
        raise NotImplementedError("MuonOpt does not support getting a representative genome.")
    
class AdamOpt(MomentumOpt):
    def __init__(self, total_steps: int, learning_rate: float, perturb_scale: float, warmup_steps: int = 0, scheduler: str = "none", momentum: float = 0.6, beta2: float = 0.85, epsilon: float = 1e-5, accumulate_fp32=True, cutoff_steps = 30, norm_by_mean : bool = True, norm_by_stddev : bool = True, force_lora_alternating: bool = False):
        super().__init__(total_steps, learning_rate, perturb_scale, warmup_steps, scheduler, momentum=momentum, cutoff_steps=cutoff_steps, norm_by_mean=norm_by_mean, norm_by_stddev=norm_by_stddev, force_lora_alternating=force_lora_alternating, optimizer_name=f"AdamOptimizer (beta2={beta2}, epsilon={epsilon}, accumulate_fp32={accumulate_fp32})")
        self.beta2 = beta2
        self.epsilon = epsilon
        self.force_disable_lr = True
        self.accumulate_fp32 = accumulate_fp32

    def step_update(self, tensor: torch.Tensor, random_offset: int, parameter_id, lr_scalar: float = 1, state: Dict = None):
        gen = torch.Generator(device=tensor.device)
        step = max(1, self.last_step)
        effective_step = step / 2 if self.force_lora_alternating else step
        correction = 1 - self.beta2 ** (effective_step / 2)
        correction = max(correction, 1e-6)

        effective_beta2 = (1 - self.beta2) / correction
        noise = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        accumulator = torch.zeros(tensor.shape, dtype=torch.float32 if self.accumulate_fp32 else tensor.dtype, device=tensor.device)
        second_moment = torch.zeros(tensor.shape, dtype=torch.float32 if self.accumulate_fp32 else tensor.dtype, device=tensor.device)
        current_head_idx = len(self.velocity_seeds_steps) - 1
        for step_idx in reversed(range(len(self.velocity_seeds_steps))):
            if step_idx % 2 != current_head_idx % 2 and self.force_lora_alternating:
                continue
            accumulator.zero_()
            for seed, weight in self.velocity_seeds_steps[step_idx]:
                gen.manual_seed(int(seed) + random_offset)
                torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
                accumulator.add_(noise.to(torch.float32 if self.accumulate_fp32 else tensor.dtype), alpha=float(weight))
            accumulator.pow_(2)
            second_moment.add_(accumulator, alpha=effective_beta2)
            effective_beta2 *= self.beta2
        second_moment.sqrt_().add_(self.epsilon)
        second_moment.reciprocal_()
        for seed, weight in zip(self.rep_genome.seeds, self.rep_genome.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise)
            update_step = noise.to(torch.float32 if self.accumulate_fp32 else tensor.dtype)
            update_step.mul_(second_moment)
            update_step.mul_(float(weight * lr_scalar * self.last_lr))
            tensor.add_(update_step.to(tensor.dtype))
            del update_step
        del noise
        del accumulator
        del second_moment

    def get_representative(self) -> Genome:
        raise NotImplementedError("AdamOpt does not support getting a representative genome.")