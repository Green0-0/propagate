from abc import ABC, abstractmethod
import copy
from typing import Any, List, Dict, Tuple

import torch

from propagate.genome import Genome

import math

class Optimizer(ABC):
    """Optimizers are responsible for maintaining the learning rate, perturb scale, and other parameters necessary for updating the model. They also contain the behavior to calculate the gradient using stored genome statistics (update_self) and apply it from the backend (step_update). Some optimizers may function without appying the gradient to the backend, simply by maintaining a representative genome (get_representative). Basic behavior for saving the model and restoring it from a checkpoint is also provided (get_update_history and restore_from_history).

    Attributes
    ----------
    optimizer_name : str
        The name of the optimizer.
    total_steps : int
        The total number of steps to train for.
    learning_rate : float
        The learning rate of the optimizer.
    perturb_scale : float
        The scale of the perturbation (sigma).
    warmup_steps : int
        The number of warmup steps.
    scheduler : str
        The learning rate scheduler to use.
    norm_by_mean : bool
        Whether to normalize the gradient by the mean reward. This is required for non-mirrored training.
    norm_by_stddev : bool
        Whether to normalize the gradient by the standard deviation of the reward.
    force_lora_alternating : bool
        Whether to force the use of alternating LoRA updates. This is required if you are training LoRA on an alternating schedule (train b then a, and repeat) with momentum-based optimizers.
    """
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
        """Performs a weight update on the provided tensor. Accepts a parameter ID to identify the parameter being updated. An additional lr_scalar can be provided to scale the learning rate from the backend directly. 
        
        The update cannot modify its internal state due to backend behavior, so the state dict should be updated and used instead.

        Reference random offset when calculating the random seed, so that parameter perturbations are IID.
        """
        pass

    @abstractmethod
    def get_representative(self) -> Genome:
        """Returns a representative genome for the current step. Some optimizers are unable to store a representative genome and will raise an exception."""
        pass

    @abstractmethod
    def get_update_history(self) -> Any:
        """Returns a list of lists of genomes representing the history of updates. This is used for saving a checkpoint as a list of seeds and the optimizer state (if any). It is very storage efficient; a set of seeds is just a list of integers and does not require the model."""
        pass

    @abstractmethod
    def restore_from_history(self, history, backend):
        """Restores the optimizer's state from the provided history of updates. This is used for loading a checkpoint."""
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
    """This the optimizer implements the standard ES update rule by approximating the gradient with the rewards, applying mean centering and standard deviation normalizing if set. Mirroring is supported through the sign of the perturb scale, without explicit modification to the update loop."""
    def __init__(self, total_steps: int, learning_rate: float, perturb_scale: float, warmup_steps: int = 0, scheduler: str = "none", norm_by_mean : bool = True, norm_by_stddev : bool = True, force_lora_alternating: bool = False):
        super().__init__("SimpleOptimizer", total_steps, learning_rate, perturb_scale, warmup_steps, scheduler, norm_by_mean=norm_by_mean, norm_by_stddev=norm_by_stddev, force_lora_alternating=force_lora_alternating)

    def update_self(self, genomes: List[Genome], current_step: int):
        """Updates the optimizer's internal state based on the provided genomes (assumed to have rewards calculated) and current step. Begins by calculating the population mean and standard deviation from the rewards, which is used to create the update scale with the lr. Then, builds a representative genome (which represents the new state of the model) by iterating through the current genomes seeds. If a seed is part of a previous gradient step, it is copied over (duplicates should be averaged), otherwise it is considered a direction of the gradient and scaled by the reward."""
        self.rep_genome = Genome()
        lr = self.get_lr(current_step)
        
        # Calculate reward statistics for normalization (if enabled)
        rewards = [g.historical_rewards[-1] for g in genomes]
        reward_mean = sum(rewards) / len(genomes)
        reward_stddev = (sum([(r - reward_mean) ** 2 for r in rewards]) / len(genomes)) ** 0.5

        # Calculate general gradient scaling
        update_scale = lr * (1/len(genomes))
        if self.norm_by_stddev:
            update_scale /= (reward_stddev + 1e-8)

        # Build representative genome from a combination of old seeds (history) and new seeds (gradients)
        new_seeds = {}
        old_seeds = {}
        old_seeds_count = {}
        
        for g in genomes:
            assert len(g.latest_inputs) > 0, "Genomes has no prompts for gradient calculation! Did you forget to generate data and evaluate first?"
            assert len(g.latest_outputs) > 0, "Genomes has no outputs for gradient calculation! Did you forget to generate data and evaluate first?"
            assert len(g.latest_rewards) > 0, "Genomes has no rewards for gradient calculation! Did you forget to generate data and evaluate first?"

            # Calculate the weighting of the gradient for that genome (high reward means more influence)
            grad_scale = g.historical_rewards[-1] * update_scale
            if self.norm_by_mean:
                grad_scale -= reward_mean * update_scale
            
            for i, seed in enumerate(g.seeds):
                weight = g.perturb_scales[i]
                if i < g.starting_index:
                    # If the seed is part of a previous gradient step, it is copied over (duplicates should be averaged, so that weights don't increase from historical values)
                    old_seeds[seed] = old_seeds.get(seed, 0) + weight
                    old_seeds_count[seed] = old_seeds_count.get(seed, 0) + 1
                else:
                    # If the seed is part of the new gradient step, it is considered a direction of the gradient and scaled by the reward
                    update_value = math.copysign(1, weight) * grad_scale
                    new_seeds[seed] = new_seeds.get(seed, 0) + update_value

        # Average old seeds and add to new seeds
        for seed, count in old_seeds_count.items():
            old_seeds[seed] /= count
            new_seeds[seed] = new_seeds.get(seed, 0) + old_seeds[seed]

        # Use new seeds to build representative genome
        for seed, weight in new_seeds.items():
            self.rep_genome.seeds.append(seed)
            self.rep_genome.perturb_scales.append(weight)
            self.rep_genome.historical_rewards.append(float('-inf'))
        self.rep_genome.starting_index = len(self.rep_genome.seeds)
        self.update_history.append(copy.deepcopy(self.rep_genome))

    def step_update(self, tensor: torch.Tensor, random_offset: int, parameter_id, lr_scalar: float = 1, state: Dict = None):
        """Performs a standard weight update on the given tensor by applying the representative genome's seeds and perturb scales."""
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
        """Returns the representative genomes of each update step."""
        return self.update_history
    
    def restore_from_history(self, history, backend):
        """Restores the optimizer's state by tracing the updates from the representative genomes in the provided history."""
        self.update_history = copy.deepcopy(history)
        for step_genome in history:
            self.rep_genome = step_genome
            backend.update(self)
        self.rep_genome = Genome()

class MomentumOpt(Optimizer):
    """Initialize the Momentum Optimizer. This optimizer keeps track of a list of seeds over time, which are used to recalculate the momentum update without additional memory overhead and minimum compute overhead.

    Warning: If an alternating lora is used, force_lora_alternating should be set to True to ensure that the optimizer does not mix up the seeds.

    Attributes
    ----------
    momentum : float
        The momentum factor.
    cutoff_steps : int
        The number of steps to keep in the velocity history. This is important for memory usage and stability. The optimizer will only look back this many steps when calculating the momentum update. If you use a high momentum factor, you may need to increase this valaue.
    """
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
        """Update the optimizer's state based on the genomes as usual, but also update the momentum history.
        For the momentum update, we add the new seeds to the velocity history and recalculate the momentum coefficient while looping over the appropriate velocity seeds.
        The old seeds are combined with the seeds yielded from the momentum buffer to create the representative genome (all other behavior matches SimpleOpt).
        """
        self.rep_genome = Genome()
        self.last_lr = self.get_lr(current_step)
        self.last_step = current_step
        lr_used = self.last_lr if not self.force_disable_lr else 1.0
        
        rewards = [g.historical_rewards[-1] for g in genomes]
        reward_mean = sum(rewards) / len(genomes)
        reward_stddev = (sum([(r - reward_mean) ** 2 for r in rewards]) / len(genomes)) ** 0.5
        
        # Drop the lr scalar (we apply it later, or not at all if we are using a subclass optimizer which has its own lr)
        update_scale = (1/len(genomes))
        if self.norm_by_stddev:
            update_scale /= (reward_stddev + 1e-8)

        new_seeds = {}
        old_seeds = {}
        old_seeds_count = {}
        
        for g in genomes:
            assert len(g.latest_inputs) > 0, "Genomes has no prompts for gradient calculation! Did you forget to generate data and evaluate first?"
            assert len(g.latest_outputs) > 0, "Genomes has no outputs for gradient calculation! Did you forget to generate data and evaluate first?"
            assert len(g.latest_rewards) > 0, "Genomes has no rewards for gradient calculation! Did you forget to generate data and evaluate first?"
            
            grad_scale = g.historical_rewards[-1] * update_scale
            if self.norm_by_mean:
                grad_scale -= reward_mean * update_scale

            for i, seed in enumerate(g.seeds):
                weight = g.perturb_scales[i]

                if i < g.starting_index:
                    old_seeds[seed] = old_seeds.get(seed, 0) + weight
                    old_seeds_count[seed] = old_seeds_count.get(seed, 0) + 1
                else:
                    update_value = math.copysign(1, weight) * grad_scale
                    new_seeds[seed] = new_seeds.get(seed, 0) + update_value
        
        # Manage old seeds seperately, since we don't want to contaminate the velocity
        for seed, count in old_seeds_count.items():
            old_seeds[seed] /= count
        for seed, weight in old_seeds.items():
            self.rep_genome.seeds.append(seed)
            self.rep_genome.perturb_scales.append(weight)
            self.rep_genome.historical_rewards.append(float('-inf'))

        # Update velocity history with the new seeds
        self.velocity_seeds_steps.append([(seed, new_seeds[seed]) for seed in new_seeds.keys()])
        if len(self.velocity_seeds_steps) > self.cutoff_steps:
            self.velocity_seeds_steps.pop(0)

        # Traverse velocity seeds in reverse. Note that we must consider the alternating lora and avoid cross-contamination of unrelated gradients.
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
        self.update_history.append({
            "genome": copy.deepcopy(self.rep_genome),
            "velocity_seeds_steps": copy.deepcopy(self.velocity_seeds_steps),
            "last_lr": self.last_lr
        })

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
    
    def get_update_history(self) -> Any:
        """Note: The update history is a list of dictionaries containing the genome, velocity seeds steps, and last lr."""
        return self.update_history
    
    def restore_from_history(self, history, backend):
        """Restore both the momentum history and load the model checkpoint."""
        self.update_history = copy.deepcopy(history)
        for item in history:
            self.rep_genome = item["genome"]
            self.velocity_seeds_steps = item["velocity_seeds_steps"]
            self.last_lr = item["last_lr"]
            backend.update(self)
        self.rep_genome = Genome()

class MuonOpt(MomentumOpt):
    """Muon is just momentum with a Newton-Schulz iteration to orthogonalize the gradients. 
    However, because of this it is significantly more memory intensive than momentum. 
    Muon is also not a linear combination of seeds so no representative can be computed.
    Note: LR is applied after the Newton-Schulz iteration during step_update, so it is not used in the update_self call.
    """
    def __init__(self, total_steps: int, learning_rate: float, perturb_scale: float, warmup_steps: int = 0, scheduler: str = "none", momentum: float = 0.6, cutoff_steps = 30, norm_by_mean : bool = True, norm_by_stddev : bool = True, force_lora_alternating: bool = False):
        
        super().__init__(total_steps, learning_rate, perturb_scale, warmup_steps, scheduler, momentum=momentum, cutoff_steps=cutoff_steps, norm_by_mean=norm_by_mean, norm_by_stddev=norm_by_stddev, force_lora_alternating=force_lora_alternating, optimizer_name="MuonOptimizer")
        self.force_disable_lr = True

    def step_update(self, tensor: torch.Tensor, random_offset: int, parameter_id, lr_scalar: float = 1, state: Dict = None):
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
    """This version of Adam is ONLY performant with LoRA. It works by recomputing the variance on the fly from the momentum history. This takes many operations and was not a good idea; the code here will be rewritten soon.

    Additionally, this optimizer does not support getting a representative genome, and it was found to be ineffective in practice (highly unstable, tends to diverge). It is recommended to avoid this optimizer.
    """
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