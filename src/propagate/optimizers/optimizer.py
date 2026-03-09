import copy
from typing import Dict, List

import torch

from propagate.genome import Genome

import math

class Optimizer():
    """TODO: Update
    
    Optimizers are responsible for maintaining the learning rate, perturb scale, and other parameters necessary for updating the model. They also contain the behavior to calculate the gradient using stored genome statistics (update_self) and apply it from the backend (step_update). Some optimizers may function without appying the gradient to the backend, simply by maintaining a representative genome (get_representative). Basic behavior for saving the model and restoring it from a checkpoint is also provided (get_update_history and restore_from_history).
    
    Warning: The optimizer supports both stateful genomes (which maintain their historic seeds), and stateless genomes (where the updates are baked into the backend). However, the stateful genomes do not work with weight updates, meaning you should never call apply_grad.

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
    """
    def __init__(self, optimizer_name, total_steps: int, learning_rate: float, perturb_scale: float, population_size: int, perturb_chain: List[OptimizerChain], inverted_perturb_chain: List[OptimizerChain], update_chain: List[OptimizerChain], warmup_steps: int = 0, scheduler: str = "none", norm_by_mean: bool = True, rank_norm_rewards: bool = True):
        self.optimizer_name = optimizer_name
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.perturb_scale = perturb_scale
        self.population_size = population_size
        
        self.last_lr = learning_rate
        self.last_step = 1
        self.last_rstd = 1
        
        self.perturb_chain = perturb_chain
        self.inverted_perturb_chain = inverted_perturb_chain
        self.update_chain = update_chain
        
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.norm_by_mean = norm_by_mean
        self.rank_norm_rewards = rank_norm_rewards
        
        self.rep_genome = Genome()
        self.update_history = []
    
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
    
    def update_self(self, genomes: List[Genome], current_step: int):
        """Updates the optimizer's internal state based on the provided genomes (assumed to have rewards calculated) and current step. Begins by calculating the population mean and standard deviation from the rewards, which is used to create the update scale with the lr. Then, builds a representative genome (which represents the new state of the model) by iterating through the current genomes seeds. If a seed is part of a previous gradient step, it is copied over (duplicates should be averaged), otherwise it is considered a direction of the gradient and scaled by the reward."""
        self.rep_genome = Genome()
        self.last_step = current_step
        self.last_lr = self.get_lr(current_step)
        
        # Calculate reward statistics for normalization (if enabled)
        rewards = [g.historical_rewards[-1] for g in genomes]
        reward_mean = sum(rewards) / len(genomes)
        self.rstd = (sum([(r - reward_mean) ** 2 for r in rewards]) / len(genomes)) ** 0.5

        # Build representative genome from a combination of old seeds (history) and new seeds (gradients)
        new_seeds = {}
        old_seeds = {}
        old_seeds_count = {}
        
        # Create centered ranks for genomes
        n_genomes = len(genomes)
        if self.rank_norm_rewards:
            sorted_indices = sorted(range(n_genomes), key=lambda i: rewards[i])
            centered_ranks = [0.0] * n_genomes
            if n_genomes > 1:
                for rank, idx in enumerate(sorted_indices):
                    centered_ranks[idx] = (rank / (n_genomes - 1)) - 0.5
        
        for c, g in enumerate(genomes):
            assert len(g.latest_inputs) > 0, "Genomes has no prompts for gradient calculation! Did you forget to generate data and evaluate first?"
            assert len(g.latest_outputs) > 0, "Genomes has no outputs for gradient calculation! Did you forget to generate data and evaluate first?"
            assert len(g.latest_rewards) > 0, "Genomes has no rewards for gradient calculation! Did you forget to generate data and evaluate first?"

            # Calculate the weighting of the gradient for that genome (high reward means more influence)
            if self.rank_norm_rewards:
                grad_scale = centered_ranks[c]
            else:
                grad_scale = g.historical_rewards[-1] - (reward_mean if self.norm_by_mean else 0)
            
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

    def apply_perturb(self, genome: Genome, tensor: torch.Tensor, random_offset: int, parameter_id, invert, lr_scalar: float = 1, state: Dict = None, do_log: bool = False):
        state["step"] = self.last_step
        state["lr"] = self.last_lr
        state["std"] = self.perturb_scale
        state["rstd"] = self.last_rstd
        state["population_size"] = self.population_size
        state["lr_scalar"] = lr_scalar
        
        if invert:
            for p in self.inverted_perturb_chain:
                p.apply(genome, state, parameter_id, tensor, random_offset, do_log)
        else:
            for p in self.perturb_chain:
                p.apply(genome, state, parameter_id, tensor, random_offset, do_log)
    
    def apply_grad(self, tensor: torch.Tensor, random_offset: int, parameter_id, lr_scalar: float = 1, state: Dict = None, do_log: bool = False):
        state["step"] = self.last_step
        state["lr"] = self.last_lr
        state["std"] = self.perturb_scale
        state["rstd"] = self.last_rstd
        state["population_size"] = self.population_size
        state["lr_scalar"] = lr_scalar
        
        for p in self.update_chain:
            p.apply(self.rep_genome, state, parameter_id, tensor, random_offset, do_log)
            
    def get_representative(self) -> Genome:
        return self.rep_genome
    
    def get_update_history(self) -> List[Genome]:
        """Returns the representative genomes of each update step."""
        return self.update_history
    
    def restore_from_history(self, history, backend):
        """Restores the latest genome's state by tracing the updates from the representative genomes in the provided history. 
        
        WARNING: Will not restore the optimizer state."""
        self.update_history = copy.deepcopy(history)
        for step_genome in history:
            self.rep_genome = step_genome
            backend.update(self)
        self.rep_genome = Genome()