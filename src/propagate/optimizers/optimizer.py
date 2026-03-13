import copy
from typing import Dict, List

import torch

from propagate.genome import Genome
from propagate.optimizers.chain import OptimizerChain, Sub_Perturb_Buffer, Add_Perturb_Buffer
import math

from propagate.training_config import TrainingConfig

class Optimizer():
    """
    The optimizer is responsibly for orchestrating the model update. It contains the hyperparameters, and the manner in which perturbations/gradients are applied to the model. 

    It works by internally updating and calculating a "representative genome" (update_self) which stores a list of seeds which are reweighted and represent a gradient. This is done right after genines are evaluated, using the new reward data.

    It stores three optimizer paths: one calculates the perturbation (perturb_chain), one reverses the perturbation to get the original model (inverted_perturb_chain), and one applies the gradient update (update_chain).

    Note that vLLM workers (ray actors) cannot update the global optimizer state, and as such, the state is passed to and updated by the vLLM worker through the state dictionary.

    Basic behavior for saving the model and restoring it from a checkpoint is also provided (get_update_history and restore_from_history). However, be warned that the optimizer state will not be restored.
    
    Warning: The optimizer supports both stateful genomes (which maintain their historic seeds), and stateless genomes (where the updates are baked into the backend). However, the stateful genomes do not work with weight updates, meaning you should never call apply_grad. You MUST setup your pipeline so that the training works solely through apply_perturb with cached seeds.

    Attributes
    ----------
    optimizer_name : str
        The name of the optimizer.
    config : TrainingConfig
        The config for training.
    perturb_chain : List[OptimizerChain]
        The optimizer path for perturbation.
    update_chain : List[OptimizerChain]
        The optimizer path for updating the model.
    inverted_perturb_chain : List[OptimizerChain]
        The optimizer path for reversing the perturbation. Automatically generated if not provided.
    """
    def __init__(self, optimizer_name, config: TrainingConfig, perturb_chain: List[OptimizerChain], update_chain: List[OptimizerChain], inverted_perturb_chain: List[OptimizerChain] = None):
        self.optimizer_name = optimizer_name
        self.config = config
        self.perturb_chain = perturb_chain
        self.update_chain = update_chain
        if inverted_perturb_chain is None:
            subtractor = Sub_Perturb_Buffer()
            self.inverted_perturb_chain = copy.deepcopy(self.perturb_chain)
            found = False
            for i in range(len(self.inverted_perturb_chain) - 1, -1, -1):
                if isinstance(self.inverted_perturb_chain[i], Add_Perturb_Buffer):
                    found = True
                    self.inverted_perturb_chain[i] = subtractor
                    break
            assert found, "Did not find an add operation to replace with a sub operation for the inverted perturbation chain."
        else:
            self.inverted_perturb_chain = inverted_perturb_chain
        
        self.rep_genome = Genome()
        
        self.last_lr = config.learning_rate
        self.last_step = 1
        self.last_rstd = 1
    
    def get_lr(self, current_step: int) -> float:
        """
        Returns the learning rate based on the current step, applying the warmup and scheduler.

        Args:
            current_step (int): The current step of the optimizer.
        Returns:
            float: The learning rate.
        """
        if self.config.warmup_steps > 0 and current_step < self.config.warmup_steps:
            return self.config.learning_rate * (current_step / self.config.warmup_steps)

        t = max(0.0, min(1.0, (current_step - self.config.warmup_steps) / (self.config.total_steps - self.config.warmup_steps)))

        sched = (self.config.lr_scheduler or "none").lower()
        if sched in ("none", "constant"):
            return self.config.learning_rate
        if sched.startswith("linear"):
            return self.config.learning_rate * (1.0 - t)
        if sched.startswith("cosine"):
            return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * t))
        if sched.startswith("exponential"):
            return self.config.learning_rate * math.exp(-2 * math.sqrt(t))
        raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler}")
    
    def get_representative(self) -> Genome:
        """Returns a copy of the representative genome."""
        return self.rep_genome
    
    def update_self(self, genomes: List[Genome], current_step: int, true_reward_mean: float = None):
        """
        Update the optimizer's internal state based on the provided genomes.

        Based on the given rewards and hyperparameter config (norm_by_mean, rank_norm_rewards), calculates the gradient as a linear combination of the seeds of the given genomes. This gradient is packed into a new representative genome and saved.

        In the event there are historical seeds, seeds that are part of a previous gradient step are copied over with duplicates averaged.
        
        Args:
            genomes (List[Genome]): The genomes to update the optimizer with. These genomes MUST have their rewards calculated, or the update will become corrupted.
            current_step (int): The current step of the optimizer.
        """
        self.rep_genome = Genome()
        self.last_step = current_step
        self.last_lr = self.get_lr(current_step)
        
        # Calculate reward statistics for normalization (if enabled)
        rewards = [g.historical_rewards[-1] for g in genomes]
        reward_mean = sum(rewards) / len(genomes)
        true_mean = true_reward_mean if true_reward_mean is not None else reward_mean
        self.last_rstd = (sum([(r - reward_mean) ** 2 for r in rewards]) / len(genomes)) ** 0.5

        # Build representative genome from a combination of old seeds (history) and new seeds (gradients)
        new_seeds = {}
        old_seeds = {}
        old_seeds_count = {}
        
        # Create centered ranks for genomes
        n_genomes = len(genomes)
        if self.config.rank_norm_rewards:
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
            if self.config.rank_norm_rewards:
                grad_scale = centered_ranks[c]
            else:
                grad_scale = g.historical_rewards[-1] - (0 if self.config.mirror else true_mean)
            
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
    
    def apply_perturb(self, invert, genome: Genome, tensor: torch.Tensor, random_offset: int, parameter_id, state: Dict, lr_scalar: float = 1, do_log: bool = False):
        """
        Apply the perturbation to the given genome. This must be called INTERNALLY through vLLM, and should never be used outside of a vLLM worker.

        Args:
            genome (Genome): The genome to apply the perturbation with.
            tensor (torch.Tensor): The tensor to apply the perturbation to.
            random_offset (int): The random offset to apply to the perturbation. This is used to keep perturbations on different tensors I.I.D. Make sure the same tensor gets the same random offset.
            parameter_id (int): The parameter ID, to track parameter-specific states like RMSProp buffers.
            invert (bool): Whether to invert the perturbation. If true, applies the inverted perturbation chain.
            state (Dict): The state which stores the optimizer state.
            lr_scalar (float, optional): A generic scalar. Defaults to 1. May be used to scale the update from the backend.
            do_log (bool, optional): Used to force only rank 0 (gpu 0) to log, for very special logs such as grad norms.
        """
        # Setup state variables from optimizer global state
        self.set_state(state, lr_scalar)
        
        if invert:
            # Restore from perturbation
            for p in self.inverted_perturb_chain:
                p.apply(genome, state, parameter_id, tensor, random_offset, do_log)
        else:
            # Apply perturbation
            for p in self.perturb_chain:
                p.apply(genome, state, parameter_id, tensor, random_offset, do_log)
    
    def apply_grad(self, tensor: torch.Tensor, random_offset: int, parameter_id, state: Dict, lr_scalar: float = 1, do_log: bool = False):
        """
        Apply the gradient to the given tensor. This must be called INTERNALLY through vLLM, and should never be used outside of a vLLM worker.

        Args:
            tensor (torch.Tensor): The tensor to apply the gradient to.
            random_offset (int): The random offset to apply to the gradient.
            parameter_id (int): The parameter ID to apply the gradient to.
            lr_scalar (float, optional): The learning rate scalar. Defaults to 1.
            state (Dict, optional): The state to apply the gradient to. Defaults to None.
            do_log (bool, optional): Whether to log the gradient. Defaults to False.
        """
        # Setup state variables from optimizer global state
        self.set_state(state, lr_scalar)
        
        # Apply gradient update
        for p in self.update_chain:
            p.apply(self.rep_genome, state, parameter_id, tensor, random_offset, do_log)
            
    def set_state(self, state: Dict, lr_scalar: float):
        state["step"] = self.last_step
        state["lr"] = self.last_lr
        state["std"] = self.config.perturb_scale
        state["rstd"] = self.last_rstd
        state["population_size"] = self.config.population_size
        state["lr_scalar"] = lr_scalar