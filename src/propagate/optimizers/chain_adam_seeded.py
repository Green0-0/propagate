from propagate.optimizers.chain import OptimizerChain
from propagate.genome import Genome

from typing import Dict

import torch

class OC_Update_Seed_History(OptimizerChain):
    """Updates the history of seeds used for gradient updates. This should be done with the representative genome, and only be called on the gradient update.
    
    Attributes
    ----------
    max_steps : int
        The maximum number of steps to keep in history.
    """
    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if (parameter_id, "seed_history") not in state:
            state[(parameter_id, "seed_history")] = []
        history = state[(parameter_id, "seed_history")]
        history.append(list(zip(source.seeds, source.perturb_scales)))
        while len(history) > self.max_steps:
            history.pop(0)
            
class OC_Apply_Momentum_Seeded(OptimizerChain):
    """
    Applies momentum directly to the buffer by reconstructing the gradient from historical seeds.
    This saves memory by not storing the momentum vector directly, but requires regenerating noise.
    
    As with all momentum implementations, you are given the option to either initialize with 0s or bias correction. If you initialize with 0s, bias correction is NOT applied, and momentum slowly drifts to its true value over time, as in polyak momentum. If you initialize without 0s, bias correction is applied so the gradient gets upscaled.
    
    Warning: Momentum is ADDED to the buffer and does not replace it. Make sure to zero the buffer if you want only the momentum in it.
    
    Attributes
    ----------
    coeff_old : float
        The decay rate for old momentum (beta1).
    coeff_new : float
        The scaling factor for new momentum (1 - beta1).
    force_init_zeros : bool
        Whether to force initialization to zero.
    bernoulli_center : float
        If set, uses Bernoulli noise centered at this value instead of Gaussian.
    """
    def __init__(self, coeff_old: float = 0.95, coeff_new: float = 0.05, force_init_zeros: bool = True, bernoulli_center: float = -999):
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.force_init_zeros = force_init_zeros
        self.bernoulli_center = bernoulli_center
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to compute momentum.")
        if (parameter_id, "seed_history") not in state or not state[(parameter_id, "seed_history")]:
            print("Seed history missing for seeded momentum, skipping...")
            return
        perturbation = state["perturb_buffer"]
        
        running_factor = 1
        momentum_buffer = torch.zeros_like(perturbation)
        buffer = torch.empty_like(tensor)
        gen = torch.Generator(device=tensor.device)
        for step in reversed(state[(parameter_id, "seed_history")]):
            for seed, weight in step:
                gen.manual_seed(int(seed) + random_offset)
                if self.bernoulli_center != -999:
                    buffer.random_(0, 2, generator=gen).sub_(self.bernoulli_center).mul_(2)
                else:
                    buffer.normal_(generator=gen)
                momentum_buffer.add_(buffer, alpha=float(running_factor * weight))
            running_factor *= self.coeff_old
        bias_correction = self.coeff_new if self.force_init_zeros else self.coeff_new/(1 - self.coeff_old ** len(state[(parameter_id, "seed_history")]))
        perturbation.add_(momentum_buffer, alpha=bias_correction)
        
        del momentum_buffer, buffer
        
class OC_Apply_RMSProp_Seeded(OptimizerChain):
    """
    Applies RMSProp directly to the buffer by reconstructing the second moment estimate from historical seeds.
    This saves memory but is extremely computationally expensive.
    
    As with all RMSProp implementations, you are given the option to either initialize with some value or not (-999). If you initialize with some value, bias correction is NOT applied, and RMSProp slowly drifts to its true value over time, serving as a warmup. If choose not to do so and set the init value to -999, bias correction is applied; this may cause erratic behavior as the perturbation suddenly gets rescaled.
    
    Attributes
    ----------
    coeff_old : float
        The decay rate for the running average.
    coeff_new : float
        The scaling factor for the new squared gradient.
    force_init_value : float
        The value to initialize the running average to, or -999.
    epsilon : float
        The epsilon value for stability.
    bernoulli_center : float
        If set, uses Bernoulli noise centered at this value instead of Gaussian.
    """
    def __init__(self, coeff_old: float = 0.95, coeff_new: float = 0.05, force_init_value: float = 1e-4, epsilon: float = 1e-5, bernoulli_center: float = -999):
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.force_init_value = force_init_value
        self.epsilon = epsilon
        self.bernoulli_center = bernoulli_center
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to compute RMSProp.")
        if (parameter_id, "seed_history") not in state or not state[(parameter_id, "seed_history")]:
            print("Seed history missing for seeded RMSProp, skipping...")
            return
        perturbation = state["perturb_buffer"]
        
        init_val = self.force_init_value if self.force_init_value != -999 else 0
        running_factor = 1
        rmsprop_buffer = torch.zeros_like(perturbation)
        square_buffer = torch.zeros_like(perturbation)
        buffer = torch.empty_like(tensor)
        gen = torch.Generator(device=tensor.device)
        for step in reversed(state[(parameter_id, "seed_history")]):
            for seed, weight in step:
                gen.manual_seed(int(seed) + random_offset)
                if self.bernoulli_center != -999:
                    buffer.random_(0, 2, generator=gen).sub_(self.bernoulli_center).mul_(2)
                else:
                    buffer.normal_(generator=gen)
                square_buffer.add_(buffer, alpha=float(weight))
            rmsprop_buffer.addcmul_(square_buffer, square_buffer, value=float(running_factor * self.coeff_new))
            running_factor *= self.coeff_old
            square_buffer.zero_()
        rmsprop_buffer.add_(init_val, alpha=running_factor)
        bias_correction = 1 if self.force_init_value != -999 else 1 - self.coeff_old ** len(state[(parameter_id, "seed_history")])
        rmsprop_buffer.div_(bias_correction).sqrt_().add_(self.epsilon)
        perturbation.div_(rmsprop_buffer)
        
        del rmsprop_buffer, square_buffer, buffer