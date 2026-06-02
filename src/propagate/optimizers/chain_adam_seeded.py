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
    psampler : PSampler
        The sampler to use for generating the reconstructed noise.
    coeff_old : float
        The decay rate for old momentum (beta1).
    coeff_new : float
        The scaling factor for new momentum (1 - beta1).
    force_init_zeros : bool
        Whether to force initialization to zero.
    """
    def __init__(self, psampler, coeff_old: float = 0.95, coeff_new: float = 0.05, force_init_zeros: bool = True):
        self.psampler = psampler
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.force_init_zeros = force_init_zeros
        
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
        for step in reversed(state[(parameter_id, "seed_history")]):
            mock_genome = Genome().from_data({
                "seeds": [s for s, w in step],
                "perturb_scales": [float(running_factor * w) for s, w in step]
            })
            step_noise = self.psampler.sample(mock_genome, state, parameter_id, tensor, random_offset, do_log)
            momentum_buffer.add_(step_noise)
            running_factor *= self.coeff_old
        bias_correction = self.coeff_new if self.force_init_zeros else self.coeff_new/(1 - self.coeff_old ** len(state[(parameter_id, "seed_history")]))
        perturbation.add_(momentum_buffer, alpha=bias_correction)
        
        del momentum_buffer
        
class OC_Apply_RMSProp_Seeded(OptimizerChain):
    """
    Applies RMSProp directly to the buffer by reconstructing the second moment estimate from historical seeds.
    This saves memory but is extremely computationally expensive.
    
    As with all RMSProp implementations, you are given the option to either initialize with some value or not (-999). If you initialize with some value, bias correction is NOT applied, and RMSProp slowly drifts to its true value over time, serving as a warmup. If choose not to do so and set the init value to -999, bias correction is applied; this may cause erratic behavior as the perturbation suddenly gets rescaled.
    
    Attributes
    ----------
    psampler : PSampler
        The sampler to use for generating the reconstructed noise.
    coeff_old : float
        The decay rate for the running average.
    coeff_new : float
        The scaling factor for the new squared gradient.
    force_init_value : float
        The value to initialize the running average to, or -999.
    epsilon : float
        The epsilon value for stability.
    """
    def __init__(self, psampler, coeff_old: float = 0.95, coeff_new: float = 0.05, force_init_value: float = 1e-4, epsilon: float = 1e-5):
        self.psampler = psampler
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.force_init_value = force_init_value
        self.epsilon = epsilon
        
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
        for step in reversed(state[(parameter_id, "seed_history")]):
            mock_genome = Genome().from_data({
                "seeds": [s for s, w in step],
                "perturb_scales": [w for s, w in step]
            })
            step_noise = self.psampler.sample(mock_genome, state, parameter_id, tensor, random_offset, do_log)
            rmsprop_buffer.addcmul_(step_noise, step_noise, value=float(running_factor * self.coeff_new))
            running_factor *= self.coeff_old
        rmsprop_buffer.add_(init_val, alpha=running_factor)
        bias_correction = 1 if self.force_init_value != -999 else 1 - self.coeff_old ** len(state[(parameter_id, "seed_history")])
        rmsprop_buffer.div_(bias_correction).sqrt_().add_(self.epsilon)
        perturbation.div_(rmsprop_buffer)
        
        del rmsprop_buffer