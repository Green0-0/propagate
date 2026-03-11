from propagate.optimizers.chain import OptimizerChain
from propagate.genome import Genome

from typing import Dict

import torch

class OC_Compute_RMSProp_Blockwise(OptimizerChain):
    """
    Computes the RMSProp scaling factor in a blockwise manner, using a single scaling factor for each tensor.
    This saves memory and greatly reduces variance, at the cost of granularity.
    
    Note that you must specify ``div_by_rmsprop_block`` in ``Scale_Perturbations``, otherwise the blockwise factor won't be applied.
    
    As with all RMSProp implementations, you are given the option to either initialize with some value or not (-999). If you initialize with some value, bias correction is NOT applied, and RMSProp slowly drifts to its true value over time, serving as a warmup. If choose not to do so and set the init value to -999, bias correction is applied; this may cause erratic behavior as the perturbation suddenly gets rescaled.
    
    Attributes
    ----------
    coeff_old : float
        The coefficient for the running average.
    coeff_new : float
        The coefficient for the new gradient.
    force_init_value : float
        The value to initialize the running average to, or -999.
    """
    def __init__(self, coeff_old: float = 0.95, coeff_new: float = 0.05, force_init_value: float = 1e-4):
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.force_init_value = force_init_value
        if force_init_value != -999: 
            self.decay_coeff = 0
        else:
            self.decay_coeff = coeff_old

    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to compute RMSProp.")
        if (parameter_id, "rmsprop_block") not in state:
            if self.force_init_value != -999:
                state[(parameter_id, "rmsprop_block")] = self.force_init_value
            else:
                state[(parameter_id, "rmsprop_block")] = 0
        if (parameter_id, "rmsprop_block_decay_coeff") not in state:
            state[(parameter_id, "rmsprop_block_decay_coeff")] = self.decay_coeff
        elif self.force_init_value == -999:
            state[(parameter_id, "rmsprop_block_decay_coeff")] *= self.decay_coeff

        perturbation = state["perturb_buffer"]
        grads_squared_new = (torch.linalg.vector_norm(perturbation, dtype=torch.float32) ** 2 / perturbation.numel()).item()
        state[(parameter_id, "rmsprop_block")] = state[(parameter_id, "rmsprop_block")] * self.coeff_old + grads_squared_new * self.coeff_new

class OC_Compute_RMSProp(OptimizerChain):
    """Computes the RMSProp scaling factor element-wise. This will only compute and update the scaling factors, it will not apply them.
    
    As with all RMSProp implementations, you are given the option to either initialize with some value or not (-999). If you initialize with some value, bias correction is NOT applied, and RMSProp slowly drifts to its true value over time, serving as a warmup. If choose not to do so and set the init value to -999, bias correction is applied; this may cause erratic behavior as the perturbation suddenly gets rescaled.
    
    Attributes
    ----------
    coeff_old : float
        The coefficient for the running average.
    coeff_new : float
        The coefficient for the new gradient.
    force_init_value : float
        The value to initialize the running average to, or -999.
    """
    def __init__(self, coeff_old: float = 0.95, coeff_new: float = 0.05, force_init_value: float = 1e-4):
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.force_init_value = force_init_value
        if force_init_value != -999: 
            self.decay_coeff = 0
        else:
            self.decay_coeff = coeff_old

    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to compute RMSProp.")
        if (parameter_id, "rmsprop_decay_coeff") not in state:
            state[(parameter_id, "rmsprop_decay_coeff")] = self.decay_coeff
        elif self.force_init_value == -999:
            state[(parameter_id, "rmsprop_decay_coeff")] *= self.decay_coeff
        perturbation = state["perturb_buffer"]
        if (parameter_id, "rmsprop") not in state:
            init_val = self.force_init_value if self.force_init_value != -999 else 0
            state[(parameter_id, "rmsprop")] = perturbation.pow(2).mul_(self.coeff_new).add_(init_val * self.coeff_old)
        else:
            state[(parameter_id, "rmsprop")].mul_(self.coeff_old)
            state[(parameter_id, "rmsprop")].addcmul_(perturbation, perturbation, value=self.coeff_new)

class OC_Apply_RMSProp(OptimizerChain):
    """Applies the RMSProp scaling to the perturbation buffer. If the RMSProp buffer is not present, this will be skipped (such as if you want to scale the perturbation on the very first step), which is equivalent to having RMSProp = 1.
    
    Attributes
    ----------
    epsilon : float
        The epsilon value for stability.
    """
    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to apply RMSProp to.")
        if (parameter_id, "rmsprop") not in state:
            print("RMSProp not initialized, skipping...")
            return
        perturbation = state["perturb_buffer"]
        denom = state[(parameter_id, "rmsprop")].div(1 - state[(parameter_id, "rmsprop_decay_coeff")]).sqrt_().add_(self.epsilon)
        perturbation.div_(denom)
        del denom

class OC_Compute_Momentum(OptimizerChain):
    """Computes the momentum vector.
    
    As with all momentum implementations, you are given the option to either initialize with 0s or bias correction. If you initialize with 0s, bias correction is NOT applied, and momentum slowly drifts to its true value over time, as in polyak momentum. If you initialize without 0s, bias correction is applied so the gradient gets upscaled.
    
    Attributes
    ----------
    coeff_old : float
        The coefficient for the running average (beta1).
    coeff_new : float
        The coefficient for the new gradient (1 - beta1).
    force_init_zeros : bool
        Whether to force initialization of momentum to zero.
    """
    def __init__(self, coeff_old: float = 0.5, coeff_new: float = 0.5, force_init_zeros: bool = False):
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.force_init_zeros = force_init_zeros
        if force_init_zeros: 
            # Note: This will make momentum start very slowly, so it is better used with polyak, ie. coeff_new = 1 and coeff_old = 0.8
            self.decay_coeff = 0
        else:
            self.decay_coeff = coeff_old

    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to compute momentum.")
        if (parameter_id, "momentum_decay_coeff") not in state:
            state[(parameter_id, "momentum_decay_coeff")] = self.decay_coeff
        elif not self.force_init_zeros:
            state[(parameter_id, "momentum_decay_coeff")] *= self.decay_coeff
        perturbation = state["perturb_buffer"]
        if (parameter_id, "momentum") not in state:
            state[(parameter_id, "momentum")] = perturbation.mul(self.coeff_new)
        else:
            state[(parameter_id, "momentum")].mul_(self.coeff_old)
            state[(parameter_id, "momentum")].add_(perturbation, alpha=self.coeff_new)


class OC_Add_Momentum(OptimizerChain):
    """Adds the momentum to the perturbation buffer. If the momentum buffer is not present, this will be skipped (such as if you want to add to the perturbation on the very first step), which is equivalent to having momentum = 0.
    
    Warning: Momentum is ADDED to the buffer and does not replace it. Make sure to zero the buffer if you want only the momentum in it.
    """
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to compute momentum.")
        if (parameter_id, "momentum") not in state:
            print("Momentum not initialized, skipping...")
            return
        perturbation = state["perturb_buffer"]
        perturbation.add_(state[(parameter_id, "momentum")], alpha=1/(1 - state[(parameter_id, "momentum_decay_coeff")]))