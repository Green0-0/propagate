from propagate.optimizers.chain import OptimizerChain
from propagate.genome import Genome

from typing import Dict

import torch

class OC_Compute_RMSProp_Blockwise(OptimizerChain):
    def __init__(self, coeff_old = 0.95, coeff_new = 0.05, force_init_ones = True):
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.force_init_ones = force_init_ones
        if force_init_ones: 
            self.decay_coeff = 0
        else:
            self.decay_coeff = coeff_old

    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to compute RMSProp.")
        if (parameter_id, "rmsprop_block") not in state:
            if self.force_init_ones:
                state[(parameter_id, "rmsprop_block")] = 1
            else:
                state[(parameter_id, "rmsprop_block")] = 0
        if (parameter_id, "rmsprop_block_decay_coeff") not in state:
            state[(parameter_id, "rmsprop_block_decay_coeff")] = self.decay_coeff
        elif not self.force_init_ones:
            state[(parameter_id, "rmsprop_block_decay_coeff")] *= self.decay_coeff

        perturbation = state["perturb_buffer"]
        grads_squared_new = (torch.linalg.vector_norm(perturbation, dtype=torch.float32) ** 2 / perturbation.numel()).item()
        state[(parameter_id, "rmsprop_block")] = state[(parameter_id, "rmsprop_block")] * self.coeff_old + grads_squared_new * self.coeff_new

class OC_Compute_RMSProp(OptimizerChain):
    def __init__(self, coeff_old = 0.95, coeff_new = 0.05, force_init_ones = True):
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.force_init_ones = force_init_ones
        if force_init_ones: 
            self.decay_coeff = 0
        else:
            self.decay_coeff = coeff_old

    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to compute RMSProp.")
        if (parameter_id, "rmsprop_decay_coeff") not in state:
            state[(parameter_id, "rmsprop_decay_coeff")] = self.decay_coeff
        elif not self.force_init_ones:
            state[(parameter_id, "rmsprop_decay_coeff")] *= self.decay_coeff
        perturbation = state["perturb_buffer"]
        if (parameter_id, "rmsprop") not in state:
            init_val = 1 if self.force_init_ones else 0
            state[(parameter_id, "rmsprop")] = perturbation.pow(2).mul_(self.coeff_new).add_(init_val * self.coeff_old)
        else:
            state[(parameter_id, "rmsprop")].mul_(self.coeff_old)
            state[(parameter_id, "rmsprop")].addcmul_(perturbation, perturbation, value=self.coeff_new)

class OC_Apply_RMSProp(OptimizerChain):
    def __init__(self, epsilon = 1e-5):
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
    def __init__(self, coeff_old = 0.5, coeff_new = 0.5, force_init_zeros = False):
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
    """Warning: Momentum is ADDED to the buffer and does not replace it. Make sure to zero the buffer if you want only the momentum in it."""
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to compute momentum.")
        if (parameter_id, "momentum") not in state:
            print("Momentum not initialized, skipping...")
            return
        perturbation = state["perturb_buffer"]
        perturbation.add_(state[(parameter_id, "momentum")], alpha=1/(1 - state[(parameter_id, "momentum_decay_coeff")]))