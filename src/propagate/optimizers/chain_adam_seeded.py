from propagate.optimizers.chain import OptimizerChain
from propagate.genome import Genome

from typing import Dict

import torch

class OC_Update_Seed_History(OptimizerChain):
    def __init__(self, max_steps = 50):
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
    """Warning: Momentum is ADDED to the buffer and does not replace it. Make sure to zero the buffer if you want only the momentum in it."""
    def __init__(self, coeff_old = 0.95, coeff_new = 0.05, force_init_zeros = True, bernoulli_center = -999):
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
    def __init__(self, coeff_old = 0.95, coeff_new = 0.05, force_init_ones: bool = True, epsilon: float = 1e-5, bernoulli_center: float = -999):
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.force_init_ones = force_init_ones
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
        
        init_val = 1 if self.force_init_ones else 0
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
        bias_correction = 1 if self.force_init_ones else 1 - self.coeff_old ** len(state[(parameter_id, "seed_history")])
        rmsprop_buffer.div_(bias_correction).sqrt_().add_(self.epsilon)
        perturbation.div_(rmsprop_buffer)
        
        del rmsprop_buffer, square_buffer, buffer