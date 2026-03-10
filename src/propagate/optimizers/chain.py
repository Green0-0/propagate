        
from abc import ABC, abstractmethod
from typing import Dict

import torch

from propagate.genome import Genome

import math

class OptimizerChain(ABC):
    """The optimizer chain implements a single operation within a weight update step. This may include creating the perturbation, scaling it, or applying optimizer logic (eg. ADAM). """
    @abstractmethod
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        """Modifies the state dict and/or tensor according to the optimizer's internal logic."""
        pass

class Sign_Perturb_Buffer(OptimizerChain):
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to sign.")
        perturbation = state["perturb_buffer"]
        perturbation.sign_()

class Abs_Perturb_Buffer(OptimizerChain):
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to sign.")
        perturbation = state["perturb_buffer"]
        perturbation.abs_()

class MaxSubExp_Perturb_Buffer(OptimizerChain):
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to sign.")
        perturbation = state["perturb_buffer"]
        perturbation.sub_(torch.max(perturbation))
        perturbation.exp_()
        
class Zero_Perturb_Buffer(OptimizerChain):
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to zero.")
        perturbation = state["perturb_buffer"]
        perturbation.zero_()

class Add_Perturb_Buffer(OptimizerChain):
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to commit.")
        perturbation = state["perturb_buffer"]
        tensor.add_(perturbation.to(tensor.dtype))

class Sub_Perturb_Buffer(OptimizerChain):
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to commit.")
        perturbation = state["perturb_buffer"]
        tensor.sub_(perturbation.to(tensor.dtype))

class Delete_Perturb_Buffer(OptimizerChain):
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to delete.")
        del state["perturb_buffer"]
        
class Copy_Weights_To_Perturb_Buffer(OptimizerChain):
    def __init__(self, cast_type = torch.bfloat16) -> None:
        self.cast_type = cast_type
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" in state:
            raise ValueError("Perturbation buffer was requested to be copied, but already exists! Did you forget to delete it?")
        state["perturb_buffer"] = tensor.to(dtype=self.cast_type, copy=True)
        
class Override_Weights_With_Perturb_Buffer(OptimizerChain):
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("Perturbation buffer does not exist to override?")
        tensor.copy_(state["perturb_buffer"])
        
class Init_Perturbation_Gaussian(OptimizerChain):
    def __init__(self, fp32_accumulate = True):
        self.fp32_accumulate = fp32_accumulate
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" in state:
            raise ValueError("Perturbation buffer was requested to be generated, but already exists! Did you forget to delete it?")
        gen = torch.Generator(device=tensor.device)
        perturbation = torch.zeros_like(tensor, dtype = torch.float32 if self.fp32_accumulate else tensor.dtype)
        buffer = torch.empty_like(tensor)
        for seed, weight in zip(source.seeds, source.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            buffer.normal_(generator=gen)
            perturbation.add_(buffer, alpha=float(weight))
        state["perturb_buffer"] = perturbation
        del buffer

class Init_Perturbation_Bernoulli(OptimizerChain):
    def __init__(self, center=0.5, fp32_accumulate = True):
        self.center = center
        self.fp32_accumulate = fp32_accumulate        
    
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" in state:
            raise ValueError("Perturbation buffer was requested to be generated, but already exists! Did you forget to delete it?")
        gen = torch.Generator(device=tensor.device)
        perturbation = torch.zeros_like(tensor, dtype = torch.float32 if self.fp32_accumulate else tensor.dtype)
        buffer = torch.empty_like(tensor)
        for seed, weight in zip(source.seeds, source.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            buffer.random_(0, 2, generator=gen).sub_(self.center).mul_(2)
            perturbation.add_(buffer, alpha=float(weight))
        state["perturb_buffer"] = perturbation
        del buffer

class Scale_Perturbation(OptimizerChain):
    def __init__(self, div_by_pop=True, div_by_rstd=False, mul_by_std=True, mul_by_lr=True, mul_by_lr_scalar=True, div_by_std=False, div_by_rmsprop_block=False, epsilon=1e-5):
        self.div_by_pop = div_by_pop
        self.div_by_rstd = div_by_rstd
        self.mul_by_std = mul_by_std
        self.mul_by_lr = mul_by_lr
        self.mul_by_lr_scalar = mul_by_lr_scalar
        self.div_by_std = div_by_std
        self.div_by_rmsprop_block = div_by_rmsprop_block
        self.epsilon = epsilon

    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "population_size" not in state or "rstd" not in state or "std" not in state or "lr" not in state or "lr_scalar" not in state:
            raise ValueError(f"State dict is missing required keys, found: {list(state.keys())}")
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to scale.")
        total_scale = 1
        if self.div_by_pop: total_scale *= 1 / state["population_size"]
        if self.mul_by_std: total_scale *= state["std"]
        if self.div_by_std: total_scale *= 1 / state["std"]
        if self.mul_by_lr: total_scale *= state["lr"]
        if self.mul_by_lr_scalar: total_scale *= state["lr_scalar"]

        if self.div_by_rstd: total_scale *= 1 / (state["rstd"] + self.epsilon)
        if self.div_by_rmsprop_block and (parameter_id, "rmsprop_block") in state: total_scale *= 1 / (math.sqrt(state[(parameter_id, "rmsprop_block")]/(1 - state[(parameter_id, "rmsprop_block_decay_coeff")])) + self.epsilon)
        perturbation = state["perturb_buffer"]
        perturbation.mul_(total_scale)