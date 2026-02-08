"""ES:
Sequential Optimizer Chaining:
Store grad_scale (N, std, reward-std) and step_scale (lr, lr-scalar)
Store dict of optimizer states (momentum, rmsprop, rmsprop-blockwise, momentum-seeded)
Reuse class for perturb/update 
Pre-calculate normal update grad without scaling
When computing update, begin by creating a zeros buffer which is updated with the gradients by iteration through the state update funcs
Create state update funcs:
apply_scales(use_grad_scale, use_step_scale, mul_std, div_std, hadamard_weights, apply_rmsprp_block) (merged for numerical stability, add std for natural ES)
build_perturb_buffer_gaussian
build_perturb_buffer_bernoulli
zero_buffer
commit_buffer

calculate_momentum
calculate_momentum_seeded
calculate_rmsprop
calculate_rmsprop_seeded
calculate_rmsprop_blockwise
(make sure momentum/rmsprop does not leak on alternating lora adapters)
add_momentum
divide_rmsprop
apply_signsgd
newton_schulz

Which can be reused to generate both the perturbations and the weight updates, and even do things like nesterov

Optimizer: Sparse gradient mirrored optimizer (prunes low-signal mirrored pairs), automatic std calculation using 1/5th or 20% rule of std (possibly also consider difference between center reward and mean reward to assess divergance), rank normalization of rewards to stabilize update direction

Lora weight decay to suppress noise

SparseSignSGD, probabilistic gradient calculation"""
from abc import ABC, abstractmethod
import copy
from typing import Any, List, Dict, Tuple

import torch

from propagate.genome import Genome

import math

class OptimizerChain(ABC):
    """The optimizer chain implements a single operation within a weight update step. This may include creating the perturbation, scaling it, or applying optimizer logic (eg. ADAM). """
    @abstractmethod
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int):
        """Modifies the state dict and/or tensor according to the optimizer's internal logic."""
        pass

class OC_Init_Perturbation_Gaussian(OptimizerChain):
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int):
        if (parameter_id, "perturbation") in state:
            del state[(parameter_id, "perturbation")]
        gen = torch.Generator(device=tensor.device)
        perturbation = torch.zeros_like(tensor)
        buffer = torch.empty_like(tensor)
        for seed, weight in zip(source.seeds, source.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=buffer)
            perturbation.add_(buffer, alpha=float(weight))
        del buffer

        state[(parameter_id, "perturbation")] = perturbation

class OC_Init_Perturbation_Bernoulli(OptimizerChain):
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int):
        if (parameter_id, "perturb_buffer") in state:
            del state[(parameter_id, "perturb_buffer")]
        gen = torch.Generator(device=tensor.device)
        perturbation = torch.zeros_like(tensor)
        buffer = torch.empty_like(tensor)
        for seed, weight in zip(source.seeds, source.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            torch.rand(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=buffer)
            buffer.sub_(0.5).mul_(2)
            perturbation.add_(buffer, alpha=float(weight))
        del buffer

        state[(parameter_id, "perturb_buffer")] = perturbation

class OC_Scale_Perturbation(OptimizerChain):
    def __init__(self, div_by_pop=True, div_by_rstd=False, mul_by_std=True, mul_by_lr=True, mul_by_lr_scalar=True, div_by_std=False, div_by_rmsprop_block=False, epsilon=1e-8):
        self.div_by_pop = div_by_pop
        self.div_by_rstd = div_by_rstd
        self.mul_by_std = mul_by_std
        self.mul_by_lr = mul_by_lr
        self.mul_by_lr_scalar = mul_by_lr_scalar
        self.div_by_std = div_by_std
        self.div_by_rmsprop_block = div_by_rmsprop_block
        self.epsilon = epsilon

    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int):
        if "population_size" not in state or "rstd" not in state or "std" not in state or "lr" not in state or (parameter_id, "lr_scalar") not in state:
            raise ValueError(f"State dict is missing required keys, found: {list(state.keys())}")
        if (parameter_id, "perturb_buffer") not in state:
            raise ValueError("State dict is missing the perturbation to scale.")
        total_scale = 1
        if self.div_by_pop: total_scale *= 1 / state["population_size"]
        if self.mul_by_std: total_scale *= state["std"]
        if self.div_by_std: total_scale *= 1 / state["std"]
        if self.mul_by_lr: total_scale *= state["lr"]
        if self.mul_by_lr_scalar: total_scale *= state[(parameter_id, "lr_scalar")]

        if self.div_by_rstd: total_scale *= 1 / (state["rstd"] + self.epsilon)
        if self.div_by_rmsprop_block and (parameter_id, "rmsprop_block") in state: total_scale *= 1 / (math.sqrt(state[(parameter_id, "rmsprop_block")]/(1 - state[(parameter_id, "rmsprop_block_decay_coeff")])) + self.epsilon)
        perturbation = state[(parameter_id, "perturb_buffer")]
        perturbation.mul_(total_scale)

class OC_Sign_Perturb_Buffer(OptimizerChain):
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int):
        if (parameter_id, "perturb_buffer") not in state:
            raise ValueError("State dict is missing the perturbation to sign.")
        perturbation = state[(parameter_id, "perturb_buffer")]
        perturbation.sign_()

class OC_Clear_Perturb_Buffer(OptimizerChain):
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int):
        if (parameter_id, "perturb_buffer") not in state:
            raise ValueError("State dict is missing the perturbation to zero.")
        perturbation = state[(parameter_id, "perturb_buffer")]
        perturbation.zero_()

class OC_Commit_Perturb_Buffer(OptimizerChain):
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int):
        if (parameter_id, "perturb_buffer") not in state:
            raise ValueError("State dict is missing the perturbation to commit.")
        perturbation = state[(parameter_id, "perturb_buffer")]
        tensor.add_(perturbation)

class OC_Delete_Perturb_Buffer(OptimizerChain):
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int):
        if (parameter_id, "perturb_buffer") not in state:
            raise ValueError("State dict is missing the perturbation to delete.")
        del state[(parameter_id, "perturb_buffer")]

class OC_Muon_Whiten_Perturb_Buffer(OptimizerChain):
    """WARNING: THIS WILL UNSCALE THE GRADIENT, ALSO ONLY WORKS ON NDIM==2 TENSORS"""
    def newtonschulz5(self, G: torch.Tensor, steps=5, eps=1e-7):
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.T if G.size(0) > G.size(1) else G
        X.div_(X.norm() + eps)
        A = torch.empty((X.size(0), X.size(0)), dtype=X.dtype, device=X.device)
        B = torch.empty((X.size(0), X.size(0)), dtype=X.dtype, device=X.device)
        for _ in range(steps):
            torch.matmul(X, X.T, out=A)
            torch.addmm(A, A, A, beta=b, alpha=c, out=B)
            X.addmm_(B, X, beta=a)

    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int):
        if (parameter_id, "perturb_buffer") not in state:
            raise ValueError("State dict is missing the perturbation to whiten.")
        perturbation = state[(parameter_id, "perturb_buffer")]
        if tensor.ndim == 2:
            self.newtonschulz5(perturbation)

class OC_Compute_RMSProp_Blockwise(OptimizerChain):
    def __init__(self, coeff_old = 0.95, coeff_new = 0.05, decay_coeff = 0.95):
        self.coeff_old = coeff_old
        self.coeff_new = coeff_new
        self.decay_coeff = decay_coeff

    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int):
        if (parameter_id, "perturb_buffer") not in state:
            raise ValueError("State dict is missing the perturbation to compute RMSProp.")
        if (parameter_id, "rmsprop_block") not in state:
            state[(parameter_id, "rmsprop_block")] = 0
        if (parameter_id, "rmsprop_block_decay_coeff") not in state:
            state[(parameter_id, "rmsprop_block_decay_coeff")] = self.decay_coeff
        else:
            state[(parameter_id, "rmsprop_block_decay_coeff")] *= self.decay_coeff

        perturbation = state[(parameter_id, "perturb_buffer")]
        grads_squared_new = (torch.linalg.vector_norm(perturbation, dtype=torch.float32) ** 2 / perturbation.numel()).item()
        state[(parameter_id, "rmsprop_block")] = (state[(parameter_id, "rmsprop_block")] * self.coeff_old + grads_squared_new * self.coeff_new)
