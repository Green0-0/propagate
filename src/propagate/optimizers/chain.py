from abc import ABC, abstractmethod
from typing import Dict

import torch

from propagate.genome import Genome

import math

class OptimizerChain(ABC):
    """
    To encourage modularity and customizability, optimizers have been broken down into individual operations, which are meant to be chained in sequence. For example, ADAM in ES is simply a chain consisting of generating the gradient buffer, updating the momentum, updating the rmsprop, resetting the buffer, adding the momentum back into the buffer, scaling by the rmsprop, and finally updating the tensor.
    
    Always remember your basic operations! Don't forget to delete and zero buffers (so that you don't add the gradient and then the momentum which double counts), and remember to correctly scale your seeds by your hyperparameters, as they don't come scaled naturally."""
    @abstractmethod
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        """Does some operation by referencing the source genome and the internal optimizer state. May update the optimizer state or the given tensor, and may also log.
        
        Args: 
            source : Genome
                The genome to use as a source for seeds and weights. This genome may also encapsulate a gradient update.
            state : Dict
                The internal state dictionary of the optimizer.
            parameter_id : str
                The unique identifier of the parameter.
            tensor : torch.Tensor
                The tensor to apply the operation to.
            random_offset : int
                A random offset to use for seeding, so that tensors get different perturbations. This random_offset should be the same for the same tensor.
            do_log : bool
                Whether to log statistics. Only rank 0 should log, ie. the first GPU.
        """
        pass

class Sign_Perturb_Buffer(OptimizerChain):
    """Takes the sign of the current perturbation buffer in-place. Used for SignSGD."""
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to sign.")
        perturbation = state["perturb_buffer"]
        perturbation.sign_()

class Abs_Perturb_Buffer(OptimizerChain):
    """Computes the absolute value of the current perturbation buffer in-place. Probably useless."""
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to sign.")
        perturbation = state["perturb_buffer"]
        perturbation.abs_()

class MaxSubExp_Perturb_Buffer(OptimizerChain):
    """Standard operations for converting logits to probabilities. Subtracts the max and exponentiates the buffer. Might be useful for Deepseek MHC, if that ever gets used anywhere."""
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to sign.")
        perturbation = state["perturb_buffer"]
        perturbation.sub_(torch.max(perturbation))
        perturbation.exp_()
        
class Zero_Perturb_Buffer(OptimizerChain):
    """Zeroes the current perturbation buffer. This is very important, as you must zero the perturbation buffer after calculating momentum, and add the momentum back (unless you want to double the current gradient, which is fine too)."""
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to zero.")
        perturbation = state["perturb_buffer"]
        perturbation.zero_()

class Add_Perturb_Buffer(OptimizerChain):
    """Adds the perturbation buffer to the model parameters (tensor)."""
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to commit.")
        perturbation = state["perturb_buffer"]
        tensor.add_(perturbation.to(tensor.dtype))

class Sub_Perturb_Buffer(OptimizerChain):
    """Subtracts the perturbation buffer from the model parameters (tensor)."""
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to commit.")
        perturbation = state["perturb_buffer"]
        tensor.sub_(perturbation.to(tensor.dtype))

class Delete_Perturb_Buffer(OptimizerChain):
    """Deletes the perturbation buffer from the state dictionary. This is probably useless, but good practice and may occasionally save peak memory if used appropriately."""
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to delete.")
        del state["perturb_buffer"]
        
class Copy_Weights_To_Perturb_Buffer(OptimizerChain):
    """Copies the current tensor into the perturbation buffer. Useful if you want to directly modify the tensor.
    
    Attributes
    ----------
    cast_type : torch.dtype
        The data type to cast the weights to.
    """
    def __init__(self, cast_type: torch.dtype = torch.bfloat16) -> None:
        self.cast_type = cast_type
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" in state:
            raise ValueError("Perturbation buffer was requested to be copied, but already exists! Did you forget to delete it?")
        state["perturb_buffer"] = tensor.to(dtype=self.cast_type, copy=True)
        
class Override_Weights_With_Perturb_Buffer(OptimizerChain):
    """Overrides the model weights with the values in the perturbation buffer. Meant to be used with the copying optimizer chain, so that you copy the tensor, modify it, and write it back in."""
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("Perturbation buffer does not exist to override?")
        tensor.copy_(state["perturb_buffer"])
        
class Init_Perturbation_Gaussian(OptimizerChain):
    """Initializes the perturbation buffer with Gaussian noise. 
    Warning: The gaussian is multiplied by the weight of the seed. This is important for mirroring (where the weight might be -1), but make sure to not double count the perturbation scale by accident. By default the perturbation is not generated with the perturbation scale, it must be scaled later. 

    Attributes
    ----------
    fp32_accumulate : bool
        Whether to accumulate the noise in float32.
    """
    def __init__(self, fp32_accumulate: bool = True):
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
    """Initializes the perturbation buffer with Bernoulli/Rademacher noise: {(-center) * 2, (0.5 - center) * 2}.
    Warning: The bernoulli is multiplied by the weight of the seed. This is important for mirroring (where the weight might be -1), but make sure to not double count the perturbation scale by accident. By default the perturbation is not generated with the perturbation scale, it must be scaled later.
    
    Attributes
    ----------
    center : float
        The value to recenter the noise around. Defaults to 0.5, which means we take the standard bernoulli mean, 0.5, and shift it to 0.
    fp32_accumulate : bool
        Whether to accumulate the noise in float32.
    """
    def __init__(self, center: float = 0.5, fp32_accumulate: bool = True):
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
    """Scales the perturbation buffer by various factors. For perturbation, the default is to only scale with ``mul_by_std``, while for gradient update you might consider various mixtures. It is recommended to always set ``mul_by_lr_scalar`` to true, as it helps with lora compatibility.
    
    A recommended mixture for gradient update with mirrors is ``mul_by_lr``, ``mul_by_lr_scalar``, ``mul_by_std``. OpenAI suggests to divide by std instead (std is the perturbation_scale) but according to the inverse fisher matrix from natural gradient descent, we should multiply by std^2/std, which works out to std. Intuitively, this means if we try out small perturbations, we should also update with small perturbations, which makes sense.

    It is EXTREMELY important the order in which you apply the perturbation scale. If you apply it early, it might get cancelled out by RMSProp/Muon/Sign, and applying it late affects how your momentum values are computed. You could also try applying it multiple times. Double check your math!
    
    Attributes
    ----------
    div_by_pop : bool
        Whether to divide by the population size.
    div_by_rstd : bool
        Whether to divide by the reward standard deviation.
    mul_by_std : bool
        Whether to multiply by the perturbation standard deviation. This is important for stable unmirrored gradients.
    mul_by_lr : bool
        Whether to multiply by the learning rate.
    mul_by_lr_scalar : bool
        Whether to multiply by the learning rate scalar (usually 1.0).
    div_by_std : bool
        Whether to divide by the perturbation standard deviation.
    div_by_rmsprop_block : bool
        Whether to divide by the blockwise RMSProp value.
    epsilon : float
        The epsilon value for stability.
    """
    def __init__(self, div_by_pop: bool = False, div_by_rstd: bool = False, mul_by_std: bool = False, mul_by_lr: bool = False, mul_by_lr_scalar: bool = False, div_by_std: bool = False, div_by_rmsprop_block: bool = False, epsilon: float = 1e-5):
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