from propagate.optimizers.chain import OptimizerChain
from propagate.genome import Genome

from typing import Dict

import torch

class OC_Muon_Whiten_Perturb_Buffer(OptimizerChain):
    def __init__(self, force_bf16 = True) -> None:
        self.force_bf16 = force_bf16
        
    """WARNING: THIS WILL UNSCALE THE GRADIENT, ALSO ONLY WORKS ON NDIM==2 TENSORS"""
    def newtonschulz5(self, G: torch.Tensor, steps=5, eps=1e-5):
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.T if G.size(0) > G.size(1) else G
        X = X.to(dtype=torch.bfloat16 if self.force_bf16 else G.dtype, copy=True)
        
        X.div_(X.norm() + eps)
        A = torch.empty((X.size(0), X.size(0)), dtype=X.dtype, device=X.device)
        B = torch.empty((X.size(0), X.size(0)), dtype=X.dtype, device=X.device)
        X_next = torch.empty_like(X)
        for _ in range(steps):
            torch.matmul(X, X.T, out=A)
            torch.addmm(A, A, A, beta=b, alpha=c, out=B)
            torch.addmm(X, B, X, beta=a, out = X_next)
            X, X_next = X_next, X
        G.copy_(X.T if G.size(0) > G.size(1) else X)
        del A, B, X_next, X

    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to whiten.")
        perturbation = state["perturb_buffer"]
        if tensor.ndim == 2:
            self.newtonschulz5(perturbation)
            
class OC_Manifold_Project(OptimizerChain): 
    """NOTE: T MUST BE POSITIVE!"""       
    def sinkhorn_knopp(self, T: torch.Tensor, steps = 20, eps=1e-5):
        assert T.ndim == 2
        assert (T >= 0).all(), "Sinkhorn input must be strictly positive."
        for _ in range(0, steps):
            T.div_(torch.clamp(T.sum(dim=1, keepdim=True), min=eps))
            T.div_(torch.clamp(T.sum(dim=0, keepdim=True), min=eps))
            
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to project.")
        perturbation = state["perturb_buffer"]
        if tensor.ndim == 2:
            self.sinkhorn_knopp(perturbation)

class Probabilistic_Zero_Perturb_Buffer(OptimizerChain):
    def __init__(self, probability_to_zero = 0.5):
        self.probability_to_zero = probability_to_zero
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "perturb_buffer" not in state:
            raise ValueError("State dict is missing the perturbation to zero.")
        perturbation = state["perturb_buffer"]
        if torch.rand(1).item() < self.probability_to_zero:
            perturbation.zero_()
        
class Direct_Weight_Decay(OptimizerChain):
    def __init__(self, lambda_val = 0.0001, exponent=0.5) -> None:
        self.lambda_val = lambda_val
        self.exponent = exponent
    
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        decay = tensor.abs().pow_(self.exponent).copysign_(tensor).mul_(self.lambda_val)
        tensor.sub_(decay)
        del decay
        
# todo?: Variance/std calculation on independent seeds to squeeze certain parts of the gradient which are noisy/have low confidence
# a statistical optimizer could be useful...