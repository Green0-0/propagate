from propagate.optimizers.chain import OptimizerChain
from abc import ABC, abstractmethod
from typing import Dict

import torch
import random

from propagate.genome import Genome

class PSampler(ABC):
    """
    The base class for all perturbation samplers.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def sample(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        pass

class Gaussian_PSampler(PSampler):
    """Samples Gaussian noise for the perturbation buffer."""
    def __init__(self, fp32_accumulate: bool = True):
        self.fp32_accumulate = fp32_accumulate

    @torch.no_grad()
    def sample(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        gen = torch.Generator(device=tensor.device)
        perturbation = torch.zeros_like(tensor, dtype = torch.float32 if self.fp32_accumulate else tensor.dtype)
        buffer = torch.empty_like(tensor)
        for seed, weight in zip(source.seeds, source.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            buffer.normal_(generator=gen)
            perturbation.add_(buffer, alpha=float(weight))
        return perturbation

class Bernoulli_PSampler(PSampler):
    """Samples Bernoulli noise for the perturbation buffer."""
    def __init__(self, center: float = 0.5, fp32_accumulate: bool = True):
        self.center = center
        self.fp32_accumulate = fp32_accumulate

    @torch.no_grad()
    def sample(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        gen = torch.Generator(device=tensor.device)
        perturbation = torch.zeros_like(tensor, dtype = torch.float32 if self.fp32_accumulate else tensor.dtype)
        buffer = torch.empty_like(tensor)
        for seed, weight in zip(source.seeds, source.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            buffer.random_(0, 2, generator=gen).sub_(self.center).mul_(2)
            perturbation.add_(buffer, alpha=float(weight))
        return perturbation

class Layerwise_Deterministic_Sparse_PSampler(PSampler):
    """Samples from the target PSampler, and then for a specific layer and seed, deterministically either zero the noise or keep it.
    
    Note: A fixed seed on a fixed layer will either always be zero, or always be kept."""
    def __init__(self, sparse_probability: float, base_sampler: PSampler):
        self.sparse_probability = sparse_probability
        self.base_sampler = base_sampler

    @torch.no_grad()
    def sample(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        step_noise = self.base_sampler.sample(source, state, parameter_id, tensor, random_offset, do_log)
        
        keep_gen = torch.Generator(device='cpu')
        keep_gen.manual_seed(random_offset + int(source.seeds[-1]) + 99999)
        
        if torch.rand(1, generator=keep_gen).item() < self.sparse_probability:
            step_noise.zero_()
            
        return step_noise

class Elementwise_Deterministic_Sparse_PSampler(PSampler):
    """Samples from the target PSampler, and then for each element, deterministically either zero the noise or keep it.
    
    Note: A fixed seed on a fixed element will either always be zero, or always be kept.
    """
    def __init__(self, sparse_probability: float, base_sampler: PSampler):
        self.sparse_probability = sparse_probability
        self.base_sampler = base_sampler

    @torch.no_grad()
    def sample(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        step_noise = self.base_sampler.sample(source, state, parameter_id, tensor, random_offset, do_log)
        
        keep_gen = torch.Generator(device=tensor.device)
        keep_gen.manual_seed(random_offset + int(source.seeds[-1]) + 99999)
        
        mask = torch.rand_like(tensor, dtype=torch.float32, generator=keep_gen) < self.sparse_probability
        step_noise.masked_fill_(mask, 0.0)
        
        return step_noise

# Note: The resamplers do not currently work

class Memorizer(OptimizerChain):
    """Memorizes the current Genome's perturbation status and adds it to a set (to prevent duplicates on different layers) with key 'mem_genomes' in the state dict."""
    def __init__(self):
        pass

    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "mem_genomes" not in state:
            state["mem_genomes"] = []
            
        source_seeds_tuple = tuple(source.seeds)
        already_exists = False
        
        if source.perturb_scales:
            v1 = torch.tensor(source.perturb_scales, dtype=torch.float32)
            v1_sqnorm = torch.dot(v1, v1)
            
        for mem_g in state["mem_genomes"]:
            if tuple(mem_g.seeds) == source_seeds_tuple:
                if not source.perturb_scales:
                    already_exists = True
                    break
                    
                v2 = torch.tensor(mem_g.perturb_scales, dtype=torch.float32)
                v2_sqnorm = torch.dot(v2, v2)
                
                if v2_sqnorm < 1e-9:
                    is_multiple = (v1_sqnorm < 1e-9)
                else:
                    is_multiple = torch.allclose(v1, (torch.dot(v1, v2) / v2_sqnorm) * v2, atol=1e-5)
                    
                if is_multiple:
                    already_exists = True
                    break
                
        if not already_exists:
            state["mem_genomes"].append(source.get_copy())

class Resample_PSampler(PSampler):
    """Samples perturbations from the memorized genomes (multiplied randomly by randmul_min/max) with some probability."""
    def __init__(self, base_sampler: PSampler, resample_probability: float, resample_min:int = 1, randmul_min=0.75, randmul_max=1.25):
        self.base_sampler = base_sampler
        self.resample_probability = resample_probability
        self.resample_min = resample_min
        self.randmul_min = randmul_min
        self.randmul_max = randmul_max

    @torch.no_grad()
    def sample(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "step" not in state:
            raise ValueError("Step not received, the resampler won't function!")
        step = state["step"]
        
        key_decision = f"resample_decision_{id(self)}"
        if key_decision not in state or state[key_decision].get("step") != step:
            state[key_decision] = {"step": step, "decisions": {}}
            
        genome_key = id(source)
        
        if genome_key not in state[key_decision]["decisions"]:
            mem_genomes = state.get("mem_genomes", [])
            
            if len(mem_genomes) >= self.resample_min and random.random() < self.resample_probability:
                mem_genome = random.choice(mem_genomes)
                rand_mul = random.uniform(self.randmul_min, self.randmul_max)
                state[key_decision]["decisions"][genome_key] = True
                
                source.seeds = mem_genome.seeds.copy()
                source.perturb_scales = [s * rand_mul for s in mem_genome.perturb_scales]
            else:
                state[key_decision]["decisions"][genome_key] = False
                
        return self.base_sampler.sample(source, state, parameter_id, tensor, random_offset, do_log)

class Phased_Resampler(PSampler):
    """Alternates sampling such that normal sampling is done for sampling_steps, and then resampling under memorized genomes is done for resampling_steps."""
    def __init__(self, base_sampler: PSampler, sampling_steps: int, resampling_steps:int, randmul_min=0.75, randmul_max=1.25):
        self.base_sampler = base_sampler
        self.sampling_steps = sampling_steps
        self.resampling_steps = resampling_steps
        self.randmul_min = randmul_min
        self.randmul_max = randmul_max

    @torch.no_grad()
    def sample(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if "step" not in state:
            raise ValueError("Step not received, the phased resampler won't function!")
        step = state["step"]
        
        phase_length = self.sampling_steps + self.resampling_steps
        current_phase_step = (step - 1) % phase_length
        is_resampling_phase = current_phase_step >= self.sampling_steps
        
        key_decision = f"phased_resample_decision_{id(self)}"
        if key_decision not in state or state[key_decision].get("step") != step:
            state[key_decision] = {"step": step, "decisions": {}}
            
        genome_key = id(source)
        
        if genome_key not in state[key_decision]["decisions"]:
            mem_genomes = state.get("mem_genomes", [])
            if is_resampling_phase and len(mem_genomes) > 0:
                mem_genome = random.choice(mem_genomes)
                rand_mul = random.uniform(self.randmul_min, self.randmul_max)
                state[key_decision]["decisions"][genome_key] = True
                
                source.seeds = mem_genome.seeds.copy()
                source.perturb_scales = [s * rand_mul for s in mem_genome.perturb_scales]
            else:
                state[key_decision]["decisions"][genome_key] = False
                
        return self.base_sampler.sample(source, state, parameter_id, tensor, random_offset, do_log)