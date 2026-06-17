import pytest
import torch
import random
from propagate.genome import Genome
from propagate.optimizers.psamplers import Gaussian_PSampler, Memorizer, Resample_PSampler, Phased_Resampler
from propagate.optimizers.chain import OptimizerChain

def test_memorizer_basic():
    # Test that memorizer records genomes and prevents duplicates across steps.
    memo = Memorizer(dedupe=True, discount_longer=True)
    state = {"step": 1}
    g1 = Genome()
    g1.seeds=[1]
    g1.perturb_scales=[1.0]
    
    memo.apply(g1, state, "dummy_param", None, 0)
    assert "mem_genomes" in state
    assert len(state["mem_genomes"]) == 1
    assert state["mem_genomes"][0].seeds == [1]
    
    # Test dedupe: Identical genome on the SAME step but different layer is skipped by last_memo_step
    memo.apply(g1, state, "dummy_param_2", None, 0)
    assert len(state["mem_genomes"]) == 1
    
    # Test dedupe: Identical genome on a DIFFERENT step is skipped by dedupe logic
    state["step"] = 2
    memo.apply(g1, state, "dummy_param", None, 0)
    assert len(state["mem_genomes"]) == 1

def test_memorizer_expansion():
    # Test that memorizer expands resampled seeds before storing
    memo = Memorizer(dedupe=True, discount_longer=False)
    state = {"step": 1, "resample_mapping": {}}
    
    # Genome with seed_1 resampled into mem_genome_1
    mem_genome_1 = Genome()
    mem_genome_1.seeds=[100]
    mem_genome_1.perturb_scales=[0.5]
    state["resample_mapping"][1] = (mem_genome_1, 2.0)
    
    g = Genome()
    g.seeds=[1]
    g.perturb_scales=[1.0]
    memo.apply(g, state, "dummy_param", None, 0)
    
    assert len(state["mem_genomes"]) == 1
    stored_g = state["mem_genomes"][0]
    # The stored genome should have the expanded seeds
    assert stored_g.seeds == [100]
    # Scale = 1.0 (original) * 0.5 (mem) * 2.0 (rand_mul) = 1.0
    assert stored_g.perturb_scales == [1.0]

def test_resample_psampler_basic():
    base = Gaussian_PSampler()
    # 100% resample prob
    sampler = Resample_PSampler(base, resample_probability=1.0, resample_min=1, randmul_min=1.0, randmul_max=1.0)
    
    mem_g = Genome()
    mem_g.seeds=[100]
    mem_g.perturb_scales=[0.5]
    state = {"step": 1, "mem_genomes": [mem_g]}
    g = Genome()
    g.seeds=[1]
    g.perturb_scales=[1.0]
    
    tensor = torch.zeros((10, 10))
    # Should resample since prob=1.0 and min=1
    out_tensor = sampler.sample(g, state, "dummy_param", tensor, 0)
    
    assert "resample_mapping" in state
    assert 1 in state["resample_mapping"]
    mapping = state["resample_mapping"][1]
    assert mapping is not None
    assert mapping[0].seeds == [100]
    
    # Same seed on different layer should use SAME mapping
    tensor2 = torch.zeros((10, 10))
    sampler.sample(g, state, "dummy_param_2", tensor2, 0)
    # The mapping should not be cleared because step is the same
    assert state["resample_mapping"][1] is mapping

def test_resample_psampler_cache_flush():
    base = Gaussian_PSampler()
    sampler = Resample_PSampler(base, resample_probability=1.0, resample_min=1, randmul_min=1.0, randmul_max=1.0)
    
    mem_g = Genome()
    mem_g.seeds=[100]
    mem_g.perturb_scales=[0.5]
    state = {"step": 1, "mem_genomes": [mem_g]}
    g = Genome()
    g.seeds=[1]
    g.perturb_scales=[1.0]
    
    tensor = torch.zeros((10, 10))
    sampler.sample(g, state, "dummy_param", tensor, 0)
    assert 1 in state["resample_mapping"]
    
    # Next step: should flush
    state["step"] = 2
    g2 = Genome()
    g2.seeds=[2]
    g2.perturb_scales=[1.0]
    sampler.sample(g2, state, "dummy_param", tensor, 0)
    
    assert 1 not in state["resample_mapping"]
    assert 2 in state["resample_mapping"]

def test_phased_resampler():
    base = Gaussian_PSampler()
    sampler = Phased_Resampler(base, sampling_steps=2, resampling_steps=2, randmul_min=1.0, randmul_max=1.0)
    
    mem_g = Genome()
    mem_g.seeds=[100]
    mem_g.perturb_scales=[0.5]
    state = {"mem_genomes": [mem_g]}
    g = Genome()
    g.seeds=[1]
    g.perturb_scales=[1.0]
    tensor = torch.zeros((10, 10))
    
    # Step 1: sampling phase
    state["step"] = 1
    sampler.sample(g, state, "dummy", tensor, 0)
    assert state["resample_mapping"][1] is None
    
    # Step 2: sampling phase
    state["step"] = 2
    sampler.sample(g, state, "dummy", tensor, 0)
    assert state["resample_mapping"][1] is None
    
    # Step 3: resampling phase
    state["step"] = 3
    sampler.sample(g, state, "dummy", tensor, 0)
    assert state["resample_mapping"][1] is not None
    
    # Step 4: resampling phase
    state["step"] = 4
    sampler.sample(g, state, "dummy", tensor, 0)
    assert state["resample_mapping"][1] is not None
    
    # Step 5: back to sampling phase
    state["step"] = 5
    sampler.sample(g, state, "dummy", tensor, 0)
    assert state["resample_mapping"][1] is None
