import pytest
import torch
from propagate.genome import Genome

def test_genome_initialization():
    g = Genome()
    assert g.seeds == []
    assert g.perturb_scales == []
    assert g.historical_rewards == []
    assert g.starting_index == 0

def test_mutate_seed():
    g = Genome()
    scale = 0.5
    seed = g.mutate_seed(scale)
    
    assert len(g.seeds) == 1
    assert len(g.perturb_scales) == 1
    assert g.perturb_scales[0] == scale
    assert isinstance(seed, int)
    assert g.seeds[0] == seed

def test_genome_copy():
    g = Genome()
    g.mutate_seed(0.5)
    g.historical_rewards = [1.0]
    
    g_copy = g.get_copy()
    assert g_copy.seeds == g.seeds
    assert g_copy.perturb_scales == g.perturb_scales
    assert g_copy.historical_rewards == g.historical_rewards
    assert g_copy.starting_index == g.starting_index
    
    # Verify deep copy behavior for lists
    g.seeds.append(123)
    assert len(g_copy.seeds) == 1  # Copy should not change

def test_genome_mirror():
    g = Genome()
    g.mutate_seed(0.5)
    
    g_mirror = g.get_mirrored()
    assert g_mirror.seeds == g.seeds
    assert len(g_mirror.perturb_scales) == 1
    # Check if the last scale is negated
    assert g_mirror.perturb_scales[-1] == -0.5
    
    # Should raise error if no seeds
    empty_g = Genome()
    with pytest.raises(ValueError):
        empty_g.get_mirrored()
