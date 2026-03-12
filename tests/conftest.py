import pytest
import torch
from propagate.genome import Genome

@pytest.fixture
def dummy_tensor():
    # A small fixed tensor for easy manual verification
    return torch.zeros(10, dtype=torch.float32)

@pytest.fixture
def dummy_state():
    # A clean state dictionary
    return {"step": 1, "lr": 1.0, "std": 1.0, "rstd": 1.0, "population_size": 1, "lr_scalar": 1.0}

@pytest.fixture
def basic_genome():
    g = Genome()
    g.mutate_seed(1.0) # Adds one seed
    g.historical_rewards = [0.0]
    return g

@pytest.fixture
def complex_genome():
    g = Genome()
    g.seeds = [100, 200, 300]
    g.perturb_scales = [0.5, 0.3, -0.2]
    return g
