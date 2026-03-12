import pytest
import torch
from propagate.optimizers import chain
from propagate.genome import Genome

"""
Tests for basic Chain Operations.
These tests cover simple arithmetic and logic operations in the perturbation chain
such as scaling, adding, subtracting, clipping, and initialization.
"""

def test_scale_perturbation(dummy_tensor, dummy_state, basic_genome):
    """
    Test individual and combined scaling flags:
    - div_by_pop: Divide by population size
    - mul_by_std: Multiply by standard deviation
    - mul_by_lr: Multiply by learning rate
    """
    # Setup state
    dummy_state.update({
        "population_size": 10,
        "std": 2.0,
        "rstd": 0.5, # rstd is 1/std usually, but here just a param
        "lr": 0.1,
    })
    
    # Base perturbation
    perturbation = torch.ones_like(dummy_tensor)
    dummy_state["perturb_buffer"] = perturbation.clone()

    # 1. Test div_by_pop
    op = chain.Scale_Perturbation(div_by_pop=True)
    op.apply(basic_genome, dummy_state, "param1", dummy_tensor, 0)
    assert torch.allclose(dummy_state["perturb_buffer"], perturbation / 10), \
        "Failed to divide by population size"
    
    # Reset
    dummy_state["perturb_buffer"] = perturbation.clone()
    
    # 2. Test mul_by_std
    op = chain.Scale_Perturbation(mul_by_std=True)
    op.apply(basic_genome, dummy_state, "param1", dummy_tensor, 0)
    # Scale uses 'std' from state
    assert torch.allclose(dummy_state["perturb_buffer"], perturbation * 2.0), \
        "Failed to multiply by std"

    # Reset
    dummy_state["perturb_buffer"] = perturbation.clone()
    
    # 3. Test combination (div_by_pop + mul_by_lr)
    op = chain.Scale_Perturbation(div_by_pop=True, mul_by_lr=True)
    # Expected: 1.0 * (0.1 / 10) = 0.01
    op.apply(basic_genome, dummy_state, "param1", dummy_tensor, 0)
    assert torch.allclose(dummy_state["perturb_buffer"], perturbation * 0.01), \
        "Failed combined scaling logic"

def test_add_sub_perturbation(dummy_tensor, dummy_state, basic_genome):
    """Test applying and removing the perturbation buffer from weights."""
    # Weights start at 0 (from fixture)
    dummy_state["perturb_buffer"] = torch.full_like(dummy_tensor, 1.5)
    
    # Add
    op_add = chain.Add_Perturb_Buffer()
    op_add.apply(basic_genome, dummy_state, "p", dummy_tensor, 0)
    assert torch.allclose(dummy_tensor, torch.full_like(dummy_tensor, 1.5)), \
        "Add_Perturb_Buffer failed to update weights"
    
    # Sub (Reverse)
    op_sub = chain.Sub_Perturb_Buffer()
    op_sub.apply(basic_genome, dummy_state, "p", dummy_tensor, 0)
    assert torch.allclose(dummy_tensor, torch.zeros_like(dummy_tensor)), \
        "Sub_Perturb_Buffer failed to restore weights"

def test_sign_perturbation(dummy_tensor, dummy_state, basic_genome):
    """Test converting perturbation to its sign {-1, 0, 1}."""
    op = chain.Sign_Perturb_Buffer()
    
    dummy_state["perturb_buffer"] = torch.tensor([-2.0, 0.0, 2.0])
    # Dummy tensor needs to match size 3 for this manual buffer
    short_tensor = torch.zeros(3)
    
    op.apply(basic_genome, dummy_state, "p", short_tensor, 0)
    
    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(dummy_state["perturb_buffer"], expected), \
        "Sign_Perturb_Buffer produced incorrect signs"

def test_abs_perturbation(dummy_tensor, dummy_state, basic_genome):
    """Test taking the absolute value of the perturbation buffer."""
    op = chain.Abs_Perturb_Buffer()
    
    dummy_state["perturb_buffer"] = torch.tensor([-1.0, 2.0, -3.0])
    short_tensor = torch.zeros(3)
    
    op.apply(basic_genome, dummy_state, "p", short_tensor, 0)
    
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(dummy_state["perturb_buffer"], expected), \
        "Abs_Perturb_Buffer failed"

def test_max_sub_exp(dummy_tensor, dummy_state, basic_genome):
    """
    Test Max-Sub-Exp logic (Numerical stability trick for Softmax).
    Steps:
    1. Find Max
    2. Subtract Max
    3. Exponentiate
    """
    op = chain.MaxSubExp_Perturb_Buffer()
    
    # Input: All 10s. Max is 10.
    # Sub(10) -> [0, 0, 0].
    # Exp(0) -> [1, 1, 1].
    dummy_state["perturb_buffer"] = torch.tensor([10.0, 10.0, 10.0])
    short_tensor = torch.zeros(3)
    
    op.apply(basic_genome, dummy_state, "p", short_tensor, 0)
    
    expected = torch.tensor([1.0, 1.0, 1.0])
    assert torch.allclose(dummy_state["perturb_buffer"], expected), \
        "MaxSubExp logic failed"

def test_copy_weights_chain(dummy_tensor, dummy_state, basic_genome):
    """Test copying current weights INTO the perturbation buffer."""
    op = chain.Copy_Weights_To_Perturb_Buffer()
    
    # Set weights to specific value
    dummy_tensor.fill_(0.5)
    
    # Clear buffer to ensure it's overwritten
    if "perturb_buffer" in dummy_state: 
        del dummy_state["perturb_buffer"]
        
    op.apply(basic_genome, dummy_state, "p", dummy_tensor, 0)
    
    assert "perturb_buffer" in dummy_state
    # Check values (Copy usually casts to bfloat16/float depending on config, check closeness)
    assert torch.allclose(dummy_state["perturb_buffer"].float(), dummy_tensor), \
        "Failed to copy weights to buffer"

def test_override_weights_chain(dummy_tensor, dummy_state, basic_genome):
    """Test overwriting weights WITH the perturbation buffer."""
    op = chain.Override_Weights_With_Perturb_Buffer()
    
    # Set buffer
    dummy_state["perturb_buffer"] = torch.full_like(dummy_tensor, 9.0)
    # Weights are 0
    dummy_tensor.zero_()
    
    op.apply(basic_genome, dummy_state, "p", dummy_tensor, 0)
    
    assert torch.allclose(dummy_tensor, torch.full_like(dummy_tensor, 9.0)), \
        "Failed to override weights with buffer"

def test_bernoulli_init(dummy_tensor, dummy_state, basic_genome):
    """
    Test Bernoulli Initialization.
    Logic: Random(0, 2) -> {0, 1}. Center(0.5) -> {-0.5, 0.5}. Scale(2) -> {-1, 1}.
    """
    op = chain.Init_Perturbation_Bernoulli(center=0.5, fp32_accumulate=False)
    
    if "perturb_buffer" in dummy_state: del dummy_state["perturb_buffer"]
    
    op.apply(basic_genome, dummy_state, "p", dummy_tensor, 0)
    
    result = dummy_state["perturb_buffer"]
    unique_vals = torch.unique(torch.round(result)) # Round to handle float precision
    
    for val in unique_vals:
        assert abs(abs(val.item()) - 1.0) < 1e-5, \
            f"Bernoulli init produced {val.item()}, expected -1 or 1"
