import pytest
import torch
import math
from propagate.optimizers import chain, chain_adam, chain_misc, chain_adam_seeded
from propagate.genome import Genome

"""
Tests for Complex Chain Operations.
Includes:
- RMSProp (Compute, Apply, Blockwise, Seeded)
- Momentum
- Muon Whitening
- Seed History Management
"""

# --- ADAM / RMSPROP ---

def test_rmsprop_computation(dummy_tensor, dummy_state, basic_genome):
    """
    Test standard RMSProp calculation: E[g^2] = beta * E[g^2] + (1-beta) * g^2
    """
    # Force init value > 0 disables bias correction for this test
    op = chain_adam.OC_Compute_RMSProp(coeff_old=0.9, coeff_new=0.1, force_init_value=1.0)
    
    # Step 1: Perturbation = 2.0
    dummy_state["perturb_buffer"] = torch.full_like(dummy_tensor, 2.0)
    op.apply(basic_genome, dummy_state, "param1", dummy_tensor, 0)
    
    # Expected: 1.0 * 0.9 + (2.0^2) * 0.1 = 0.9 + 0.4 = 1.3
    expected_sq = 1.3
    assert torch.allclose(dummy_state[("param1", "rmsprop")], torch.tensor(expected_sq)), \
        "RMSProp step 1 calculation incorrect"
    
    # Step 2: Perturbation = 0.0
    dummy_state["perturb_buffer"] = torch.zeros_like(dummy_tensor)
    op.apply(basic_genome, dummy_state, "param1", dummy_tensor, 0)
    
    # Expected: 1.3 * 0.9 + 0.0 = 1.17
    expected_sq = 1.17
    assert torch.allclose(dummy_state[("param1", "rmsprop")], torch.tensor(expected_sq)), \
        "RMSProp step 2 calculation incorrect"

def test_rmsprop_bias_correction(dummy_tensor, dummy_state, basic_genome):
    """
    Test RMSProp with Bias Correction (init from 0).
    Steps:
    1. Init 0.
    2. Update with 2.0 -> Avg = 0.4.
    3. Correction factor 1 - 0.9 = 0.1.
    4. Corrected Avg = 0.4 / 0.1 = 4.0.
    5. Sqrt(4.0) = 2.0.
    6. Perturb(2.0) / 2.0 = 1.0.
    """
    # force_init_value = -999 enables bias correction logic
    op_compute = chain_adam.OC_Compute_RMSProp(coeff_old=0.9, coeff_new=0.1, force_init_value=-999)
    op_apply = chain_adam.OC_Apply_RMSProp(epsilon=0.0)

    dummy_state["perturb_buffer"] = torch.full_like(dummy_tensor, 2.0)
    
    # Compute Update
    op_compute.apply(basic_genome, dummy_state, "param1", dummy_tensor, 0)
    
    # Internal raw value check
    raw_val = 0.4 # 0*0.9 + 4*0.1
    assert torch.allclose(dummy_state[("param1", "rmsprop")], torch.tensor(raw_val)), \
        "Raw RMSProp accumulation incorrect"
    
    # Apply Normalization
    op_apply.apply(basic_genome, dummy_state, "param1", dummy_tensor, 0)
    
    # Result should be 1.0
    assert torch.allclose(dummy_state["perturb_buffer"], torch.ones_like(dummy_tensor)), \
        "Bias Corrected RMSProp normalization incorrect"

def test_rmsprop_blockwise(dummy_tensor):
    """Test Blockwise RMSProp (reduces stats over parameter blocks/groups)."""
    param_id = "test_block"
    # Basic state setup
    state = {"step": 1, "perturb_buffer": None} 
    
    # Step 1: Input 2.0. MeanSq = 4.0.
    # Coeff 0.5/0.5 for easy math.
    op = chain_adam.OC_Compute_RMSProp_Blockwise(coeff_old=0.5, coeff_new=0.5, force_init_value=-999)
    g_dummy = Genome() 
    
    perturbation = torch.full_like(dummy_tensor, 2.0)
    state["perturb_buffer"] = perturbation.clone()
    
    # Init from 0. Update: 0*0.5 + 4.0*0.5 = 2.0.
    op.apply(g_dummy, state, param_id, dummy_tensor, 0)
    
    val = state[(param_id, "rmsprop_block")]
    assert abs(val - 2.0) < 1e-5, "Blockwise step 1 failed"
    
    # Step 2: Input 4.0. MeanSq = 16.0.
    # Update: 2.0*0.5 + 16.0*0.5 = 1.0 + 8.0 = 9.0.
    state["perturb_buffer"] = torch.full_like(dummy_tensor, 4.0)
    op.apply(g_dummy, state, param_id, dummy_tensor, 0)
    
    val_step2 = state[(param_id, "rmsprop_block")]
    assert abs(val_step2 - 9.0) < 1e-5, "Blockwise step 2 failed"

def test_seeded_rmsprop_equivalence(dummy_tensor, basic_genome):
    """
    Verify that Seeded RMSProp (Stateless) mathematically matches standard Stateful RMSProp.
    Crucial for validating that reloading from seed history is accurate.
    """
    steps = 5
    coeff_old = 0.9
    coeff_new = 0.1
    param_id = "test_param"
    
    # --- A: Stateful Calculation ---
    state_A = {"step": 0, "perturb_buffer": None}
    op_calc = chain_adam.OC_Compute_RMSProp(coeff_old=coeff_old, coeff_new=coeff_new, force_init_value=-999)
    
    history_genomes = []
    
    for i in range(steps):
        g = Genome()
        seed = g.mutate_seed(1.0)
        g.perturb_scales=[1.0]
        history_genomes.append(g)
        
        # Consistent noise
        torch.manual_seed(seed)
        noise = torch.randn_like(dummy_tensor)
        state_A["perturb_buffer"] = noise.clone()
        
        op_calc.apply(g, state_A, param_id, dummy_tensor, 0)
        
    final_rmsprop_A = state_A[(param_id, "rmsprop")]
    dist_decay_coeff = state_A[(param_id, "rmsprop_decay_coeff")]
    # Valid bias correction factor at step T
    bias_corr_A = 1.0 - dist_decay_coeff
    
    # Expected final RMS for comparison: standard_rms / bias_correction
    expected_corrected_rms = final_rmsprop_A / bias_corr_A

    # --- B: Seeded Calculation ---
    state_B = {
        "step": 0, 
        "perturb_buffer": torch.ones_like(dummy_tensor) # Dummy buffer to be normalized
    }
    
    # Populate history structure
    state_B[(param_id, "seed_history")] = []
    for g in history_genomes:
        state_B[(param_id, "seed_history")].append(list(zip(g.seeds, g.perturb_scales)))
        
    op_seeded = chain_adam_seeded.OC_Apply_RMSProp_Seeded(
        coeff_old=coeff_old, coeff_new=coeff_new, force_init_value=-999, epsilon=1e-8
    )
    
    # Run Seeded Op
    # This reconstructs the RMS accumulated from history, 
    # then divides the current buffer (Ones) by sqrt(reconstructed).
    # Result = 1.0 / sqrt(CorrectedRMS)
    op_seeded.apply(basic_genome, state_B, param_id, dummy_tensor, 0)
    result_scaling = state_B["perturb_buffer"]
    
    # Inverse the result to get the RMS value used
    # result = 1/sqrt(RMS) -> RMS = (1/result)^2
    reconstructed_rms = (1.0 / result_scaling).pow(2)
    
    # Compare
    assert torch.allclose(reconstructed_rms, expected_corrected_rms, rtol=1e-4), \
        "Seeded RMSProp reconstruction failed to match Stateful RMSProp logic"

# --- MOMENTUM ---

def test_momentum_accumulation(dummy_tensor, dummy_state, basic_genome):
    """Test standard momentum accumulation (Polyak)."""
    # coeff_new=1.0, coeff_old=0.9 -> m = 0.9*m + 1.0*g
    op = chain_adam.OC_Compute_Momentum(coeff_old=0.9, coeff_new=0.1, force_init_zeros=True)
    
    # Step 1: Input 1.0.
    # m = 0*0.9 + 1.0*0.1 = 0.1? No, logic depends on force_init
    # If force_init_zeros=True, just raw accumulation.
    dummy_state["perturb_buffer"] = torch.full_like(dummy_tensor, 1.0)
    op.apply(basic_genome, dummy_state, "p", dummy_tensor, 0)
    
    assert torch.allclose(dummy_state[("p", "momentum")], torch.full_like(dummy_tensor, 0.1))

# --- MISC / ADVANCED ---

def test_muon_whitening():
    """
    Test Muon Whitening (Newton-Schulz Iteration).
    Ensures it runs on 2D tensors and applies a transformation.
    """
    state = {"step": 1, "perturb_buffer": None}
    
    # 2D Tensor required
    tensor_2d = torch.randn(20, 20)
    p = torch.randn(20, 20)
    state["perturb_buffer"] = p.clone()
    
    op = chain_misc.OC_Muon_Whiten_Perturb_Buffer(force_bf16=False)
    
    # Run
    op.apply(Genome(), state, "p", tensor_2d, 0)
    
    whitened = state["perturb_buffer"]
    
    assert whitened.shape == (20, 20)
    assert not torch.allclose(whitened, p), "Muon whitening failed to change tensor"
    
    # Test 1D Safety (Should assert or skip)
    # The implementation asserts ndim == 2
    tensor_1d = torch.randn(20)
    state["perturb_buffer"] = tensor_1d
    
    # We must call internal method to verify assertion, apply likely checks dimensions first
    with pytest.raises(AssertionError):
        op.newtonschulz5(tensor_1d)

def test_seed_history_truncation(dummy_tensor):
    """Test that seed history is correctly truncated to max_steps."""
    param_id = "test"
    state = {"step": 0}
    max_steps = 3
    op = chain_adam_seeded.OC_Update_Seed_History(max_steps=max_steps)
    
    # Simulate 5 steps
    for i in range(5):
        g = Genome()
        g.seeds = [i]
        g.perturb_scales = [1.0]
        # OC_Update... appends current genome seeds to history
        op.apply(g, state, param_id, dummy_tensor, 0)
        
    history = state[(param_id, "seed_history")]
    
    assert len(history) == max_steps, f"History length {len(history)} != {max_steps}"
    
    # Check content (FIFO). History should contain steps 2, 3, 4.
    # Structure: [[(seed, scale)], [(seed, scale)]...]
    oldest_seed = history[0][0][0]
    newest_seed = history[-1][0][0]
    
    assert oldest_seed == 2
    assert newest_seed == 4
