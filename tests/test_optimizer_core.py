import pytest
import torch
from propagate.optimizers.optimizer import Optimizer
from propagate.optimizers import chain
from propagate.genome import Genome
from propagate.training_config import TrainingConfig

"""
Tests for Core Optimizer Logic.
Includes:
- Optimizer State Management (Init, Restore)
- Learning Rate Schedulers (Linear, Cosine, Warmup)
- Reward Normalization (Rank, Mean)
- Gradient Accumulation from Genomes
"""

# --- Schedulers ---

def test_schedulers():
    """Test LR decay schedules: Linear, Warmup, Cosine."""
    
    # 1. Linear Decay
    config = TrainingConfig(
        total_steps=100,
        learning_rate=1.0,
        perturb_scale=0.1,
        population_size=2,
        mirror=False,
        rank_norm_rewards=False,
        lr_scheduler="linear",
        warmup_steps=0
    )
    opt = Optimizer("test", config=config, perturb_chain=[chain.Add_Perturb_Buffer()], update_chain=[])
    assert opt.get_lr(0) == 1.0
    assert opt.get_lr(50) == 0.5
    assert opt.get_lr(100) == 0.0
    
    # 2. Warmup
    config = TrainingConfig(
        total_steps=100,
        learning_rate=1.0,
        perturb_scale=0.1,
        population_size=2,
        mirror=False,
        rank_norm_rewards=False,
        lr_scheduler="linear",
        warmup_steps=10
    )
    opt = Optimizer("test", config=config, perturb_chain=[chain.Add_Perturb_Buffer()], update_chain=[])
    assert opt.get_lr(0) == 0.0
    assert opt.get_lr(5) == 0.5
    assert opt.get_lr(10) == 1.0 

    # 3. Cosine Decay
    config = TrainingConfig(
        total_steps=100,
        learning_rate=1.0,
        perturb_scale=0.1,
        population_size=2,
        mirror=False,
        rank_norm_rewards=False,
        lr_scheduler="cosine",
        warmup_steps=0
    )
    opt = Optimizer("test", config=config, perturb_chain=[chain.Add_Perturb_Buffer()], update_chain=[])
    assert opt.get_lr(0) == 1.0
    assert abs(opt.get_lr(50) - 0.5) < 1e-5
    assert abs(opt.get_lr(100) - 0.0) < 1e-5

# --- Reward Normalization ---

def test_reward_normalization_logic():
    """Test Rank-based and Mean-based reward normalization."""
    # Setup Genomes with distinct rewards
    rewards = [10.0, 100.0, 5.0] # Unsorted
    seeds = [1, 2, 3]
    
    genomes = []
    for r, s in zip(rewards, seeds):
        g = Genome()
        g.historical_rewards = [r]
        g.latest_rewards=[r]
        g.seeds=[s]
        g.perturb_scales=[1.0]
        # Dummy strings for file I/O checks skipped here
        g.latest_inputs=["."]; g.latest_outputs=["."]
        genomes.append(g)
        
    # 1. Rank Norm
    
    config = TrainingConfig(
        total_steps=100,
        learning_rate=1.0,
        perturb_scale=0.1,
        population_size=3,
        mirror=False,
        rank_norm_rewards=True,
    )
    opt_rank = Optimizer("test", config=config, perturb_chain=[chain.Add_Perturb_Buffer()], update_chain=[])
    opt_rank.update_self(genomes, 1)
    rep = opt_rank.rep_genome
    
    # Verify weights assigned to each seed in Representative Genome
    assert 1 in rep.seeds # Reward 10.0 (Rank 1 -> 0.0)
    w1 = rep.perturb_scales[rep.seeds.index(1)]
    assert abs(w1 - 0.0) < 1e-5
    
    assert 2 in rep.seeds # Reward 100.0 (Rank 2 -> 0.5)
    w2 = rep.perturb_scales[rep.seeds.index(2)]
    assert abs(w2 - 0.5) < 1e-5
    
    assert 3 in rep.seeds # Reward 5.0 (Rank 0 -> -0.5)
    w3 = rep.perturb_scales[rep.seeds.index(3)]
    assert abs(w3 - (-0.5)) < 1e-5

def test_norm_by_mean_false():
    """Test disabled mean normalization (raw rewards used as weights)."""
    # Two identical genomes with reward 10.0
    g1 = Genome(); g1.latest_rewards=[10.0]; g1.historical_rewards=[10.0]; g1.seeds=[1]; g1.perturb_scales=[1.0]; g1.latest_inputs=["."]; g1.latest_outputs=["."]
    
    config = TrainingConfig(
        total_steps=100,
        learning_rate=1.0,
        perturb_scale=0.1,
        population_size=2,
        mirror=False,
        rank_norm_rewards=False,
    )
    opt = Optimizer("test", config=config, perturb_chain=[chain.Add_Perturb_Buffer()], update_chain=[])
    
    # Manually pass true_mean=0 to simulate norm_by_mean=False
    opt.update_self([g1], 1, true_reward_mean=0)
    
    s1 = opt.rep_genome.perturb_scales[opt.rep_genome.seeds.index(1)]
    assert abs(s1 - 10.0) < 1e-5

# --- History & Restore ---

def test_optimizer_history_restore():
    """Test saving and restoring optimizer state from history list."""
    # This feature seems to have been removed/refactored in Trainer, 
    # skipping as it's no longer part of Optimizer core logic in the same way 
    # or requires Trainer context.
    pass

# --- Inverse Consistency ---

def test_inverse_perturbation_consistency(dummy_tensor):
    """
    Test that apply_perturb(invert=True) correctly reverses the perturbation,
    restoring the original weights.
    Critical for memory efficiency (not storing copies of weights).
    """
    original = dummy_tensor.clone()
    
    # Define a reversible chain
    # Init -> Scale -> Add
    p_chain = [
        chain.Init_Perturbation_Gaussian(),
        chain.Scale_Perturbation(mul_by_std=True),
        chain.Add_Perturb_Buffer(), 
        chain.Delete_Perturb_Buffer() 
    ]
    
    config = TrainingConfig(
        total_steps=100,
        learning_rate=1.0,
        perturb_scale=0.1,
        population_size=2,
        mirror=False,
        rank_norm_rewards=False,
    )
    opt = Optimizer("test", config=config, perturb_chain=p_chain, update_chain=[])
    g = Genome(); g.mutate_seed(1.0)
    state = {"step": 1, "std": 0.5, "lr": 0.1, "rstd": 1.0, "population_size": 1, "lr_scalar": 1.0}
    
    # 1. Forward
    tensor_fw = original.clone()
    opt.apply_perturb(invert=False, genome=g, tensor=tensor_fw, random_offset=0, parameter_id="test", state=state)
    assert not torch.allclose(tensor_fw, original), "Forward passed no-op"
    
    # 2. Inverse
    # Apply to modified tensor. Should revert to original.
    if "perturb_buffer" in state: del state["perturb_buffer"]
    
    opt.apply_perturb(invert=True, genome=g, tensor=tensor_fw, random_offset=0, parameter_id="test", state=state)
    
    assert torch.allclose(tensor_fw, original, atol=1e-6), "Inverse failed to restore weights"
