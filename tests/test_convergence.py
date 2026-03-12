import pytest
import torch
from propagate.optimizers.optimizer import Optimizer
from propagate.optimizers import chain, chain_adam, chain_log

def test_convergence_quadratic(dummy_tensor):
    """
    Test convergence on a simple quadratic problem x^2.
    Optimizer should drive x from [1,1,1...] to [0,0,0...]
    """
    
    # 1. Setup the Problem
    target = torch.zeros(10, dtype=torch.float32)
    current_weights = torch.ones(10, dtype=torch.float32) * 5.0 # Start far away
    
    def calculate_reward(weights):
        # Reward = - MSE
        mse = torch.mean((weights - target)**2)
        return -mse.item()
    
    # 2. Setup Optimizer (Simple SGD)
    # Pipeline:
    # Perturb: Gaussian -> Scale
    # Update: Scale(mul_by_lr) -> Add to Buffer
    
    perturb_chain = [
        chain.Init_Perturbation_Gaussian(),
        chain.Scale_Perturbation(mul_by_std=True),
        chain.Add_Perturb_Buffer(), 
        chain.Delete_Perturb_Buffer()
    ]
    
    update_chain = [
        chain.Init_Perturbation_Gaussian(),
        chain.Scale_Perturbation(mul_by_lr=True, div_by_pop=True),
        chain.Add_Perturb_Buffer(),
        chain.Delete_Perturb_Buffer(),
    ]
    
    opt = Optimizer(
        optimizer_name="test_sgd",
        total_steps=50,
        learning_rate=1.0, # High LR for fast convergence
        perturb_scale=0.1,
        population_size=20,
        mirror=True,
        perturb_chain=perturb_chain,
        update_chain=update_chain,
        norm_by_mean=True,
        rank_norm_rewards=False
    )
    
    # 3. Training Loop simulation without Backend
    
    initial_loss = calculate_reward(current_weights)
    
    # Re-check Optimizer logic in `update_self`:
    # It calculates `perturb_scales` for the representative genome.
    # Does it scale them by something small?
    # `grad_scale = (reward - reward_mean) / (self.last_rstd + 1e-8)` (if normalized? No, code says:
    # self.last_rstd = ...
    # if self.norm_by_mean: grad_scale = reward - reward_mean.
    # It does NOT divide by std unless implemented elsewhere?
    # Code review of `optimizer.py` (not visible here but usually standard ES doesn't divide by std unless specified).
    
    # Setup for fix: Increase steps to 100 for more convergence on random problem
    
    for step in range(100):
        # Create population
        # Simulate simple backend (stateless for now, recreating genomes like trainer)
        from propagate.genome import Genome
        population = [Genome() for _ in range(opt.population_size)] # Half population
        for g in population: 
            g.mutate_seed(1.0) # Add fresh seed
        
        # Mirroring (manual as trainer does it)
        mirrored = [g.get_mirrored() for g in population]
        full_pop = population + mirrored
        
        # Evaluate
        for g in full_pop:
            # We need to manually simulate the backend perturbation
            # 1. Create a temporary weight copy
            temp_weights = current_weights.clone()
            state = {"step": step, "lr": 1.0, "std": 0.1, "rstd": 1.0, "population_size": 40, "lr_scalar": 1.0}
            
            # Apply Perturbation
            opt.apply_perturb(invert=False, genome=g, tensor=temp_weights, random_offset=0, parameter_id="test", state=state)
            
            # evaluate
            reward = calculate_reward(temp_weights)
            
            g.historical_rewards = [reward]
            g.latest_inputs = ["."]
            g.latest_outputs = ["."]
            g.latest_rewards = [reward]
            
            # Cleanup perturbation buffer from state if failed 
            if "perturb_buffer" in state:
                del state["perturb_buffer"]

        # Update
        opt.update_self(full_pop, step)
        
        # Apply Update to main weights
        state = {"step": step, "lr": 1.0, "std": 0.1, "rstd": 1.0, "population_size": 40, "lr_scalar": 1.0}
        opt.apply_grad(current_weights, random_offset=0, parameter_id="test", state=state)
        
        # Cleanup
        if "perturb_buffer" in state:
            del state["perturb_buffer"]
            
    final_loss = calculate_reward(current_weights)
    
    print(f"Initial: {initial_loss}, Final: {final_loss}")
    
    # Check Improvement
    assert final_loss > initial_loss
    assert final_loss > -0.5 # Should be close to 0 (reward of perfect match)
