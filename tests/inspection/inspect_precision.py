import torch
import torch.nn.functional as F
from propagate.optimizers import chain
from propagate.optimizers import chain_adam
from propagate.optimizers import chain_misc
from propagate.genome import Genome
from propagate.optimizers.optimizer import Optimizer
import pytest
import math
import random

"""
Inspection: BF16 vs FP32 Precision Analysis
This script charts the numerical divergence between Brain Float 16 (BF16) and Float 32 (FP32)
accumulators in evolution strategies by using the ACTUAL optimizer chain components.
"""

DIM = 4096

def test_independent_perturbation_eval():
    """Test standard Gaussian perturbation generation (Eval phase)"""
    # This checks if summing seeds in FP32 vs BF16 diverges.
    # Note: The chain.Init_Perturbation_Gaussian has an fp32_accumulate flag.
    
    random.seed(42)
    genome = Genome()
    # Simulate a genome with multiple seeds
    for _ in range(10): 
        genome.mutate_seed(1.0)
    
    # FP32 Accumulation
    op_fp32_acc = chain.Init_Perturbation_Gaussian(fp32_accumulate=True)
    # BF16 Accumulation (Simulated by disabling fp32_accumulate if the op supports it, 
    # but Init_Perturbation_Gaussian defaults to float32 accumulator usually unless we change it?)
    # Looking at code: `perturbation = torch.zeros_like(tensor, dtype = torch.float32 if self.fp32_accumulate else tensor.dtype)`
    op_bf16_acc = chain.Init_Perturbation_Gaussian(fp32_accumulate=False) 
    
    # We want to see if accumulating in BF16 is bad compared to FP32.
    
    print("\n>> Running Test: Gradient Accumulation (Seed Summing)")
    
    w_bf16 = torch.zeros(DIM, dtype=torch.bfloat16) # Always use BF16 template for noise generation

    state = {}
    
    # Run FP32 Accumulation (using BF16 noise source)
    # Passed w_bf16 so noise is generated in BF16, but accumulates in FP32 (due to flag)
    op_fp32_acc.apply(genome, state, "p", w_bf16, 0)
    res_fp32_acc = state["perturb_buffer"].clone()
    del state["perturb_buffer"]

    # Run BF16 Accumulation (using BF16 noise source)
    op_bf16_acc.apply(genome, state, "p", w_bf16, 0) 
    res_bf16_acc = state["perturb_buffer"].clone()
    
    diff_tensor = (res_fp32_acc - res_bf16_acc.float()).abs()
    diff = diff_tensor.mean().item()
    base_mean_fp32 = res_fp32_acc.abs().mean().item()
    base_mean_bf16 = res_bf16_acc.float().abs().mean().item()
    print(f"   Divergence (Acc FP32 vs Acc BF16) with 10 seeds: {diff:.8f} (FP32 Mean: {base_mean_fp32:.8f} | BF16 Mean: {base_mean_bf16:.8f})")

def test_bernoulli_accumulation():
    # Comparing FP32 accumulation vs BF16 accumulation of Bernoulli noise (generated in BF16)
    # The Chain `Init_Perturbation_Bernoulli` also has `fp32_accumulate`
    print("\n>> Running Test: Bernoulli Noise Accumulation")
    
    random.seed(42)
    g = Genome()
    # Add multiple seeds to ensure accumulation happens and we test the drift.
    # We use non-integer weights to force rounding errors in BF16.
    for _ in range(20):
        g.mutate_seed(1.2345) 
    
    op_fp32_acc = chain.Init_Perturbation_Bernoulli(fp32_accumulate=True)
    op_bf16_acc = chain.Init_Perturbation_Bernoulli(fp32_accumulate=False)
    
    w_bf16 = torch.zeros(DIM, dtype=torch.bfloat16) # Always use BF16 template for noise generation
    
    state = {}
    
    # Sync by same random offset/seed implied by genome? 
    # The seeds are in the genome. The generator uses manual_seed(seed + offset).
    
    # 1. FP32 Accumulation (but noise source is BF16 based on template)
    op_fp32_acc.apply(g, state, "p", w_bf16, 42)
    res32 = state["perturb_buffer"].clone()
    del state["perturb_buffer"]
    
    # 2. BF16 Accumulation
    op_bf16_acc.apply(g, state, "p", w_bf16, 42)
    res16 = state["perturb_buffer"].clone()
    
    diff = (res32 - res16.float()).abs()
    print_stats(0, diff, "Bernoulli Acc Output", res32, res16)
    print(f"   Divergence Bernoulli (Acc FP32 vs Acc BF16) over 20 seeds: {diff.mean().item():.8f}")
        
def print_stats(step, diff_tensor, msg="", base_tensor_fp32=None, base_tensor_bf16=None):
    mean = diff_tensor.mean().item()
    max_val = diff_tensor.max().item()
    std = diff_tensor.std().item()
    
    if torch.isnan(diff_tensor).any() or torch.isinf(diff_tensor).any():
        print(f"   [Step {step}] NaN/Inf detected! {msg}")
    
    base_info = ""
    if base_tensor_fp32 is not None and base_tensor_bf16 is not None:
        base_mean_fp32 = base_tensor_fp32.abs().mean().item()
        base_mean_bf16 = base_tensor_bf16.float().abs().mean().item()
        base_info = f" | FP32 Mean: {base_mean_fp32:.8f} | BF16 Mean: {base_mean_bf16:.8f}"
    elif base_tensor_fp32 is not None:
        base_mean_fp32 = base_tensor_fp32.abs().mean().item()
        base_info = f" | FP32 Mean: {base_mean_fp32:.8f}"
        
    print(f"   {msg} Mean Div: {mean:.8f} | Max Div: {max_val:.8f} | Std: {std:.8f}{base_info}")
    return mean

def test_momentum_divergence():
    print("\n>> Running Test: Momentum State Divergence")
    
    # We need to check internal state, similar to RMSProp
    state_fp32 = {}
    state_bf16 = {}
    w_fp32 = torch.zeros(DIM, dtype=torch.float32)
    w_bf16 = torch.zeros(DIM, dtype=torch.bfloat16)
    
    g = Genome()
    ops = chain_adam.OC_Compute_Momentum(coeff_old=0.9, coeff_new=0.1)
    
    errors = []
    
    for i in range(100):
         # Generate synchronized noise (BF16 source)
        gen = torch.Generator()
        gen.manual_seed(i)
        p16 = torch.randn(DIM, generator=gen, dtype=torch.bfloat16)
        p32 = p16.float()
        
        state_fp32["perturb_buffer"] = p32.clone()
        state_bf16["perturb_buffer"] = p16.clone()
        
        ops.apply(g, state_fp32, "p", w_fp32, 0)
        ops.apply(g, state_bf16, "p", w_bf16, 0)
        
        m32 = state_fp32.get(("p", "momentum"))
        m16 = state_bf16.get(("p", "momentum"))
        
        if m32 is None or m16 is None:
             continue
             
        diff = (m32 - m16.float()).abs()
        mean_diff = diff.mean().item()
        errors.append(mean_diff)
        
        if i % 20 == 0:
            print_stats(i, diff, f"Step {i:3d}", m32, m16)

    # Final stats
    if m32 is not None and m16 is not None:
        diff = (m32 - m16.float()).abs()
        print_stats(99, diff, "Final Stats ->", m32, m16)

    print(f"   Final Divergence Momentum State: {errors[-1]:.8f}")

def test_rmsprop_divergence():
    ops = chain_adam.OC_Compute_RMSProp(coeff_old=0.99, coeff_new=0.01)
    # Checks the rmsprop buffer (variance estimate)
    
    # We need a custom runner to check the internal state 'rmsprop' instead of perturb buffer
    print("\n>> Running Test: RMSProp State Divergence")
    
    state_fp32 = {}
    state_bf16 = {}
    w_fp32 = torch.zeros(DIM, dtype=torch.float32)
    w_bf16 = torch.zeros(DIM, dtype=torch.bfloat16)

    g = Genome()
    
    ops = chain_adam.OC_Compute_RMSProp(coeff_old=0.99, coeff_new=0.01, force_init_value=-999)

    for i in range(200):
        # Noise
        gen = torch.Generator()
        gen.manual_seed(i)
        p16 = torch.randn(DIM, generator=gen, dtype=torch.bfloat16)
        p32 = p16.float()
        
        # Simulate gradient, sometimes large
        if i % 10 == 0: 
            p32 *= 10.0
            p16 *= 10.0
        
        state_fp32["perturb_buffer"] = p32.clone()
        state_bf16["perturb_buffer"] = p16.clone()
        
        ops.apply(g, state_fp32, "p", w_fp32, 0)
        ops.apply(g, state_bf16, "p", w_bf16, 0)
        
    v32 = state_fp32.get(("p", "rmsprop"))
    v16 = state_bf16.get(("p", "rmsprop"))
    
    if v32 is None or v16 is None:
        print("   [ERROR] RMSProp state not initialized!")
        return

    diff = (v32 - v16.float()).abs()
    print_stats(199, diff, "Final RMSProp State ->", v32, v16)
    print(f"   Final Divergence RMSProp State: {diff.mean().item():.8f}")
    
    # Test Apply RMSProp (Normalization)
    print("   Testing Apply RMSProp (Division)...")
    apply_ops = chain_adam.OC_Apply_RMSProp()
    apply_ops.apply(g, state_fp32, "p", w_fp32, 0)
    apply_ops.apply(g, state_bf16, "p", w_bf16, 0)
    
    res32 = state_fp32["perturb_buffer"]
    res16 = state_bf16["perturb_buffer"]
    
    if res32 is None or res16 is None:
         print("   [ERROR] Perturbation buffer missing!")
         return

    diff_norm = (res32 - res16.float()).abs()
    print_stats(199, diff_norm, "Final Apply RMSProp ->", res32, res16)
    print(f"   Final Divergence Apply RMSProp: {diff_norm.mean().item():.8f}")

def test_muon_divergence():
    # Test Muon Whitening precision loss
    # Compare performing whitening in BF16 (forced) vs pure FP32
    
    print("\n>> Running Test: Muon Whitening (Newton-Schulz)")
    dim_sq = 64
    
    # Generate random FP32 noise
    torch.manual_seed(42)
    p32 = torch.randn((dim_sq, dim_sq), dtype=torch.float32)
    
    # Case 1: Force BF16 (Standard behavior for memory/speed)
    ops_force = chain_misc.OC_Muon_Whiten_Perturb_Buffer(force_bf16=True)
    state_force = {"perturb_buffer": p32.clone()}
    # Note: Tensor argument is only for device/dtype inference if needed, but Muon only looks at buffer usually?
    # Checking chain_misc.py: newtonschulz5 uses G.copy_ so it modifies buffer. 
    # It casts X to bfloat16 if force_bf16 is True.
    ops_force.apply(Genome(), state_force, "p", p32, 0)
    res_force = state_force["perturb_buffer"]
    
    # Case 2: No Force (Full FP32 Precision)
    ops_full = chain_misc.OC_Muon_Whiten_Perturb_Buffer(force_bf16=False)
    state_full = {"perturb_buffer": p32.clone()}
    ops_full.apply(Genome(), state_full, "p", p32, 0)
    res_full = state_full["perturb_buffer"]
    
    diff = (res_force - res_full).abs()
    print_stats(0, diff, "Muon Force BF16 vs Pure FP32", res_full, res_force)
    print(f"   Divergence (Impact of BF16 optimization): {diff.mean().item():.8f}")

def test_full_optimizer_divergence():
    print("\n>> Running Test: Full Optimization Path (Adam w/ BF16 vs FP32)")
    
    # 1. Setup Chains for Standard Adam-like ES
    # Logic: 
    #   Calculate Momentum & RMSProp statistics from current gradient (perturbation).
    #   Replace perturbation with Momentum.
    #   Scale by RMSProp.
    #   Add to Model.
    
    chain_seq = [
        # Note: Input is already in perturb_buffer (Noise)
        chain_adam.OC_Compute_Momentum(coeff_old=0.9, coeff_new=0.1),
        chain_adam.OC_Compute_RMSProp(coeff_old=0.999, coeff_new=0.001, force_init_value=-999),
        chain.Zero_Perturb_Buffer(), # Clear noise
        chain_adam.OC_Add_Momentum(), # Add momentum (bias corrected)
        chain_adam.OC_Apply_RMSProp(), # Scale by RMSProp
        chain.Add_Perturb_Buffer() # Update Weights
    ]
    
    # We can't reuse valid references across fp32/bf16 easily if we just pass list of objects
    # providing they don't have internal state (they don't, state is in dict).
    # But for safety in python, list logic is fine.
    
    # We use a custom runner here because we want to track Weight Divergence.
    
    state_fp32 = {}
    state_bf16 = {}
    w_fp32 = torch.zeros(DIM, dtype=torch.float32)
    w_bf16 = torch.zeros(DIM, dtype=torch.bfloat16) # Weights in BF16
    
    # Setup States
    g = Genome()
    
    # Since we are using a list of chains, we need an Apply wrapper or loop
    def apply_chain(state, tensor, seed):
        for op in chain_seq:
            op.apply(g, state, "p", tensor, seed)

    errors = []

    for i in range(100):
        # 1. Generate Noise (BF16 Source)
        gen = torch.Generator()
        gen.manual_seed(i) # Sync seed
        p16 = torch.randn(DIM, generator=gen, dtype=torch.bfloat16)
        p32 = p16.float()
        
        state_fp32["perturb_buffer"] = p32.clone()
        state_bf16["perturb_buffer"] = p16.clone()
        
        # 2. Apply Full Optimizer Chain
        apply_chain(state_fp32, w_fp32, 0)
        apply_chain(state_bf16, w_bf16, 0)
        
        # 3. Measure Weight Divergence
        diff = (w_fp32 - w_bf16.float()).abs()
        mean_diff = diff.mean().item()
        errors.append(mean_diff)
        
        if i % 20 == 0:
            print_stats(i, diff, f"Step {i:3d} Weights", w_fp32, w_bf16)

    diff = (w_fp32 - w_bf16.float()).abs()
    print_stats(99, diff, "Final Weights ->", w_fp32, w_bf16)
    print(f"   Final Divergence Weights: {errors[-1]:.8f}")

def main():
    test_independent_perturbation_eval()
    test_bernoulli_accumulation()
    test_momentum_divergence()
    test_rmsprop_divergence()
    test_muon_divergence()
    test_full_optimizer_divergence()
    
if __name__ == "__main__":
    main()
