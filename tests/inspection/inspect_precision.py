import torch
import torch.nn.functional as F
from propagate.optimizers import chain
from propagate.optimizers import chain_adam
from propagate.optimizers import chain_misc
from propagate.optimizers.psamplers import Gaussian_PSampler, Bernoulli_PSampler
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
    random.seed(42)
    genome = Genome()
    for _ in range(10): 
        genome.mutate_seed(1.0)
    
    op_fp32_acc = chain.Init_Perturbation(Gaussian_PSampler(fp32_accumulate=True))
    op_bf16_acc = chain.Init_Perturbation(Gaussian_PSampler(fp32_accumulate=False)) 
        
    print("\n>> Running Test: Gradient Accumulation (Seed Summing)")
    
    w_bf16 = torch.zeros(DIM, dtype=torch.bfloat16)

    state = {}

    op_fp32_acc.apply(genome, state, "p", w_bf16, 0)
    res_fp32_acc = state["perturb_buffer"].clone()
    del state["perturb_buffer"]

    op_bf16_acc.apply(genome, state, "p", w_bf16, 0) 
    res_bf16_acc = state["perturb_buffer"].clone()
    
    diff_tensor = (res_fp32_acc - res_bf16_acc.float()).abs()
    diff = diff_tensor.mean().item()
    base_mean_fp32 = res_fp32_acc.abs().mean().item()
    base_mean_bf16 = res_bf16_acc.float().abs().mean().item()
    print(f"   Divergence (Acc FP32 vs Acc BF16) with 10 seeds: {diff:.8f} (FP32 Mean: {base_mean_fp32:.8f} | BF16 Mean: {base_mean_bf16:.8f})")

def test_bernoulli_accumulation():
    print("\n>> Running Test: Bernoulli Noise Accumulation")
    
    random.seed(42)
    g = Genome()
    for _ in range(20):
        g.mutate_seed(1.2345) 
    
    op_fp32_acc = chain.Init_Perturbation(Bernoulli_PSampler(fp32_accumulate=True))
    op_bf16_acc = chain.Init_Perturbation(Bernoulli_PSampler(fp32_accumulate=False))
    
    w_bf16 = torch.zeros(DIM, dtype=torch.bfloat16)
    
    state = {}
    
    op_fp32_acc.apply(g, state, "p", w_bf16, 42)
    res32 = state["perturb_buffer"].clone()
    del state["perturb_buffer"]
    
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
    
    state_fp32 = {}
    state_bf16 = {}
    w_fp32 = torch.zeros(DIM, dtype=torch.float32)
    w_bf16 = torch.zeros(DIM, dtype=torch.bfloat16)
    
    g = Genome()
    ops = chain_adam.OC_Compute_Momentum(coeff_old=0.9, coeff_new=0.1)
    
    errors = []
    
    for i in range(100):
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

    print("\n>> Running Test: RMSProp State Divergence")
    
    state_fp32 = {}
    state_bf16 = {}
    w_fp32 = torch.zeros(DIM, dtype=torch.float32)
    w_bf16 = torch.zeros(DIM, dtype=torch.bfloat16)

    g = Genome()
    
    ops = chain_adam.OC_Compute_RMSProp(coeff_old=0.99, coeff_new=0.01, force_init_value=-999)

    for i in range(200):
        gen = torch.Generator()
        gen.manual_seed(i)
        p16 = torch.randn(DIM, generator=gen, dtype=torch.bfloat16)
        p32 = p16.float()
        
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
    print("\n>> Running Test: Muon Whitening (Newton-Schulz)")
    dim_sq = 64
    
    torch.manual_seed(42)
    p32 = torch.randn((dim_sq, dim_sq), dtype=torch.float32)
    
    ops_force = chain_misc.OC_Muon_Whiten_Perturb_Buffer(force_bf16=True)
    state_force = {"perturb_buffer": p32.clone()}
    ops_force.apply(Genome(), state_force, "p", p32, 0)
    res_force = state_force["perturb_buffer"]
    
    ops_full = chain_misc.OC_Muon_Whiten_Perturb_Buffer(force_bf16=False)
    state_full = {"perturb_buffer": p32.clone()}
    ops_full.apply(Genome(), state_full, "p", p32, 0)
    res_full = state_full["perturb_buffer"]
    
    diff = (res_force - res_full).abs()
    print_stats(0, diff, "Muon Force BF16 vs Pure FP32", res_full, res_force)
    print(f"   Divergence (Impact of BF16 optimization): {diff.mean().item():.8f}")

def test_full_optimizer_divergence():
    print("\n>> Running Test: Full Optimization Path (Adam w/ BF16 vs FP32)")
    
    chain_seq = [
        chain_adam.OC_Compute_Momentum(coeff_old=0.9, coeff_new=0.1),
        chain_adam.OC_Compute_RMSProp(coeff_old=0.999, coeff_new=0.001, force_init_value=-999),
        chain.Zero_Perturb_Buffer(),
        chain_adam.OC_Add_Momentum(),
        chain_adam.OC_Apply_RMSProp(),
        chain.Add_Perturb_Buffer()
    ]
    
    state_fp32 = {}
    state_bf16 = {}
    w_fp32 = torch.zeros(DIM, dtype=torch.float32)
    w_bf16 = torch.zeros(DIM, dtype=torch.bfloat16)
    
    g = Genome()
    
    def apply_chain(state, tensor, seed):
        for op in chain_seq:
            op.apply(g, state, "p", tensor, seed)

    errors = []

    for i in range(100):
        gen = torch.Generator()
        gen.manual_seed(i)
        p16 = torch.randn(DIM, generator=gen, dtype=torch.bfloat16)
        p32 = p16.float()
        
        state_fp32["perturb_buffer"] = p32.clone()
        state_bf16["perturb_buffer"] = p16.clone()
        
        apply_chain(state_fp32, w_fp32, 0)
        apply_chain(state_bf16, w_bf16, 0)
        
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
