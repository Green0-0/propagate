import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from propagate.genome import Genome
from propagate.optimizers.optimizer import Optimizer
from propagate.optimizers import chain, chain_adam, chain_misc

"""
Inspection: Convergence on 2D Reward Landscape
Visualizes the path of the optimizer on a 2D surface (e.g., Rosenbrock or Rastrigin function).
Saves plots to tests/inspection/output/
"""

OUTPUT_DIR = "/home/user/Documents/Data/Coding/Python/AI/propagate/tests/inspection/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def ackley(x, y):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + np.e + 20

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def get_reward_grid(func, value_range=(-5.0, 5.0), res=250):
    x = np.linspace(value_range[0], value_range[1], res)
    y = np.linspace(value_range[0], value_range[1], res)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(func)(X, Y)
    return X, Y, Z

def run_optimizer_path(opt_name, setup_func, reward_func, steps=50, start_pos=(-1.5, -1.0), value_range=(-5.0, 5.0)):
    print(f"Simulating {opt_name}...")
    
    # Init Weights (2 params)
    w = torch.tensor(list(start_pos), dtype=torch.float32)
    
    path_x = [w[0].item()]
    path_y = [w[1].item()]
    perturbations_x = []
    perturbations_y = []
    
    opt = setup_func()
    
    # Persistent state dict — must survive across steps so that Momentum/RMSProp/Adam buffers accumulate
    state = {}
    
    for i in range(steps):
        # 1. Population
        pop_size = opt.population_size
        pop = [Genome() for _ in range(pop_size)]
        full_pop = []
        
        # Need distinct seeds for plotting perturbations
        for j, g in enumerate(pop):
            g.mutate_seed(1.0) # Adds random seed
            full_pop.append(g)
            if opt.mirror:
                full_pop.append(g.get_mirrored())
                
        # 2. Evaluate
        step_rewards = []
        
        for g in full_pop:
            w_pert = w.clone()
            
            # Apply Perturbation (apply_perturb overwrites step/lr/std/rstd/population_size/lr_scalar)
            opt.apply_perturb(invert=False, genome=g, tensor=w_pert, random_offset=0, parameter_id="test", state=state)
            
            # Calculate loss (maximize reward = minimize function)
            if w_pert[0].item() < value_range[0] or w_pert[0].item() > value_range[1] or w_pert[1].item() < value_range[0] or w_pert[1].item() > value_range[1]:
                 loss = 999999 + abs(w_pert[0].item()) * 100 + abs(w_pert[1].item()) * 100
                 r = -loss
            else:
                 loss = reward_func(w_pert[0].item(), w_pert[1].item())
                 r = -loss
                 
            # Record perturbation point for visualization
            if i % 3 == 0 and loss < 99999: # Downsample points, only plot valid ones
                perturbations_x.append(w_pert[0].item())
                perturbations_y.append(w_pert[1].item())
            
            g.historical_rewards = [r]
            g.latest_rewards = [r]
            g.latest_inputs=["."]; g.latest_outputs=["."]
            
            # Clean buffer
            if "perturb_buffer" in state: del state["perturb_buffer"]
            
        # 3. Update
        opt.update_self(full_pop, i+1)
        
        # apply_grad overwrites step/lr/std/rstd/population_size/lr_scalar from optimizer state
        opt.apply_grad(w, random_offset=0, parameter_id="test", state=state)
        
        path_x.append(w[0].item())
        path_y.append(w[1].item())
        
    # Calculate Distances
    step_distances = [np.sqrt((path_x[k] - path_x[k-1])**2 + (path_y[k] - path_y[k-1])**2) for k in range(1, len(path_x))]
    total_distance = sum(step_distances)
    mean_step = total_distance / steps if steps > 0 else 0
        
    return path_x, path_y, perturbations_x, perturbations_y, total_distance, mean_step

def chart_convergence():
    # 1. Simple SGD Mirrored
    def setup_sgd():
        p_chain = [chain.Init_Perturbation_Gaussian(), chain.Scale_Perturbation(mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        u_chain = [chain.Init_Perturbation_Gaussian(), chain.Scale_Perturbation(mul_by_lr=True, div_by_pop=True, mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        return Optimizer("SGD_Mir", 150, 0.04, 0.5, 40, True, p_chain, u_chain, False, False)

    def setup_sgd_nomirror():
        p_chain = [chain.Init_Perturbation_Gaussian(), chain.Scale_Perturbation(mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        u_chain = [chain.Init_Perturbation_Gaussian(), chain.Scale_Perturbation(mul_by_lr=True, div_by_pop=True, mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        return Optimizer("SGD_NoMir", 150, 0.04, 0.5, 40, False, p_chain, u_chain, True, False)

    def setup_sgd_rank():
        p_chain = [chain.Init_Perturbation_Gaussian(), chain.Scale_Perturbation(mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        u_chain = [chain.Init_Perturbation_Gaussian(), chain.Scale_Perturbation(mul_by_lr=True, div_by_pop=True, mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        return Optimizer("SGD_Rank", 150, 2.0, 0.5, 40, True, p_chain, u_chain, False, True)

    # 2. SGD with Momentum
    def setup_momentum():
        p_chain = [chain.Init_Perturbation_Gaussian(), chain.Scale_Perturbation(mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        u_chain = [
            chain.Init_Perturbation_Gaussian(), 
            chain.Scale_Perturbation(div_by_pop=True, mul_by_std=True), 
            chain_adam.OC_Compute_Momentum(0.9, 0.1),
            chain.Zero_Perturb_Buffer(),
            chain_adam.OC_Add_Momentum(),
            chain.Scale_Perturbation(mul_by_lr=True), 
            chain.Add_Perturb_Buffer(), 
            chain.Delete_Perturb_Buffer()
        ]
        return Optimizer("Momentum", 150, 0.06, 0.5, 40, True, p_chain, u_chain, False, False)

    # 3. RMSProp
    def setup_rmsprop():
        p_chain = [chain.Init_Perturbation_Gaussian(), chain.Scale_Perturbation(mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        u_chain = [
            chain.Init_Perturbation_Gaussian(),
            chain.Scale_Perturbation(div_by_pop=True, mul_by_std=True),
            chain_adam.OC_Compute_RMSProp(0.9, 0.1, -999),
            chain_adam.OC_Apply_RMSProp(),
            chain.Scale_Perturbation(mul_by_lr=True),
            chain.Add_Perturb_Buffer(),
            chain.Delete_Perturb_Buffer()
        ]
        return Optimizer("RMSProp", 150, 0.2, 0.5, 40, True, p_chain, u_chain, False, False)

    # 4. Adam (RMSProp + Momentum)
    def setup_adam():
        p_chain = [chain.Init_Perturbation_Gaussian(), chain.Scale_Perturbation(mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        u_chain = [
            chain.Init_Perturbation_Gaussian(),
            chain.Scale_Perturbation(div_by_pop=True, mul_by_std=True),
            chain_adam.OC_Compute_Momentum(0.9, 0.1),
            chain_adam.OC_Compute_RMSProp(0.999, 0.001, -999),
            chain.Zero_Perturb_Buffer(),
            chain_adam.OC_Add_Momentum(),
            chain_adam.OC_Apply_RMSProp(),
            chain.Scale_Perturbation(mul_by_lr=True),
            chain.Add_Perturb_Buffer(),
            chain.Delete_Perturb_Buffer()
        ]
        return Optimizer("Adam", 150, 0.2, 0.5, 40, True, p_chain, u_chain, False, False)

    # 5. NAdam
    def setup_nadam():
        p_chain = [
            chain.Init_Perturbation_Gaussian(), 
            chain.Scale_Perturbation(mul_by_std=True), 
            chain_adam.OC_Add_Momentum(),
            chain_adam.OC_Apply_RMSProp(),
            chain.Add_Perturb_Buffer(), 
            chain.Delete_Perturb_Buffer()
        ]
        u_chain = [
            chain.Init_Perturbation_Gaussian(),
            chain.Scale_Perturbation(div_by_pop=True, mul_by_std=True),
            chain_adam.OC_Compute_Momentum(0.9, 0.1),
            chain_adam.OC_Compute_RMSProp(0.999, 0.001, -999),
            chain.Zero_Perturb_Buffer(),
            chain_adam.OC_Add_Momentum(),
            chain_adam.OC_Apply_RMSProp(),
            chain.Scale_Perturbation(mul_by_lr=True),
            chain.Add_Perturb_Buffer(),
            chain.Delete_Perturb_Buffer()
        ]
        return Optimizer("NAdam", 150, 0.15, 0.5, 40, True, p_chain, u_chain, False, False)

    # 6. SignSGD
    def setup_sign_sgd():
        p_chain = [chain.Init_Perturbation_Gaussian(), chain.Scale_Perturbation(mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        u_chain = [
            chain.Init_Perturbation_Gaussian(), 
            chain.Scale_Perturbation(div_by_pop=True),
            chain.Sign_Perturb_Buffer(),
            chain.Scale_Perturbation(mul_by_lr=True), 
            chain.Add_Perturb_Buffer(), 
            chain.Delete_Perturb_Buffer()
        ]
        return Optimizer("SignSGD", 150, 0.2, 0.5, 40, True, p_chain, u_chain, False, False)

    # 7. Bernoulli Random Search (Pure SGD but Bernoulli Noise)
    def setup_bernoulli_sgd():
        p_chain = [chain.Init_Perturbation_Bernoulli(), chain.Scale_Perturbation(mul_by_std=True), chain.Add_Perturb_Buffer(), chain.Delete_Perturb_Buffer()]
        u_chain = [
            chain.Init_Perturbation_Bernoulli(), 
            chain.Scale_Perturbation(div_by_pop=True, mul_by_lr=True, mul_by_std=True), 
            chain.Add_Perturb_Buffer(), 
            chain.Delete_Perturb_Buffer()
        ]
        return Optimizer("Bernoulli SGD", 150, 0.08, 0.5, 40, True, p_chain, u_chain, False, False)

    runs = [
        ("SGD_Mir", setup_sgd),
        ("SGD_NoMir", setup_sgd_nomirror),
        ("SGD_Rank", setup_sgd_rank),
        ("Momentum", setup_momentum),
        ("RMSProp", setup_rmsprop),
        ("Adam", setup_adam),
        ("NAdam", setup_nadam),
        ("SignSGD", setup_sign_sgd),
        ("Bernoulli", setup_bernoulli_sgd),
    ]

    tests = [
        ("Ackley", ackley, (-3.0, 3.0), (-5.0, 5.0), (0.0, 0.0)),
        ("Beale", beale, (1.0, -1.0), (-4.5, 4.5), (3.0, 0.5)),
    ]
    
    results_report = []
    
    for func_name, cost_func, start_pos, value_range, true_min in tests:
        X, Y, Z = get_reward_grid(cost_func, value_range=value_range)
        
        # Clamp Z to avoid visual blowout on steep functions (like Beale's edges)
        z_max = np.percentile(Z, 95)
        Z_clamped = np.clip(Z, None, z_max)
        
        for base_name, opt_setup in runs:
            name = f"{base_name}_{func_name}"
            path_x, path_y, pert_x, pert_y, tot_dist, mean_step = run_optimizer_path(name, opt_setup, cost_func, steps=100, start_pos=start_pos, value_range=value_range)
            
            final_loss = cost_func(path_x[-1], path_y[-1])
            dist_to_min = np.sqrt((path_x[-1] - true_min[0])**2 + (path_y[-1] - true_min[1])**2)
            results_report.append((name, final_loss, tot_dist, mean_step, dist_to_min))
            
            # Plot
            plt.figure(figsize=(10, 8))
            
            plt.contourf(X, Y, Z_clamped, levels=50, cmap='viridis')
            plt.colorbar(label='Loss Value')
            
            # Plot Path
            plt.plot(path_x, path_y, 'r.-', linewidth=2, label='Optimizer Path', markersize=10)
            plt.plot(path_x[0], path_y[0], 'go', markersize=15, label='Start')
            plt.plot(path_x[-1], path_y[-1], 'rx', markersize=15, label='End')
            
            # Plot Perturbations (Cloud)
            plt.scatter(pert_x, pert_y, c='white', alpha=0.3, s=10, label='Perturbations')
            
            plt.title(f"Convergence of {name}")
            plt.xlim(value_range[0], value_range[1])
            plt.ylim(value_range[0], value_range[1])
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
            plt.savefig(out_path)
            print(f"Saved chart to {out_path}")
            plt.close()

    print("\n--- FINAL SIMULATION RESULTS ---")
    print(f"{'Optimizer':<28} | {'Final Loss':<12} | {'Total Dist':<12} | {'Mean Step':<10} | {'Dist to Min':<12}")
    print("-" * 87)
    for name, final_loss, tot_dist, mean_step, dist_to_min in results_report:
        print(f"{name:<28} | {final_loss:<12.5f} | {tot_dist:<12.5f} | {mean_step:<10.5f} | {dist_to_min:<12.5f}")

if __name__ == "__main__":
    chart_convergence()
