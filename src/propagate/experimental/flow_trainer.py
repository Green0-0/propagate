import json
import math
import os
from typing import Any, List, Dict
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from propagate.backend.backend_abc import Backend
from propagate.datasets.dataset import Dataset
from propagate.optimizers.optimizer import Optimizer
import time
import wandb
from propagate.genome import Genome
from propagate.backend.experimental.vllm_flow_backend import VLLMFlowBackendLoRA

class FlowES(nn.Module):
    def __init__(self, dim, hidden_layers=2, hidden_dim=16):
        super().__init__()
        self.split_dim = dim // 2
        self.dim = dim
        
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim) + np.log(0.05))
        
        layers = []
        layers.append(nn.Linear(self.split_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, (dim - self.split_dim) * 2))
        self.net = nn.Sequential(*layers)
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def generate_candidates_from_noise(self, epsilon):
        e1 = epsilon[:, :self.split_dim]
        e2 = epsilon[:, self.split_dim:]
        
        out = self.net(e1)
        s, t = out.chunk(2, dim=-1)
        s = torch.clamp(s, min=-2.0, max=2.0)
        
        z1 = e1
        z2 = e2 * torch.exp(s) + t
        z = torch.cat([z1, z2], dim=-1)
        
        log_sigma = torch.clamp(self.log_sigma, min=np.log(0.01), max=np.log(0.2))
        return z * torch.exp(log_sigma) + self.mu

    def generate_candidates(self, num_samples):
        epsilon = torch.randn(num_samples, self.dim, device=self.mu.device)
        return self.generate_candidates_from_noise(epsilon)

    def get_mode(self):
        with torch.no_grad():
            z1 = torch.zeros(1, self.split_dim, device=self.mu.device)
            out = self.net(z1)
            s, t = out.chunk(2, dim=-1)
            s = torch.clamp(s, min=-2.0, max=2.0)
            log_sigma = torch.clamp(self.log_sigma, min=np.log(0.01), max=np.log(0.2))
            
            z2 = t
            z = torch.cat([z1, z2], dim=-1)
            return (z * torch.exp(log_sigma) + self.mu).squeeze(0)
        
    def log_prob(self, x):
        log_sigma = torch.clamp(self.log_sigma, min=np.log(0.01), max=np.log(0.2))
        z = (x - self.mu) * torch.exp(-log_sigma)
        
        z1 = z[:, :self.split_dim]
        z2 = z[:, self.split_dim:]
        
        out = self.net(z1)
        s, t = out.chunk(2, dim=-1)
        s = torch.clamp(s, min=-2.0, max=2.0)
        
        e1 = z1
        e2 = (z2 - t) * torch.exp(-s)
        epsilon = torch.cat([e1, e2], dim=-1)
        
        log_det_inv = -s.sum(dim=-1) - log_sigma.sum()
        pi_tensor = torch.tensor(np.pi, device=x.device)
        base_log_prob = -0.5 * (epsilon ** 2 + torch.log(2 * pi_tensor)).sum(dim=-1)
        
        return base_log_prob + log_det_inv

class OptunaFlowTrainer:
    def __init__(self, config, backend: VLLMFlowBackendLoRA, dataset: Dataset, 
                 flow_lr: float = 0.005, alpha_entropy: float = 0.01, target_sigma: float = 0.1,
                 mu_lr: float = 0.1, mu_momentum: float = 0.9,
                 adam_beta1: float = 0.9, adam_beta2: float = 0.999,
                 flow_hidden_layers: int = 2, flow_hidden_dim: int = 16,
                 ppo_mode: bool = False, ppo_epochs: int = 4, ppo_minibatches: int = 4, 
                 clip_eps: float = 0.2, grad_clip: float = 0.5,
                 wandb_project: str = None, wandb_project_name: str = None, 
                 validate_every: int = 0, print_samples: bool = False, 
                 checkpoint_every: int = 0, checkpoint_path: str = "checkpoints/flow_model.pth", 
                 min_val_reward: float = 0.25, start_prune_min_reward_iter: int = 10):
        
        self.config = config
        backend.startup(self.config)

        print("#-- Initializing FlowTrainer [OptunaFlowTrainer] --#")
        print(f"#-- Population Size: {self.config.population_size}, Flow LR: {flow_lr}, Alpha Entropy: {alpha_entropy}, PPO Mode: {ppo_mode} --#")
        
        self.backend = backend
        self.dataset = dataset
        self.alpha_entropy = alpha_entropy
        self.target_sigma = target_sigma
        
        self.mu_lr = mu_lr
        self.mu_momentum = mu_momentum
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        
        self.flow_hidden_layers = flow_hidden_layers
        self.flow_hidden_dim = flow_hidden_dim
        
        self.ppo_mode = ppo_mode
        self.ppo_epochs = ppo_epochs
        self.ppo_minibatches = ppo_minibatches
        self.clip_eps = clip_eps
        self.grad_clip = grad_clip
        
        # FIXED: Pass the perturb target to get the correct dimension
        self.dim_params = self.backend.get_total_lora_params(self.backend.lora_perturb_target)
        print(f"#-- Target LoRA Parameters Dimension: {self.dim_params} --#")
        
        self.flow_model = FlowES(self.dim_params, hidden_layers=self.flow_hidden_layers, hidden_dim=self.flow_hidden_dim)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.flow_model.to(device)
        
        # FIXED: Split optimizers for mu and flow network
        flow_params = [p for n, p in self.flow_model.named_parameters() if n != 'mu']
        self.flow_optimizer = optim.Adam(flow_params, lr=flow_lr, betas=(adam_beta1, adam_beta2))
        self.mu_optimizer = optim.SGD([self.flow_model.mu], lr=mu_lr, momentum=mu_momentum)
        
        self.target_log_prob = -0.5 * self.dim_params * (1.0 + np.log(2 * np.pi)) - self.dim_params * np.log(self.target_sigma)
        
        self.genomes = [Genome() for _ in range(self.config.population_size)]
        self.iteration_count = 0
        self.wandb_project = wandb_project
        self.validate_every = validate_every
        self.print_samples = print_samples
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.min_val_reward = min_val_reward
        self.start_prune_min_reward_iter = start_prune_min_reward_iter

        if self.wandb_project is not None and self.wandb_project != "":
            try:
                wandb.login()
                config_dict = {
                    "population_size": self.config.population_size,
                    "flow_lr": flow_lr,
                    "mu_lr": mu_lr,
                    "mu_momentum": mu_momentum,
                    "adam_beta1": adam_beta1,
                    "adam_beta2": adam_beta2,
                    "flow_hidden_layers": flow_hidden_layers,
                    "flow_hidden_dim": flow_hidden_dim,
                    "alpha_entropy": alpha_entropy,
                    "target_sigma": target_sigma,
                    "ppo_mode": ppo_mode,
                    "ppo_epochs": ppo_epochs,
                    "ppo_minibatches": ppo_minibatches,
                    "clip_eps": clip_eps,
                    "grad_clip": grad_clip,
                    "dim_params": self.dim_params,
                    "batch_size": dataset.batch_size,
                    "dataset_train_len": len(dataset.pairs_train),
                    "dataset_val_len": len(dataset.pairs_test),
                    "backend": backend.backend_name,
                    "lora_perturb_target": backend.lora_perturb_target,
                }
                wandb.init(project=self.wandb_project, config=config_dict, name=wandb_project_name)
                wandb.define_metric("iteration_count")
                wandb.define_metric("train/*", step_metric="iteration_count")
                wandb.define_metric("val/*", step_metric="iteration_count")
                print(f"#-- WandB logging initialized for project: {self.wandb_project} --#")
            except Exception as e:
                print(f"#-- WandB logging initialization failed: {e} --#")
                self.wandb_project = None

    def train(self, optuna_trial):
        latest_val_score = 0.0
        while self.iteration_count < self.config.total_steps:
            self.iteration_count += 1
            start_time = time.time()

            # 1. Generate Candidates
            # FIXED: Restored antithetic noise to BOTH branches
            noise = torch.randn(self.config.population_size // 2, self.dim_params, device=self.flow_model.mu.device)
            epsilon = torch.cat([noise, -noise], dim=0)
            if self.config.population_size % 2 != 0:
                epsilon = torch.cat([epsilon, torch.randn(1, self.dim_params, device=self.flow_model.mu.device)], dim=0)
            
            with torch.no_grad():
                candidate_weights = self.flow_model.generate_candidates_from_noise(epsilon)
                if self.ppo_mode:
                    old_log_probs = self.flow_model.log_prob(candidate_weights)

            for i, genome in enumerate(self.genomes):
                genome.special_metadata["flow_candidate"] = candidate_weights[i].detach().cpu().clone()
            
            # 2. Evaluate Candidates
            inputs = self.dataset.next(population_size=self.config.population_size, mirror=False, center=False)
            self.backend.generate_outputs(self.genomes, None, self.dataset.suffix, inputs)
            self.dataset.score_all(self.genomes)
            
            end_time = time.time()
            print(f"#-- Iteration {self.iteration_count} completed in {end_time - start_time:.2f} seconds --#")
            
            rewards = np.array([g.historical_rewards[-1] for g in self.genomes])
            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards)
            self.log_train_stats(self.genomes, end_time - start_time, reward_mean, reward_std)
            
            # 3. Gradient Update
            norm_rewards = (rewards - reward_mean) / (reward_std + 1e-8)
            norm_rewards_t = torch.FloatTensor(norm_rewards).to(self.flow_model.mu.device)
            
            if self.ppo_mode:
                batch_size = self.config.population_size
                minibatch_size = max(1, batch_size // self.ppo_minibatches)
                
                total_loss_tracker = 0
                total_ppo_loss = 0
                total_ent_loss = 0
                total_log_prob_tracker = 0  # FIXED: Track log prob across all minibatches
                
                for epoch in range(self.ppo_epochs):
                    perm = torch.randperm(batch_size, device=self.flow_model.mu.device)
                    
                    for m in range(self.ppo_minibatches):
                        idx = perm[m * minibatch_size : (m + 1) * minibatch_size]
                        if len(idx) == 0: continue
                        
                        mb_targets = candidate_weights[idx].detach()
                        mb_advantages = norm_rewards_t[idx]
                        mb_old_log_probs = old_log_probs[idx].detach()
                        
                        new_log_probs = self.flow_model.log_prob(mb_targets)
                        
                        log_ratio = new_log_probs - mb_old_log_probs
                        ratio = torch.exp(torch.clamp(log_ratio, min=-10.0, max=10.0))
                        
                        surr1 = ratio * mb_advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                        ppo_loss = -torch.min(surr1, surr2).mean()
                        
                        entropy_loss = self.alpha_entropy * (new_log_probs.mean() - self.target_log_prob).pow(2)
                        
                        loss = ppo_loss + entropy_loss
                        
                        self.flow_optimizer.zero_grad()
                        self.mu_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.grad_clip)
                        self.flow_optimizer.step()
                        self.mu_optimizer.step()
                        
                        total_loss_tracker += loss.item()
                        total_ppo_loss += ppo_loss.item()
                        total_ent_loss += entropy_loss.item()
                        total_log_prob_tracker += new_log_probs.mean().item()
                        
                if self.wandb_project:
                    updates = self.ppo_epochs * self.ppo_minibatches
                    wandb.log({
                        "train/rl_loss": total_ppo_loss / updates,
                        "train/entropy_loss": total_ent_loss / updates,
                        "train/total_loss": total_loss_tracker / updates,
                        "train/mean_log_prob": total_log_prob_tracker / updates  # FIXED
                    }, step=self.iteration_count)
            else:
                fixed_targets = candidate_weights.detach()
                log_probs = self.flow_model.log_prob(fixed_targets)
                
                rl_loss = -(log_probs * norm_rewards_t).mean()
                entropy_loss = self.alpha_entropy * (log_probs.mean() - self.target_log_prob).pow(2)
                
                loss = rl_loss + entropy_loss
                
                self.flow_optimizer.zero_grad()
                self.mu_optimizer.zero_grad()
                loss.backward()
                self.flow_optimizer.step()
                self.mu_optimizer.step()
                
                if self.wandb_project:
                    wandb.log({
                        "train/rl_loss": rl_loss.item(),
                        "train/entropy_loss": entropy_loss.item(),
                        "train/total_loss": loss.item(),
                        "train/mean_log_prob": log_probs.mean().item()
                    }, step=self.iteration_count)

            # 4. Validation
            if self.validate_every > 0 and self.iteration_count % self.validate_every == 0:
                mode_weights = self.flow_model.get_mode()
                new_genome = Genome()
                new_genome.special_metadata["flow_candidate"] = mode_weights.detach().cpu().clone()
                
                val_start_time = time.time()
                prompts = self.dataset.get_test_set()
                self.backend.generate_outputs([new_genome], None, self.dataset.suffix, prompts)
                self.dataset.score_all([new_genome])
                val_end_time = time.time()
                
                latest_val_score = new_genome.historical_rewards[-1]
                self.log_val_stats(new_genome, val_end_time - val_start_time)
                
                optuna_trial.report(latest_val_score, self.iteration_count)
                
                if self.iteration_count >= self.start_prune_min_reward_iter:
                    if latest_val_score < self.min_val_reward:
                        print(f"Run failed early garbage check at step {self.iteration_count}. Killing it.")
                        raise optuna.TrialPruned()
                
                if optuna_trial.should_prune():
                    print(f"Run failed to pass Optuna's pruning check at step {self.iteration_count}.")
                    raise optuna.TrialPruned()

            # 5. Checkpoint
            if self.checkpoint_every > 0 and self.iteration_count % self.checkpoint_every == 0:
                base, ext = os.path.splitext(self.checkpoint_path)
                path = f"{base}_step_{self.iteration_count}{ext}"
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                torch.save(self.flow_model.state_dict(), path)
                print(f"#-- Successfully saved flow model to {path} --#")
                
            self.genomes = [Genome() for _ in range(self.config.population_size)]

        return latest_val_score

    def log_train_stats(self, genomes: List[Genome], time_taken: float, reward_mean: float, reward_std: float):
        average_response_length = sum([sum(len(response.split()) for response in genome.latest_outputs) for genome in genomes]) / len(genomes) / max(1, len(genomes[0].latest_outputs))
        best_genome = max(genomes, key=lambda g: g.historical_rewards[-1])
        worst_genome = min(genomes, key=lambda g: g.historical_rewards[-1])
        
        if self.wandb_project is not None and self.wandb_project != "":
            try:
                sample_table = wandb.Table(columns=["type", "reward", "response"])
                sample_table.add_data("best", best_genome.historical_rewards[-1], best_genome.latest_outputs[0])
                sample_table.add_data("worst", worst_genome.historical_rewards[-1], worst_genome.latest_outputs[0])
                
                log_data = {
                    f"train/average_reward": reward_mean,
                    f"train/min_reward": worst_genome.historical_rewards[-1],
                    f"train/max_reward": best_genome.historical_rewards[-1],
                    f"train/stddev_reward": reward_std,
                    f"train/time_seconds": time_taken,
                    f"train/average_response_length": average_response_length,
                    f"train/samples": sample_table,
                    f"iteration_count": self.iteration_count
                }
                wandb.log(log_data, step=self.iteration_count)
            except Exception:
                pass
                
        print(f"#-- Stats: average: {reward_mean:.4f}, min: {worst_genome.historical_rewards[-1]:.4f}, max: {best_genome.historical_rewards[-1]:.4f}, stddev: {reward_std:.4f} --#")
        if self.print_samples:
            print(f"#-- BEST GENOME --#\n{best_genome.latest_outputs[0]}\n")
            print(f"#-- WORST GENOME --#\n{worst_genome.latest_outputs[0]}\n") 

    def log_val_stats(self, genome: Genome, time_taken: float):
        score = genome.historical_rewards[-1]
        average_response_length = sum(len(response.split()) for response in genome.latest_outputs) / max(1, len(genome.latest_outputs))
        
        if self.wandb_project is not None and self.wandb_project != "":
            try:
                wandb.log({
                    f"val/validation_score": score,
                    f"val/time_seconds": time_taken,
                    f"val/average_response_length": average_response_length,
                    f"val/sample_response": wandb.Table(data=[[genome.latest_outputs[0]]], columns=["response"]),
                    f"iteration_count": self.iteration_count
                }, step=self.iteration_count)
            except Exception:
                pass
                
        print(f"#-- Mode Validation Stats: reward: {score:.4f}, time: {time_taken:.2f}s --#")
        if self.print_samples:
            print(f"#-- MODE GENOME RESPONSE --#\n{genome.latest_outputs[0]}\n")