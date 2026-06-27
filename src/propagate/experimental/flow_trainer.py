# flow_trainer.py
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
    def __init__(self, dim, hidden_layers=2, hidden_dim=16, target_sigma=0.05):
        super().__init__()
        self.split_dim = dim // 2
        self.dim = dim
        
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim) + np.log(target_sigma))
        
        layers = []
        layers.append(nn.Linear(self.split_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, dim - self.split_dim))
        self.net = nn.Sequential(*layers)
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def generate_candidates_from_noise(self, epsilon):
        e1 = epsilon[:, :self.split_dim]
        e2 = epsilon[:, self.split_dim:]
        
        # QUADRATIC PARITY FIX
        s = self.net(e1 ** 2)
        s = torch.clamp(s, min=-2.0, max=2.0)
        
        z1 = e1
        z2 = e2 * torch.exp(s)
        z = torch.cat([z1, z2], dim=-1)
        
        log_sigma = torch.clamp(self.log_sigma, min=np.log(0.001), max=np.log(0.2))
        return z * torch.exp(log_sigma) + self.mu

    def generate_candidates(self, num_samples):
        epsilon = torch.randn(num_samples, self.dim, device=self.mu.device)
        return self.generate_candidates_from_noise(epsilon)

    def get_mode(self):
        with torch.no_grad():
            z1 = torch.zeros(1, self.split_dim, device=self.mu.device)
            s = self.net(z1 ** 2)
            s = torch.clamp(s, min=-2.0, max=2.0)
            log_sigma = torch.clamp(self.log_sigma, min=np.log(0.001), max=np.log(0.2))
            
            z2 = torch.zeros_like(s)
            z = torch.cat([z1, z2], dim=-1)
            return (z * torch.exp(log_sigma) + self.mu).squeeze(0)
        
    def log_prob(self, x):
        log_sigma = torch.clamp(self.log_sigma, min=np.log(0.001), max=np.log(0.2))
        z = (x - self.mu) * torch.exp(-log_sigma)
        
        z1 = z[:, :self.split_dim]
        z2 = z[:, self.split_dim:]
        
        # QUADRATIC PARITY FIX
        s = self.net(z1 ** 2)
        s = torch.clamp(s, min=-2.0, max=2.0)
        
        e1 = z1
        e2 = z2 * torch.exp(-s)
        epsilon = torch.cat([e1, e2], dim=-1)
        
        # SCALING FIX
        log_det_inv = -s.sum(dim=-1) - log_sigma.sum()
        pi_tensor = torch.tensor(np.pi, device=x.device)
        base_log_prob = -0.5 * (epsilon ** 2 + torch.log(2 * pi_tensor)).sum(dim=-1)
        
        return base_log_prob + log_det_inv

class OptunaFlowTrainer:
    def __init__(self, config, backend: VLLMFlowBackendLoRA, dataset: Dataset, 
                 flow_lr: float = 0.005, alpha_entropy: float = 0.01, target_sigma: float = 0.1,
                 mu_lr: float = 0.1,
                 adam_beta1: float = 0.9, adam_beta2: float = 0.999,
                 flow_hidden_layers: int = 2, flow_hidden_dim: int = 16,
                 latent_dim: int = 64, num_subspaces: int = 36,
                 rwr_mode: bool = False, rwr_tau: float = 1.0, ppo_epochs: int = 4,
                 wandb_project: str = None, wandb_project_name: str = None, 
                 validate_every: int = 0, print_samples: bool = False, 
                 checkpoint_every: int = 0, checkpoint_path: str = "checkpoints/flow_model.pth", 
                 min_val_reward: float = 0.25, start_prune_min_reward_iter: int = 10):
        
        self.config = config
        backend.startup(self.config)

        print("#-- Initializing FlowTrainer [OptunaFlowTrainer] --#")
        
        self.backend = backend
        self.dataset = dataset
        self.alpha_entropy = alpha_entropy
        self.target_sigma = target_sigma
        
        self.mu_lr = mu_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        
        self.flow_hidden_layers = flow_hidden_layers
        self.flow_hidden_dim = flow_hidden_dim
        
        self.latent_dim = latent_dim
        self.num_subspaces = num_subspaces
        self.rwr_mode = rwr_mode
        self.rwr_tau = rwr_tau
        self.ppo_epochs = ppo_epochs
        
        self.dim_params = self.backend.get_total_lora_params(self.backend.lora_perturb_target)
        
        if self.latent_dim > 0:
            print(f"#-- Using Structured Latent Bottleneck: {self.num_subspaces} subspaces, total dim {self.latent_dim} --#")
            self.dim_flow = self.latent_dim
            if self.dim_flow % self.num_subspaces != 0:
                raise ValueError("latent_dim must be divisible by num_subspaces")
            self.subspace_k = self.dim_flow // self.num_subspaces
            
            chunk_sizes = [self.dim_params // self.num_subspaces] * self.num_subspaces
            for i in range(self.dim_params % self.num_subspaces):
                chunk_sizes[i] += 1
                
            self.P_list = [torch.randn(cs, self.subspace_k) / math.sqrt(self.subspace_k) for cs in chunk_sizes]
        else:
            print("#-- Using standard full-dimension flow (NO PROJECTION) --#")
            self.dim_flow = self.dim_params
            self.P_list = None
        
        self.flow_model = FlowES(self.dim_flow, hidden_layers=self.flow_hidden_layers, hidden_dim=self.flow_hidden_dim, target_sigma=self.target_sigma)
        device = "cpu"
        self.flow_model.to(device)
        
        self.flow_params = [p for n, p in self.flow_model.named_parameters() if n != 'mu']
        self.flow_optimizer = optim.Adam(self.flow_params, lr=flow_lr, betas=(adam_beta1, adam_beta2))
        
        per_dim_target = -0.5 * (1.0 + np.log(2 * np.pi)) - np.log(self.target_sigma)
        self.target_log_prob = per_dim_target * self.dim_flow
        
        self.num_directions = self.config.population_size
        self.total_pop = self.num_directions * 2 if self.config.mirror else self.num_directions
        if self.config.centered_eval:
            self.total_pop += 1
            
        self.genomes = [Genome() for _ in range(self.total_pop)]
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
                    "flow_lr": flow_lr, "mu_lr": mu_lr,
                    "adam_beta1": adam_beta1, "adam_beta2": adam_beta2,
                    "flow_hidden_layers": flow_hidden_layers, "flow_hidden_dim": flow_hidden_dim,
                    "alpha_entropy": alpha_entropy, "target_sigma": target_sigma,
                    "rwr_mode": rwr_mode, "rwr_tau": rwr_tau, "ppo_epochs": ppo_epochs,
                    "dim_params": self.dim_params, "latent_dim": self.latent_dim, "num_subspaces": num_subspaces,
                    "batch_size": dataset.batch_size, "dataset_train_len": len(dataset.pairs_train),
                    "dataset_val_len": len(dataset.pairs_test), "backend": backend.backend_name,
                    "lora_perturb_target": backend.lora_perturb_target,
                }
                wandb.init(project=self.wandb_project, config=config_dict, name=wandb_project_name)
                wandb.define_metric("iteration_count")
                wandb.define_metric("train/*", step_metric="iteration_count")
                wandb.define_metric("val/*", step_metric="iteration_count")
            except Exception as e:
                print(f"#-- WandB logging failed: {e} --#")
                self.wandb_project = None

    def project_to_lora(self, z):
        if self.P_list is None:
            return z
        z_chunks = z.split(self.subspace_k, dim=-1)
        lora_chunks = [z_chunk @ P.T for z_chunk, P in zip(z_chunks, self.P_list)]
        return torch.cat(lora_chunks, dim=-1)

    def train(self, optuna_trial):
        latest_val_score = 0.0
        while self.iteration_count < self.config.total_steps:
            self.iteration_count += 1
            start_time = time.time()

            noise = torch.randn(self.num_directions, self.dim_flow, device=self.flow_model.mu.device)
            if self.config.mirror:
                epsilon = torch.cat([noise, -noise], dim=0)
            else:
                epsilon = noise
                
            if self.config.centered_eval:
                epsilon = torch.cat([epsilon, torch.zeros(1, self.dim_flow, device=self.flow_model.mu.device)], dim=0)
            
            with torch.no_grad():
                candidate_z = self.flow_model.generate_candidates_from_noise(epsilon)
                candidate_weights = self.project_to_lora(candidate_z)

            for i, genome in enumerate(self.genomes):
                genome.special_metadata["flow_candidate"] = candidate_weights[i].detach().cpu().clone()
            
            inputs = self.dataset.next(
                population_size=self.num_directions, 
                mirror=self.config.mirror, 
                center=self.config.centered_eval
            )
            self.backend.generate_outputs(self.genomes, None, self.dataset.suffix, inputs)
            self.dataset.score_all(self.genomes)
            
            end_time = time.time()
            
            rewards = np.array([g.historical_rewards[-1] for g in self.genomes])
            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards)
            self.log_train_stats(self.genomes, end_time - start_time, reward_mean, reward_std)
            
            # Explicit Antithetic Advantage Decomposition
            r1 = rewards[:self.num_directions]
            if self.config.mirror:
                r2 = rewards[self.num_directions:2*self.num_directions]
                adv_mu = (r1 - r2) / 2.0
                adv_sigma = ((r1 + r2) / 2.0 - reward_mean)
            else:
                adv_mu = r1 - reward_mean
                adv_sigma = r1 - reward_mean
                
            adv_mu = (adv_mu - adv_mu.mean()) / (adv_mu.std() + 1e-8)
            adv_sigma = (adv_sigma - adv_sigma.mean()) / (adv_sigma.std() + 1e-8)
            
            adv_mu_t = torch.FloatTensor(adv_mu).to(self.flow_model.mu.device)
            adv_sigma_t = torch.FloatTensor(adv_sigma).to(self.flow_model.mu.device)
            
            # Flow Network Update
            if self.rwr_mode:
                # RWR: Weighted Maximum Likelihood
                rewards_tensor = torch.FloatTensor(rewards).to(self.flow_model.mu.device)
                rewards_norm = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
                weights = torch.softmax(rewards_norm / self.rwr_tau, dim=0)
                
                # FIXED: Re-wrapped in multi-epoch loop (M-step of EM)
                total_loss_tracker, total_rwr_loss, total_ent_loss, total_log_prob_tracker = 0, 0, 0, 0
                for epoch in range(self.ppo_epochs):
                    log_probs = self.flow_model.log_prob(candidate_z)
                    rwr_loss = -(weights * log_probs).mean()
                    
                    entropy_loss = self.alpha_entropy * (log_probs.mean() - self.target_log_prob).pow(2)
                    loss = rwr_loss + entropy_loss
                    
                    self.flow_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.flow_params, 0.5)
                    self.flow_optimizer.step()
                    
                    total_loss_tracker += loss.item()
                    total_rwr_loss += rwr_loss.item()
                    total_ent_loss += entropy_loss.item()
                    total_log_prob_tracker += log_probs.mean().item()
                
                if self.wandb_project:
                    wandb.log({
                        "iteration_count": self.iteration_count,
                        "train/rl_loss": total_rwr_loss / self.ppo_epochs, "train/entropy_loss": total_ent_loss / self.ppo_epochs,
                        "train/total_loss": total_loss_tracker / self.ppo_epochs, "train/mean_log_prob": total_log_prob_tracker / self.ppo_epochs
                    })
            else:
                # REINFORCE
                advantages_full = adv_sigma_t
                if self.config.mirror:
                    advantages_full = torch.cat([adv_sigma_t, adv_sigma_t], dim=0)
                if self.config.centered_eval:
                    advantages_full = torch.cat([advantages_full, torch.zeros(1, device=adv_sigma_t.device)], dim=0)
                
                log_probs = self.flow_model.log_prob(candidate_z.detach())
                
                rl_loss = -(log_probs * advantages_full).mean()
                entropy_loss = self.alpha_entropy * (log_probs.mean() - self.target_log_prob).pow(2)
                loss = rl_loss + entropy_loss
                
                self.flow_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_params, 0.5)
                self.flow_optimizer.step()
                
                if self.wandb_project:
                    wandb.log({
                        "iteration_count": self.iteration_count,
                        "train/rl_loss": rl_loss.item(), "train/entropy_loss": entropy_loss.item(),
                        "train/total_loss": loss.item(), "train/mean_log_prob": log_probs.mean().item()
                    })
            
            # SAFE MU UPDATE
            with torch.no_grad():
                delta_z = candidate_z - self.flow_model.mu
                mu_grad = (adv_mu_t.unsqueeze(1) * delta_z[:self.num_directions]).mean(dim=0)
                self.flow_model.mu += self.mu_lr * mu_grad

            # Validation
            if self.validate_every > 0 and self.iteration_count % self.validate_every == 0:
                mode_z = self.flow_model.get_mode()
                mode_weights = self.project_to_lora(mode_z.unsqueeze(0)).squeeze(0)
                    
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
                if self.iteration_count >= self.start_prune_min_reward_iter and latest_val_score < self.min_val_reward:
                    raise optuna.TrialPruned()
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            if self.checkpoint_every > 0 and self.iteration_count % self.checkpoint_every == 0:
                base, ext = os.path.splitext(self.checkpoint_path)
                path = f"{base}_step_{self.iteration_count}{ext}"
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                torch.save(self.flow_model.state_dict(), path)
                
            self.genomes = [Genome() for _ in range(self.total_pop)]

        return latest_val_score

    def log_train_stats(self, genomes: List[Genome], time_taken: float, reward_mean: float, reward_std: float):
        average_response_length = sum([sum(len(response.split()) for response in genome.latest_outputs) for genome in genomes]) / len(genomes) / max(1, len(genomes[0].latest_outputs))
        best_genome = max(genomes, key=lambda g: g.historical_rewards[-1])
        worst_genome = min(genomes, key=lambda g: g.historical_rewards[-1])
        if self.wandb_project:
            try:
                sample_table = wandb.Table(columns=["type", "reward", "response"])
                sample_table.add_data("best", best_genome.historical_rewards[-1], best_genome.latest_outputs[0])
                sample_table.add_data("worst", worst_genome.historical_rewards[-1], worst_genome.latest_outputs[0])
                wandb.log({
                    "train/average_reward": reward_mean, "train/min_reward": worst_genome.historical_rewards[-1],
                    "train/max_reward": best_genome.historical_rewards[-1], "train/stddev_reward": reward_std,
                    "train/time_seconds": time_taken, "train/average_response_length": average_response_length,
                    "train/samples": sample_table, "iteration_count": self.iteration_count
                })
            except Exception: pass
        print(f"#-- Stats: average: {reward_mean:.4f}, min: {worst_genome.historical_rewards[-1]:.4f}, max: {best_genome.historical_rewards[-1]:.4f}, stddev: {reward_std:.4f} --#")

    def log_val_stats(self, genome: Genome, time_taken: float):
        score = genome.historical_rewards[-1]
        if self.wandb_project:
            try:
                wandb.log({
                    "val/validation_score": score, "val/time_seconds": time_taken,
                    "val/sample_response": wandb.Table(data=[[genome.latest_outputs[0]]], columns=["response"]),
                    "iteration_count": self.iteration_count
                })
            except Exception: pass
        print(f"#-- Mode Validation Stats: reward: {score:.4f}, time: {time_taken:.2f}s --#")


# =====================================================================
# XNES TRAINER (Analytical Covariance Learning in Structured Latent Space)
# =====================================================================
class XNESTrainer(OptunaFlowTrainer):
    def __init__(self, config, backend, dataset, 
                 mu_lr=0.1, sigma_lr=0.1, cov_lr=0.1, target_sigma=0.1, latent_dim=64, num_subspaces=36,
                 wandb_project=None, wandb_project_name=None, validate_every=0, 
                 print_samples=False, min_val_reward=0.25, start_prune_min_reward_iter=10):
        
        self.config = config
        backend.startup(self.config)
        self.backend = backend
        self.dataset = dataset
        self.mu_lr = mu_lr
        self.sigma_lr = sigma_lr
        self.cov_lr = cov_lr
        self.target_sigma = target_sigma
        self.latent_dim = latent_dim
        self.num_subspaces = num_subspaces
        self.wandb_project = wandb_project
        self.validate_every = validate_every
        self.print_samples = print_samples
        self.min_val_reward = min_val_reward
        self.start_prune_min_reward_iter = start_prune_min_reward_iter
        
        self.dim_params = self.backend.get_total_lora_params(self.backend.lora_perturb_target)
        self.dim_flow = self.latent_dim
        
        if self.dim_flow % self.num_subspaces != 0:
            raise ValueError("latent_dim must be divisible by num_subspaces")
        self.subspace_k = self.dim_flow // self.num_subspaces
        
        chunk_sizes = [self.dim_params // self.num_subspaces] * self.num_subspaces
        for i in range(self.dim_params % self.num_subspaces):
            chunk_sizes[i] += 1
        self.P_list = [torch.randn(cs, self.subspace_k) / math.sqrt(self.subspace_k) for cs in chunk_sizes]
        
        self.mu = torch.zeros(self.dim_flow)
        self.sigma = torch.tensor(self.target_sigma, dtype=torch.float32)
        self.cov = torch.eye(self.dim_flow)
        self.L = torch.linalg.cholesky(self.cov)
        
        self.num_directions = self.config.population_size
        self.total_pop = self.num_directions * 2 if self.config.mirror else self.num_directions
        if self.config.centered_eval:
            self.total_pop += 1
            
        self.genomes = [Genome() for _ in range(self.total_pop)]
        self.iteration_count = 0
        
        if self.wandb_project:
            try:
                wandb.init(project=self.wandb_project, config={"algo": "xNES", "latent_dim": self.latent_dim, "num_subspaces": self.num_subspaces}, name=wandb_project_name)
                wandb.define_metric("iteration_count")
                wandb.define_metric("train/*", step_metric="iteration_count")
                wandb.define_metric("val/*", step_metric="iteration_count")
            except: 
                self.wandb_project = None

    def project_to_lora(self, z):
        z_chunks = z.split(self.subspace_k, dim=-1)
        lora_chunks = [z_chunk @ P.T for z_chunk, P in zip(z_chunks, self.P_list)]
        return torch.cat(lora_chunks, dim=-1)

    def train(self, optuna_trial):
        latest_val_score = 0.0
        while self.iteration_count < self.config.total_steps:
            self.iteration_count += 1
            start_time = time.time()

            noise = torch.randn(self.num_directions, self.dim_flow)
            if self.config.mirror:
                epsilon = torch.cat([noise, -noise], dim=0)
            else:
                epsilon = noise
                
            if self.config.centered_eval:
                epsilon = torch.cat([epsilon, torch.zeros(1, self.dim_flow)], dim=0)
            
            z = self.mu + self.sigma * (epsilon @ self.L.T)
            candidate_weights = self.project_to_lora(z)
            
            for i, genome in enumerate(self.genomes):
                genome.special_metadata["flow_candidate"] = candidate_weights[i].detach().cpu().clone()
            
            inputs = self.dataset.next(
                population_size=self.num_directions, 
                mirror=self.config.mirror, 
                center=self.config.centered_eval
            )
            self.backend.generate_outputs(self.genomes, None, self.dataset.suffix, inputs)
            self.dataset.score_all(self.genomes)
            
            rewards = np.array([g.historical_rewards[-1] for g in self.genomes])
            r_mean = np.mean(rewards)
            r1 = rewards[:self.num_directions]
            if self.config.mirror:
                r2 = rewards[self.num_directions:2*self.num_directions]
                adv_mu = (r1 - r2) / 2.0
                adv_sigma = ((r1 + r2) / 2.0 - np.mean(rewards))
            else:
                adv_mu = r1 - np.mean(rewards)
                adv_sigma = r1 - np.mean(rewards)
                
            adv_mu = (adv_mu - adv_mu.mean()) / (adv_mu.std() + 1e-8)
            adv_sigma = (adv_sigma - adv_sigma.mean()) / (adv_sigma.std() + 1e-8)
            
            adv_mu_t = torch.FloatTensor(adv_mu)
            adv_sigma_t = torch.FloatTensor(adv_sigma)
            
            with torch.no_grad():
                # Mu Update
                mu_grad = (adv_mu_t.unsqueeze(1) * epsilon[:self.num_directions]).mean(dim=0)
                self.mu += self.mu_lr * self.sigma * (self.L @ mu_grad)
                
                # Sigma Update
                norms = (epsilon[:self.num_directions] ** 2).sum(dim=-1) / self.dim_flow - 1.0
                sigma_grad = 0.5 * (adv_sigma_t * norms).mean()
                self.sigma *= torch.exp(self.sigma_lr * sigma_grad)
                self.sigma = torch.clamp(self.sigma, min=1e-4, max=0.5)
                
                # Covariance Update
                eps = epsilon[:self.num_directions]
                adv = adv_sigma_t
                cov_grad = 0.5 * (adv.view(-1, 1, 1) * (torch.bmm(eps.unsqueeze(2), eps.unsqueeze(1)) - torch.eye(self.dim_flow))).mean(dim=0)
                
                # FIXED: Sandwich product to guarantee symmetry and PSD
                exp_term = torch.linalg.matrix_exp(self.cov_lr * cov_grad)
                self.cov = exp_term @ self.cov @ exp_term.T
                self.cov = (self.cov + self.cov.T) / 2.0  # Symmetrize for numerical drift
                
                # Update Cholesky (FIXED: Eigenvalue clipping fallback)
                try:
                    self.L = torch.linalg.cholesky(self.cov + 1e-6 * torch.eye(self.dim_flow, device=self.cov.device))
                except:
                    print("#-- Cholesky failed, clipping eigenvalues --#")
                    eigvals, eigvecs = torch.linalg.eigh((self.cov + self.cov.T) / 2.0)
                    eigvals = torch.clamp(eigvals, min=1e-4)
                    self.cov = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                    self.cov = (self.cov + self.cov.T) / 2.0 + 1e-5 * torch.eye(self.dim_flow, device=self.cov.device)
                    self.L = torch.linalg.cholesky(self.cov)

            end_time = time.time()
            print(f"#-- Iteration {self.iteration_count} (xNES) completed in {end_time - start_time:.2f}s | Reward: {r_mean:.4f} --#")
            if self.wandb_project:
                wandb.log({"train/average_reward": r_mean, "train/sigma": self.sigma.item(), "iteration_count": self.iteration_count})

            if self.validate_every > 0 and self.iteration_count % self.validate_every == 0:
                mode_z = self.mu
                mode_weights = self.project_to_lora(mode_z.unsqueeze(0)).squeeze(0)
                new_genome = Genome()
                new_genome.special_metadata["flow_candidate"] = mode_weights.detach().cpu().clone()
                
                prompts = self.dataset.get_test_set()
                self.backend.generate_outputs([new_genome], None, self.dataset.suffix, prompts)
                self.dataset.score_all([new_genome])
                latest_val_score = new_genome.historical_rewards[-1]
                
                if self.wandb_project:
                    wandb.log({"val/validation_score": latest_val_score, "iteration_count": self.iteration_count})
                print(f"#-- Mode Validation Stats: reward: {latest_val_score:.4f} --#")
                
                optuna_trial.report(latest_val_score, self.iteration_count)
                if self.iteration_count >= self.start_prune_min_reward_iter and latest_val_score < self.min_val_reward:
                    raise optuna.TrialPruned()
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            self.genomes = [Genome() for _ in range(self.total_pop)]

        return latest_val_score