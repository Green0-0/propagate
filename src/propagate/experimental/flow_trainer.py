import math
import os
from typing import List, Dict
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from propagate.datasets.dataset import Dataset
import time
import wandb
from propagate.genome import Genome
from propagate.backend.experimental.vllm_flow_backend import VLLMFlowBackendLoRA

def centered_ranks(x: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(x, dtype=np.float32)
    ranks[np.argsort(x)] = np.arange(len(x), dtype=np.float32)
    ranks /= max(len(x) - 1, 1)
    return ranks - 0.5

class FlowES(nn.Module):
    def __init__(self, dim, hidden_layers=2, hidden_dim=16, target_sigma=0.05):
        super().__init__()
        self.dim = dim
        self.use_flow = dim >= 2

        self.register_buffer("pi_tensor", torch.tensor(np.pi))
        
        # requires_grad=False prevents silent gradient accumulation
        self.mu = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.log_sigma = nn.Parameter(torch.zeros(dim) + np.log(target_sigma))

        if self.use_flow:
            self.split_dim = dim // 2
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
        log_sigma = torch.clamp(self.log_sigma, min=np.log(0.001), max=np.log(0.2))
        if not self.use_flow:
            return epsilon * torch.exp(log_sigma) + self.mu

        e1 = epsilon[:, :self.split_dim]
        e2 = epsilon[:, self.split_dim:]
        # Use parity fix to maintain antithetic sampling property
        s = self.net(e1 ** 2)
        s = torch.clamp(s, min=-2.0, max=2.0)
        z1 = e1
        z2 = e2 * torch.exp(s)
        z = torch.cat([z1, z2], dim=-1)
        return z * torch.exp(log_sigma) + self.mu

    def generate_candidates(self, num_samples):
        epsilon = torch.randn(num_samples, self.dim, device=self.mu.device)
        return self.generate_candidates_from_noise(epsilon)

    def get_mode(self):
        return self.mu

    def log_prob(self, x):
        log_sigma = torch.clamp(self.log_sigma, min=np.log(0.001), max=np.log(0.2))
        z = (x - self.mu) * torch.exp(-log_sigma)

        if not self.use_flow:
            base_log_prob = -0.5 * (z ** 2 + torch.log(2 * self.pi_tensor)).sum(dim=-1) - log_sigma.sum()
            return base_log_prob

        z1 = z[:, :self.split_dim]
        z2 = z[:, self.split_dim:]
        s = self.net(z1 ** 2)
        s = torch.clamp(s, min=-2.0, max=2.0)
        e1 = z1
        e2 = z2 * torch.exp(-s)
        epsilon = torch.cat([e1, e2], dim=-1)

        log_det_inv = -s.sum(dim=-1) - log_sigma.sum()
        base_log_prob = -0.5 * (epsilon ** 2 + torch.log(2 * self.pi_tensor)).sum(dim=-1)
        return base_log_prob + log_det_inv


class OptunaFlowTrainer:
    def __init__(self, config, backend: VLLMFlowBackendLoRA, dataset: Dataset,
                 flow_lr: float = 0.005, alpha_entropy: float = 0.01, target_sigma: float = 0.1,
                 mu_lr: float = 0.1, adam_beta1: float = 0.9, adam_beta2: float = 0.999,
                 flow_hidden_layers: int = 2, flow_hidden_dim: int = 16,
                 lora_rank: int = 2, u_dim: int = 8, n_tie: int = 1,
                 trial_number: int = 0,
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
        self.trial_number = trial_number

        self.mu_lr = mu_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        self.flow_hidden_layers = flow_hidden_layers
        self.flow_hidden_dim = flow_hidden_dim

        self.lora_rank = lora_rank
        self.u_dim = u_dim
        self.n_tie = n_tie

        print(f"#-- Computing TinyLoRA SVD (lora_rank={self.lora_rank}) --#")
        svd_info = self.backend.compute_tinylora_svd(self.lora_rank)
        print(f"#-- SVD computed for {len(svd_info)} modules --#")

        self.num_modules = len(svd_info)
        self.num_groups = math.ceil(self.num_modules / self.n_tie)
        self.dim_flow = self.u_dim * self.num_groups

        print(f"#-- TinyLoRA: {self.num_modules} modules, {self.num_groups} tied groups, "
              f"u_dim={self.u_dim}, dim_flow={self.dim_flow} --#")

        self.backend.init_tinylora(self.u_dim, self.n_tie, self.lora_rank)

        self.flow_model = FlowES(self.dim_flow, hidden_layers=self.flow_hidden_layers,
                                  hidden_dim=self.flow_hidden_dim, target_sigma=self.target_sigma)
        device = "cpu"
        self.flow_model.to(device)

        self.flow_params = [p for n, p in self.flow_model.named_parameters() if n != 'mu']
        self.flow_optimizer = optim.Adam(self.flow_params, lr=flow_lr, betas=(adam_beta1, adam_beta2))

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
                    "lora_rank": lora_rank, "u_dim": u_dim, "n_tie": n_tie,
                    "num_modules": self.num_modules, "num_groups": self.num_groups,
                    "dim_flow": self.dim_flow,
                    "use_rslora": backend.use_rslora, "normalize_svd": backend.normalize_svd,
                }
                wandb.init(project=self.wandb_project, config=config_dict, name=wandb_project_name)
                wandb.define_metric("iteration_count")
                wandb.define_metric("train/*", step_metric="iteration_count")
                wandb.define_metric("val/*", step_metric="iteration_count")
            except Exception as e:
                print(f"#-- WandB logging failed: {e} --#")
                self.wandb_project = None

    def save_final_model(self, filepath: str):
        mode_v = self.flow_model.get_mode()
        self.backend.apply_mode_permanently(mode_v.detach().cpu())
        self.backend.save_weights_to_disk(filepath)

    def train(self, optuna_trial):
        latest_val_score = 0.0
        while self.iteration_count < self.config.total_steps:
            self.iteration_count += 1
            start_time = time.time()

            noise = torch.randn(self.num_directions, self.dim_flow, device=self.flow_model.mu.device)

            with torch.no_grad():
                delta_plus = self.flow_model.generate_candidates_from_noise(noise) - self.flow_model.mu
                if self.config.mirror:
                    candidate_v = torch.cat([
                        self.flow_model.mu + delta_plus,
                        self.flow_model.mu - delta_plus,
                    ], dim=0)
                else:
                    candidate_v = self.flow_model.mu + delta_plus

                if self.config.centered_eval:
                    candidate_v = torch.cat([candidate_v, self.flow_model.mu.unsqueeze(0)], dim=0)

            for i, genome in enumerate(self.genomes):
                genome.special_metadata["flow_v"] = candidate_v[i].detach().cpu().clone()

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

            rewards_for_update = rewards.copy()
            if getattr(self.config, "rank_norm_rewards", False):
                rewards_for_update = centered_ranks(rewards_for_update)

            r1 = rewards_for_update[:self.num_directions]
            if self.config.mirror:
                r2 = rewards_for_update[self.num_directions:2*self.num_directions]
                adv_mu = (r1 - r2) / 2.0
                adv_sigma = ((r1 + r2) / 2.0 - np.mean(rewards_for_update))
            else:
                adv_mu = r1 - np.mean(rewards_for_update)
                adv_sigma = r1 - np.mean(rewards_for_update)

            adv_mu = (adv_mu - adv_mu.mean()) / (adv_mu.std() + 1e-8)
            adv_sigma = (adv_sigma - adv_sigma.mean()) / (adv_sigma.std() + 1e-8)

            adv_mu_t = torch.from_numpy(adv_mu).float().to(self.flow_model.mu.device)
            adv_sigma_t = torch.from_numpy(adv_sigma).float().to(self.flow_model.mu.device)

            # --- Flow Network Update (REINFORCE) ---
            # Train the flow strictly on the symmetric advantage to avoid confounding mean location shifts
            advantages_full = adv_sigma_t
            if self.config.mirror:
                advantages_full = torch.cat([adv_sigma_t, adv_sigma_t], dim=0)

            if self.config.centered_eval:
                advantages_full = torch.cat([advantages_full, torch.zeros(1, device=adv_sigma_t.device)], dim=0)

            log_probs = self.flow_model.log_prob(candidate_v.detach())

            rl_loss = -(log_probs * advantages_full).mean()
            
            # Scale regularizer directly on clamped log_sigma to ensure it has gradients
            clamped_log_sigma = torch.clamp(self.flow_model.log_sigma, min=np.log(0.001), max=np.log(0.2))
            scale_loss = self.alpha_entropy * (torch.exp(clamped_log_sigma).mean() - self.target_sigma).pow(2)
            
            loss = rl_loss + scale_loss

            self.flow_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow_params, 0.5)
            self.flow_optimizer.step()

            if self.wandb_project:
                wandb.log({
                    "iteration_count": self.iteration_count,
                    "train/rl_loss": rl_loss.item(),
                    "train/scale_loss": scale_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/mean_log_prob": log_probs.mean().item(),
                    "train/empirical_std": torch.std(candidate_v, dim=0).mean().item()
                }, step=self.iteration_count)

            # --- Mu Update (Natural Gradient style: multiplying by sigma) ---
            with torch.no_grad():
                # By using delta_plus (which is z * sigma), we inherently multiply the gradient
                # by sigma. This matches the natural gradient rule and prevents parameter 
                # explosion as the variance shrinks, aligning with xNES and your old code.
                mu_grad = (adv_mu_t.unsqueeze(1) * delta_plus[:self.num_directions]).mean(dim=0)
                self.flow_model.mu += self.mu_lr * mu_grad

            # --- Validation ---
            if self.validate_every > 0 and self.iteration_count % self.validate_every == 0:
                mode_v = self.flow_model.get_mode()
                new_genome = Genome()
                new_genome.special_metadata["flow_v"] = mode_v.detach().cpu().clone()

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

            # --- Checkpoint ---
            if self.checkpoint_every > 0 and self.iteration_count % self.checkpoint_every == 0:
                base, ext = os.path.splitext(self.checkpoint_path)
                path = f"{base}_t{self.trial_number}_step_{self.iteration_count}{ext}"
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                torch.save(self.flow_model.state_dict(), path)

            self.genomes = [Genome() for _ in range(self.total_pop)]

        # Save final model at end of training
        self.save_final_model(f"checkpoints/final_model_t{self.trial_number}")

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
                }, step=self.iteration_count)
            except Exception:
                pass
        print(f"#-- Stats: average: {reward_mean:.4f}, min: {worst_genome.historical_rewards[-1]:.4f}, "
              f"max: {best_genome.historical_rewards[-1]:.4f}, stddev: {reward_std:.4f}, average response length: {average_response_length:.2f} --#")
        if self.print_samples:
            print(f"#-- SAMPLE RESPONSE BEST GENOME: --#\n{best_genome.latest_outputs[0]}\n")
            print(f"#-- SAMPLE RESPONSE WORST GENOME: --#\n{worst_genome.latest_outputs[0]}\n")

    def log_val_stats(self, genome: Genome, time_taken: float):
        score = genome.historical_rewards[-1]
        score_stddev = (sum((genome.latest_rewards[i] - score) ** 2 for i in range(len(genome.latest_rewards))) / (len(genome.latest_rewards)-1)) ** 0.5 if len(genome.latest_rewards) > 1 else 0
        average_response_length = sum(len(response.split()) for response in genome.latest_outputs) / len(genome.latest_outputs)
        sample_response = genome.latest_outputs[0]
        if self.wandb_project:
            try:
                wandb.log({
                    "val/validation_score": score,
                    "val/validation_stddev": score_stddev,
                    "val/time_seconds": time_taken,
                    "val/average_response_length": average_response_length,
                    "val/sample_response": wandb.Table(data=[[sample_response]], columns=["response"]),
                    "iteration_count": self.iteration_count
                }, step=self.iteration_count)
            except Exception:
                pass
        print(f"#-- Mode Validation Stats: reward: {score:.4f}, response length: {average_response_length:.2f}, time: {time_taken:.2f}s --#")
        if self.print_samples:
            print(f"#-- SAMPLE RESPONSE: --#\n{sample_response}\n")


class XNESTrainer:
    """Separable (diagonal covariance) xNES trainer for TinyLoRA."""

    def __init__(self, config, backend, dataset,
                 mu_lr=0.1, sigma_lr=0.1, cov_lr=0.1, target_sigma=0.1,
                 lora_rank=2, u_dim=8, n_tie=1,
                 trial_number: int = 0,
                 wandb_project=None, wandb_project_name=None, validate_every=0,
                 print_samples=False, min_val_reward=0.25, start_prune_min_reward_iter=10,
                 checkpoint_every=0, checkpoint_path="checkpoints/xnes_model.pth"):

        self.config = config
        backend.startup(self.config)
        self.backend = backend
        self.dataset = dataset
        self.mu_lr = mu_lr
        self.sigma_lr = sigma_lr
        self.cov_lr = cov_lr
        self.target_sigma = target_sigma
        self.lora_rank = lora_rank
        self.u_dim = u_dim
        self.n_tie = n_tie
        self.trial_number = trial_number
        self.wandb_project = wandb_project
        self.validate_every = validate_every
        self.print_samples = print_samples
        self.min_val_reward = min_val_reward
        self.start_prune_min_reward_iter = start_prune_min_reward_iter
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

        print(f"#-- Computing TinyLoRA SVD (lora_rank={self.lora_rank}) --#")
        svd_info = self.backend.compute_tinylora_svd(self.lora_rank)
        print(f"#-- SVD computed for {len(svd_info)} modules --#")

        self.num_modules = len(svd_info)
        self.num_groups = math.ceil(self.num_modules / self.n_tie)
        self.dim_flow = self.u_dim * self.num_groups

        print(f"#-- TinyLoRA (sep-xNES): {self.num_modules} modules, {self.num_groups} tied groups, "
              f"u_dim={self.u_dim}, dim_flow={self.dim_flow} --#")

        self.backend.init_tinylora(self.u_dim, self.n_tie, self.lora_rank)

        self.mu = torch.zeros(self.dim_flow)
        self.sigma = torch.tensor(self.target_sigma, dtype=torch.float32)
        self.D = torch.ones(self.dim_flow)

        self.num_directions = self.config.population_size
        self.total_pop = self.num_directions * 2 if self.config.mirror else self.num_directions
        if self.config.centered_eval:
            self.total_pop += 1

        self.genomes = [Genome() for _ in range(self.total_pop)]
        self.iteration_count = 0

        if self.wandb_project:
            try:
                wandb.init(project=self.wandb_project, config={
                    "algo": "sep-xNES", "lora_rank": lora_rank, "u_dim": u_dim, "n_tie": n_tie,
                    "num_modules": self.num_modules, "num_groups": self.num_groups, "dim_flow": self.dim_flow,
                    "use_rslora": backend.use_rslora, "normalize_svd": backend.normalize_svd,
                }, name=wandb_project_name)
                wandb.define_metric("iteration_count")
                wandb.define_metric("train/*", step_metric="iteration_count")
                wandb.define_metric("val/*", step_metric="iteration_count")
            except Exception:
                self.wandb_project = None

    def save_final_model(self, filepath: str):
        mode_v = self.mu
        self.backend.apply_mode_permanently(mode_v.detach().cpu().clone())
        self.backend.save_weights_to_disk(filepath)

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

            sqrt_D = torch.sqrt(self.D)
            v = self.mu + self.sigma * sqrt_D * epsilon

            for i, genome in enumerate(self.genomes):
                genome.special_metadata["flow_v"] = v[i].detach().cpu().clone()

            inputs = self.dataset.next(
                population_size=self.num_directions, mirror=self.config.mirror, center=self.config.centered_eval
            )
            self.backend.generate_outputs(self.genomes, None, self.dataset.suffix, inputs)
            self.dataset.score_all(self.genomes)

            rewards = np.array([g.historical_rewards[-1] for g in self.genomes])
            r_mean = np.mean(rewards)
            r_std = np.std(rewards)

            self.log_train_stats(self.genomes, time.time() - start_time, r_mean, r_std)

            rewards_for_update = rewards.copy()
            if getattr(self.config, "rank_norm_rewards", False):
                rewards_for_update = centered_ranks(rewards_for_update)

            r1 = rewards_for_update[:self.num_directions]
            if self.config.mirror:
                r2 = rewards_for_update[self.num_directions:2*self.num_directions]
                adv_mu = (r1 - r2) / 2.0
                adv_sigma = ((r1 + r2) / 2.0 - np.mean(rewards_for_update))
            else:
                adv_mu = r1 - np.mean(rewards_for_update)
                adv_sigma = r1 - np.mean(rewards_for_update)

            adv_mu = (adv_mu - adv_mu.mean()) / (adv_mu.std() + 1e-8)
            adv_sigma = (adv_sigma - adv_sigma.mean()) / (adv_sigma.std() + 1e-8)

            adv_mu_t = torch.from_numpy(adv_mu).float()
            adv_sigma_t = torch.from_numpy(adv_sigma).float()

            with torch.no_grad():
                eps = epsilon[:self.num_directions]
                adv = adv_sigma_t

                mu_grad = (adv_mu_t.unsqueeze(1) * eps).mean(dim=0)
                self.mu += self.mu_lr * self.sigma * sqrt_D * mu_grad

                norms = (eps ** 2).sum(dim=-1) / self.dim_flow - 1.0
                sigma_grad = 0.5 * (adv * norms).mean()
                self.sigma *= torch.exp(self.sigma_lr * sigma_grad)
                self.sigma = torch.clamp(self.sigma, min=1e-4, max=0.5)

                d_grad = 0.5 * (adv.unsqueeze(1) * (eps ** 2 - 1.0)).mean(dim=0)
                log_D = torch.log(self.D)
                log_D += self.cov_lr * d_grad
                log_D -= log_D.mean()  # Prevent global scale leakage into D
                self.D = torch.exp(log_D).clamp(1e-6, 10.0)

            if self.wandb_project:
                wandb.log({
                    "train/sigma": self.sigma.item(),
                    "train/D_mean": self.D.mean().item(),
                    "train/D_std": self.D.std().item(),
                    "iteration_count": self.iteration_count
                }, step=self.iteration_count)

            # --- Validation ---
            if self.validate_every > 0 and self.iteration_count % self.validate_every == 0:
                mode_v = self.mu
                new_genome = Genome()
                new_genome.special_metadata["flow_v"] = mode_v.detach().cpu().clone()

                prompts = self.dataset.get_test_set()
                self.backend.generate_outputs([new_genome], None, self.dataset.suffix, prompts)
                self.dataset.score_all([new_genome])
                latest_val_score = new_genome.historical_rewards[-1]

                self.log_val_stats(new_genome, 0.0)

                optuna_trial.report(latest_val_score, self.iteration_count)
                if self.iteration_count >= self.start_prune_min_reward_iter and latest_val_score < self.min_val_reward:
                    raise optuna.TrialPruned()
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            # --- Checkpoint ---
            if self.checkpoint_every > 0 and self.iteration_count % self.checkpoint_every == 0:
                base, ext = os.path.splitext(self.checkpoint_path)
                path = f"{base}_t{self.trial_number}_step_{self.iteration_count}{ext}"
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                torch.save({
                    "mu": self.mu, "sigma": self.sigma, "D": self.D,
                }, path)

            self.genomes = [Genome() for _ in range(self.total_pop)]

        # Save final model at end of training
        self.save_final_model(f"checkpoints/final_model_t{self.trial_number}")

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
                }, step=self.iteration_count)
            except Exception:
                pass
        print(f"#-- Stats: average: {reward_mean:.4f}, min: {worst_genome.historical_rewards[-1]:.4f}, "
              f"max: {best_genome.historical_rewards[-1]:.4f}, stddev: {reward_std:.4f}, average response length: {average_response_length:.2f} --#")
        if self.print_samples:
            print(f"#-- SAMPLE RESPONSE BEST GENOME: --#\n{best_genome.latest_outputs[0]}\n")
            print(f"#-- SAMPLE RESPONSE WORST GENOME: --#\n{worst_genome.latest_outputs[0]}\n")

    def log_val_stats(self, genome: Genome, time_taken: float):
        score = genome.historical_rewards[-1]
        score_stddev = (sum((genome.latest_rewards[i] - score) ** 2 for i in range(len(genome.latest_rewards))) / (len(genome.latest_rewards)-1)) ** 0.5 if len(genome.latest_rewards) > 1 else 0
        average_response_length = sum(len(response.split()) for response in genome.latest_outputs) / len(genome.latest_outputs)
        sample_response = genome.latest_outputs[0]
        if self.wandb_project:
            try:
                wandb.log({
                    "val/validation_score": score,
                    "val/validation_stddev": score_stddev,
                    "val/time_seconds": time_taken,
                    "val/average_response_length": average_response_length,
                    "val/sample_response": wandb.Table(data=[[sample_response]], columns=["response"]),
                    "iteration_count": self.iteration_count
                }, step=self.iteration_count)
            except Exception:
                pass
        print(f"#-- Mode Validation Stats: reward: {score:.4f}, response length: {average_response_length:.2f}, time: {time_taken:.2f}s --#")
        if self.print_samples:
            print(f"#-- SAMPLE RESPONSE: --#\n{sample_response}\n")