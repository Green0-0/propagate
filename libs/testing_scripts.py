import math
from typing import List, Dict, Optional
from collections import OrderedDict

import torch
import numpy as np

from libs.genome import Genome
from libs.optimizers import Optimizer

class exp(Optimizer):
    def __init__(
        self,
        total_steps: int,
        learning_rate: float,
        seed_weight: float,
        warmup_steps: int = 0,
        scheduler: str = "none",
        lambda_reg: float = 1e-3,
        momentum: float = 0.9,
        cutoff_seeds: int = 2000,
        sketch_size: int = 512,
        max_update_norm: Optional[float] = None,  # optional trust-region on coefficient norm
        device: str = "cpu",
    ):
        super().__init__(total_steps, learning_rate, seed_weight, warmup_steps, scheduler, optimizer_name="SeedSpaceRidgeOptimizer")
        self.lambda_reg = float(lambda_reg)
        self.momentum = float(momentum)
        self.velocity_coeffs: "OrderedDict[int, float]" = OrderedDict()
        self.cutoff_seeds = int(cutoff_seeds)
        self.sketch_size = int(sketch_size)
        self.device = device
        self.max_update_norm = None if max_update_norm is None else float(max_update_norm)

        # Pre-allocate a generator seed salt to make sketches stable across runs if desired.
        # (Optional) You can change sketch_salt if you want a different sketch family.
        self.sketch_salt = 0

    def _make_sketch(self, seed: int) -> torch.Tensor:
        """
        Generate a deterministic sketch vector for a given integer seed.
        Uses torch.Generator seeded with (seed xor salt).
        Returns a sketch_size float64 tensor on CPU (device can be changed).
        """
        gen = torch.Generator(device="cpu")
        # XOR with salt to allow controlled variation of sketches if needed
        gen.manual_seed(int(seed) ^ int(self.sketch_salt))
        # Use float64 for better numeric stability in Gram; convert to device later if needed
        sketch = torch.randn(self.sketch_size, dtype=torch.float64, generator=gen) * self.seed_weight
        return sketch

    def _build_gram_and_z(self, seeds_list: List[int], seed_rewards: List[float]) -> (torch.Tensor, torch.Tensor):
        """
        Build Gram matrix G and standardized reward vector z for the given lists.
        seeds_list: list of seed ids (length n)
        seed_rewards: list of scalar rewards assigned to each seed (length n)
        Returns:
            G: (n, n) torch.float64 tensor
            z: (n,) torch.float64 tensor (standardized)
        """
        n = len(seeds_list)
        if n == 0:
            return torch.empty((0, 0), dtype=torch.float64), torch.empty((0,), dtype=torch.float64)

        # Build sketches matrix (n x sketch_size)
        sketches = torch.empty((n, self.sketch_size), dtype=torch.float64)
        for i, s in enumerate(seeds_list):
            sketches[i] = self._make_sketch(s)

        # Compute Gram = sketches @ sketches^T (n x n)
        # Note: Gram_ij approximates s_i^T s_j in the full parameter space.
        G = sketches @ sketches.T  # (n x n) float64

        # Standardize rewards to zero-mean, unit-std (matches earlier algorithm)
        z_np = np.array(seed_rewards, dtype=np.float64)
        mean = float(np.mean(z_np)) if z_np.size > 0 else 0.0
        std = float(np.std(z_np, ddof=0))  # population std (match earlier code)
        if std <= 0:
            std = 1.0
        z = torch.from_numpy((z_np - mean) / (std + 1e-12)).to(dtype=torch.float64)

        return G, z

    def _solve_coefficients(self, G: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Solve (G + lambda I) w = z for w, using stable torch solvers.
        G: (n, n) float64, z: (n,)
        Returns w: (n,) float64 tensor.
        """
        if G.numel() == 0:
            return torch.empty((0,), dtype=torch.float64)

        n = G.shape[0]
        # regularize
        diag = torch.full((n,), float(self.lambda_reg), dtype=torch.float64)
        A = G + torch.diag(diag)

        # Use Cholesky if positive definite; fallback to solve.
        try:
            chol = torch.linalg.cholesky(A)
            w = torch.cholesky_solve(z.unsqueeze(1), chol).squeeze(1)
        except RuntimeError:
            # fallback to least squares / solve
            try:
                w = torch.linalg.solve(A, z)
            except RuntimeError:
                # As a last resort, use pinv
                A_np = A.cpu().numpy()
                z_np = z.cpu().numpy()
                w_np = np.linalg.pinv(A_np) @ z_np
                w = torch.from_numpy(w_np).to(dtype=torch.float64)
        return w

    def get_step(self, genomes: List[Genome], current_step: int) -> Genome:
        """
        Build merged Genome representing the SSR update. This function:
        - collects new seeds (i >= starting_index) and assigns each the genome's reward
        - builds Gram via sketches, solves ridge regression for coefficients w
        - scales coefficients by lr*(1/num_genomes) to match NES scale
        - applies coefficient momentum, trims velocity buffer
        - returns merged Genome that contains (1) averaged old seeds and (2) velocity seeds (with coefficient weights)
        """
        lr = self.get_lr(current_step)

        # Decay existing velocity coefficients
        if len(self.velocity_coeffs) > 0 and self.momentum != 0.0:
            for sid in list(self.velocity_coeffs.keys()):
                self.velocity_coeffs[sid] *= self.momentum

        # Compute population reward mean/std (per-Genome)
        rewards = [g.historical_rewards[-1] for g in genomes]
        reward_mean = sum(rewards) / len(rewards)
        reward_stddev = (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
        if reward_stddev <= 0:
            reward_stddev = 1.0

        # Collect old seeds (absorbed earlier) and count/aggregate them
        old_seeds: Dict[int, float] = {}
        old_seeds_count: Dict[int, int] = {}

        # Collect "new" seeds and their per-seed reward entries
        new_seeds: List[int] = []
        new_seeds_rewards: List[float] = []

        # For each genome, seeds with index >= starting_index are "new" this generation.
        for g in genomes:
            g_reward = g.historical_rewards[-1]
            for i, seed in enumerate(g.seeds):
                weight = g.seed_weights[i]
                if i < g.starting_index:
                    # seed from previous backprop step; aggregate
                    if seed not in old_seeds:
                        old_seeds[seed] = float(weight)
                        old_seeds_count[seed] = 1
                    else:
                        old_seeds[seed] += float(weight)
                        old_seeds_count[seed] += 1
                else:
                    # new seed(s); assign the genome's reward to each new seed it carries
                    new_seeds.append(seed)
                    # standardized reward (without lr scaling); we'll standardize again in _build_gram_and_z
                    # Use the raw reward value so standardization is consistent across all new_seeds
                    new_seeds_rewards.append(float(g_reward))

        # Average old seeds' aggregated weights
        for seed, cnt in old_seeds_count.items():
            old_seeds[seed] = old_seeds[seed] / float(cnt)

        # If no new seeds were produced (rare), just return merged genome using existing velocity & old seeds
        if len(new_seeds) == 0:
            # Trim velocity buffer
            if len(self.velocity_coeffs) > self.cutoff_seeds:
                # keep most recent items
                items = list(self.velocity_coeffs.items())[-self.cutoff_seeds:]
                self.velocity_coeffs = OrderedDict(items)

            merged = Genome()
            # add old seeds
            for seed, weight in old_seeds.items():
                merged.seeds.append(seed)
                merged.seed_weights.append(weight)
                merged.historical_rewards.append(float("-inf"))
            # add velocity seeds
            for seed, coeff in self.velocity_coeffs.items():
                merged.seeds.append(seed)
                merged.seed_weights.append(float(coeff))
                merged.historical_rewards.append(float("-inf"))
            merged.starting_index = len(merged.seeds)
            return merged

        # Build Gram and standardized z (z = (r - mean)/std)
        G, z = self._build_gram_and_z(new_seeds, new_seeds_rewards)

        # Solve (G + lambda I) w = z
        w = self._solve_coefficients(G, z)  # float64 torch tensor, shape (n,)

        # Scale coefficients:
        # earlier NES used factor (1/N) * learning_rate * standardized_reward * seed
        # we scale w by learning rate * (1/len(genomes)) to keep similar magnitude
        scale = float(lr) * (1.0 / max(1, len(genomes)))
        w = w * scale  # still float64

        # Optionally clip coefficient vector norm (trust region)
        if self.max_update_norm is not None:
            norm = torch.linalg.norm(w).item()
            if norm > 0 and norm > self.max_update_norm:
                w = w * (self.max_update_norm / (norm + 1e-12))

        # Apply momentum into velocity_coeffs (we already decayed existing entries above)
        # Add coefficients for the new seeds
        for sid, coeff in zip(new_seeds, w.tolist()):
            if sid in self.velocity_coeffs:
                self.velocity_coeffs[sid] += float(coeff)
            else:
                self.velocity_coeffs[sid] = float(coeff)

        # Trim velocity buffer to keep memory bounded
        if len(self.velocity_coeffs) > self.cutoff_seeds:
            items = list(self.velocity_coeffs.items())[-self.cutoff_seeds:]
            self.velocity_coeffs = OrderedDict(items)

        # Build merged genome: include averaged old seeds and velocity seeds
        merged = Genome()
        for seed, weight in old_seeds.items():
            merged.seeds.append(seed)
            merged.seed_weights.append(weight)
            merged.historical_rewards.append(float("-inf"))
        for seed, coeff in self.velocity_coeffs.items():
            merged.seeds.append(seed)
            merged.seed_weights.append(coeff)  # coefficients are the actual update multipliers
            merged.historical_rewards.append(float("-inf"))

        merged.starting_index = len(merged.seeds)
        return merged
