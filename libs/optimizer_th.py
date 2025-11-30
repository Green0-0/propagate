from abc import ABC, abstractmethod
import copy
from typing import Any, List, Dict, Tuple
from collections import OrderedDict

import torch

from libs.genome import Genome
from libs.optimizers import Optimizer

import math

class TwoHalvesEstimatorOpt(Optimizer):
    def __init__(self, total_steps: int, learning_rate: float, perturb_scale: float, warmup_steps: int = 0, scheduler: str = "none", norm_by_mean: bool = True, norm_by_stddev: bool = True, force_lora_alternating: bool = False, ema_decay: float = 0.9, tau: float = 1.0, epsilon: float = 1e-8, momentum: float = 0.9, gamma: float = 1, cutoff_steps: int = 20):
        super().__init__(f"TwoHalvesEstimatorOpt (ema_decay={ema_decay}, tau={tau}, epsilon={epsilon}, momentum={momentum}, gamma={gamma}, cutoff_steps={cutoff_steps})", total_steps, learning_rate, perturb_scale, warmup_steps, scheduler, norm_by_mean, norm_by_stddev, force_lora_alternating)
        self.rep_genome_A = None
        self.rep_genome_B = None
        self.current_step_lr = 0.0
        self.ema_decay = ema_decay
        self.tau = tau
        self.epsilon = epsilon

        self.momentum = momentum
        self.gamma = gamma
        self.cutoff_steps = cutoff_steps
        
        self.velocity_seeds_steps = []
        
        self.total_layers_updated = 0
        self.max_beff = 0.0
        self.min_beff = float('inf')
        self.max_beff_param_id = None
        self.min_beff_param_id = None
        self.total_beff = 0.0

        self.total_v = 0.0
        self.total_d = 0.0
        self.max_v = 0.0
        self.min_v = float('inf')
        self.max_v_param_id = None
        self.min_v_param_id = None
        self.max_d = 0.0
        self.min_d = float('inf')
        self.max_d_param_id = None
        self.min_d_param_id = None

    def get_representative(self) -> Genome:
        raise NotImplementedError("TwoHalvesEstimatorOpt does not support a single representative genome.")

    def _create_gradient_genome(self, genome_list: List[Genome], reward_mean: float, reward_stddev: float) -> Genome:
        rep = Genome()
        new_seeds = {}
        old_seeds_count = {}

        for g in genome_list:
            added_old_seed = False
            for i in range(len(g.seeds)):
                seed = g.seeds[i]
                weight = g.perturb_scales[i]
                if seed not in new_seeds:
                    if i < g.starting_index:
                        added_old_seed = True
                        old_seeds_count[seed] = 1
                        new_seeds[seed] = weight
                    else:
                        update_value = 0.0
                        if self.norm_by_mean:
                            update_value = math.copysign(1, weight) * (1/len(genome_list)) * (g.historical_rewards[-1] - reward_mean)
                        else:
                            update_value = math.copysign(1, weight) * (1/len(genome_list)) * g.historical_rewards[-1]
                        if self.norm_by_stddev:
                            update_value /= (reward_stddev + 1e-8)
                        new_seeds[seed] = update_value
                else:
                    if i < g.starting_index:
                        if not added_old_seed:
                            added_old_seed = True
                            old_seeds_count[seed] += 1
                        new_seeds[seed] += weight
                    else:
                        update_value = 0.0
                        if self.norm_by_mean:
                            update_value = math.copysign(1, weight) * (1/len(genome_list)) * (g.historical_rewards[-1] - reward_mean)
                        else:
                            update_value = math.copysign(1, weight) * (1/len(genome_list)) * g.historical_rewards[-1]
                        if self.norm_by_stddev:
                            update_value /= (reward_stddev + 1e-8)
                        new_seeds[seed] += update_value

        for seed, count in old_seeds_count.items():
            new_seeds[seed] /= count

        for seed, weight in new_seeds.items():
            rep.seeds.append(seed)
            rep.perturb_scales.append(weight)
            rep.historical_rewards.append(float('-inf'))
        rep.starting_index = len(rep.seeds)
        return rep

    def update_self(self, genomes: List[Genome], current_step: int):
        self.current_step_lr = self.get_lr(current_step)

        all_rewards = [g.historical_rewards[-1] for g in genomes]
        reward_mean = sum(all_rewards) / len(genomes)
        reward_stddev = (sum([(r - reward_mean) ** 2 for r in all_rewards]) / len(genomes)) ** 0.5

        groups = {}
        for g in genomes:
            key = tuple(g.seeds)
            if key not in groups:
                groups[key] = []
            groups[key].append(g)

        list_a = []
        list_b = []
        
        group_keys = list(groups.keys())
        for i, key in enumerate(group_keys):
            group_genomes = groups[key]
            if i % 2 == 0:
                list_a.extend(group_genomes)
            else:
                list_b.extend(group_genomes)

        self.rep_genome_A = self._create_gradient_genome(list_a, reward_mean, reward_stddev)
        self.rep_genome_B = self._create_gradient_genome(list_b, reward_mean, reward_stddev)
        
        self.update_history.append((copy.deepcopy(self.rep_genome_A), copy.deepcopy(self.rep_genome_B)))

        current_step_seeds = {}
        for seed, weight in zip(self.rep_genome_A.seeds, self.rep_genome_A.perturb_scales):
            if seed not in current_step_seeds: current_step_seeds[seed] = 0.0
            current_step_seeds[seed] += (weight * 0.5)
            
        for seed, weight in zip(self.rep_genome_B.seeds, self.rep_genome_B.perturb_scales):
            if seed not in current_step_seeds: current_step_seeds[seed] = 0.0
            current_step_seeds[seed] += (weight * 0.5)

        current_step_seeds_list = [(seed, weight) for seed, weight in current_step_seeds.items()]
        
        self.velocity_seeds_steps.append(current_step_seeds_list)
        if len(self.velocity_seeds_steps) > self.cutoff_steps:
            self.velocity_seeds_steps.pop(0)

    def step_update(self, tensor: torch.Tensor, random_offset: int, parameter_id, lr_scalar: float = 1, state: Dict = None):
        self.total_layers_updated += 1
        if self.total_layers_updated == 3000:
            print(f"Total layers updated in last step: {self.total_layers_updated}")
            print(f"Beff - Max: {self.max_beff} (Param ID: {self.max_beff_param_id}), Min: {self.min_beff} (Param ID: {self.min_beff_param_id}), Avg: {self.total_beff / max(1, self.total_layers_updated)}")
            print(f"V - Max: {self.max_v} (Param ID: {self.max_v_param_id}), Min: {self.min_v} (Param ID: {self.min_v_param_id}), Avg: {self.total_v / max(1, self.total_layers_updated)}")
            print(f"D - Max: {self.max_d} (Param ID: {self.max_d_param_id}), Min: {self.min_d} (Param ID: {self.min_d_param_id}), Avg: {self.total_d / max(1, self.total_layers_updated)}")

        gen = torch.Generator(device=tensor.device)
        
        noise_buffer = torch.empty_like(tensor)

        grad_a = torch.zeros_like(tensor)
        for seed, weight in zip(self.rep_genome_A.seeds, self.rep_genome_A.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise_buffer)
            grad_a.add_(noise_buffer, alpha=float(weight))

        grad_b = torch.zeros_like(tensor)
        for seed, weight in zip(self.rep_genome_B.seeds, self.rep_genome_B.perturb_scales):
            gen.manual_seed(int(seed) + random_offset)
            torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise_buffer)
            grad_b.add_(noise_buffer, alpha=float(weight))

        flat_A = grad_a.flatten()
        flat_B = grad_b.flatten()
        
        S_l = torch.dot(flat_A, flat_B).item()
        
        norm_A2 = torch.sum(flat_A.square()).item()
        norm_B2 = torch.sum(flat_B.square()).item()
        M_l = 0.5 * (norm_A2 + norm_B2)
        V_l = max(M_l - S_l, 0.0)
        if "noise_variance_ema" not in state or state["noise_variance_ema"] is None:
            state["noise_variance_ema"] = {}
        noise_variance_ema = state["noise_variance_ema"]
        if parameter_id not in noise_variance_ema:
            noise_variance_ema[parameter_id] = V_l
        else:
            noise_variance_ema[parameter_id] = (self.ema_decay * noise_variance_ema[parameter_id]) + ((1 - self.ema_decay) * V_l)

        v_smooth = noise_variance_ema[parameter_id]
        denom = S_l + (self.tau * v_smooth) + self.epsilon
        phi_l = S_l / denom if denom > 1e-12 else 0.0
        phi_l = max(0.0, min(1.0, phi_l))
        d_l = phi_l / math.sqrt(v_smooth + self.epsilon)

        # Record statistics
        self.total_d += d_l
        if d_l > self.max_d:
            self.max_d = d_l
            self.max_d_param_id = parameter_id
        if d_l < self.min_d:
            self.min_d = d_l
            self.min_d_param_id = parameter_id
        self.total_v += V_l
        if V_l > self.max_v:
            self.max_v = V_l
            self.max_v_param_id = parameter_id
        if V_l < self.min_v:
            self.min_v = V_l
            self.min_v_param_id = parameter_id

        if "layer_scales_history" not in state or state["layer_scales_history"] is None:
            state["layer_scales_history"] = {}
        layer_scales_history = state["layer_scales_history"]
        if parameter_id not in layer_scales_history:
            layer_scales_history[parameter_id] = []
        target_len = len(self.velocity_seeds_steps)
        while len(layer_scales_history[parameter_id]) < (target_len - 1):
            layer_scales_history[parameter_id].append(0.0)
        layer_scales_history[parameter_id].append(d_l)
        while len(layer_scales_history[parameter_id]) > len(self.velocity_seeds_steps):
            layer_scales_history[parameter_id].pop(0)

        grad_a.add_(grad_b).mul_(0.5).mul(d_l)
        
        m = torch.zeros_like(tensor)
        history_len = len(self.velocity_seeds_steps)
        for step_idx in reversed(range(history_len - 1)):
            step_scale = layer_scales_history[parameter_id][step_idx]
            if abs(step_scale) < 1e-9: 
                continue
            step_seeds = self.velocity_seeds_steps[step_idx]
            for seed, weight in step_seeds:
                gen.manual_seed(int(seed) + random_offset)
                torch.randn(tensor.shape, generator=gen, device=tensor.device, dtype=tensor.dtype, out=noise_buffer)

                final_w = weight * step_scale
                m.add_(noise_buffer, alpha=final_w)
        
        if history_len > 1:
            flat_p = grad_a.flatten()
            flat_m = m.flatten()
            
            norm_p = torch.norm(flat_p).item()
            norm_m = torch.norm(flat_m).item()
            
            dot_prod = torch.dot(flat_p, flat_m).item()
            
            if norm_p > 1e-12 and norm_m > 1e-12:
                cos = dot_prod / (norm_p * norm_m)
            else:
                cos = 0.0
                
            beta_eff = self.momentum * (max(0.0, cos) ** self.gamma)
            for i in range(history_len - 1):
                if abs(layer_scales_history[parameter_id][i]) < 1e-9: 
                    continue
                layer_scales_history[parameter_id][i] *= beta_eff
            del flat_p
            del flat_m
        else:
            beta_eff = 0.0
        
        # Record beff statistics
        self.total_beff += beta_eff
        if beta_eff > self.max_beff:
            self.max_beff = beta_eff
            self.max_beff_param_id = parameter_id
        if beta_eff < self.min_beff:
            self.min_beff = beta_eff
            self.min_beff_param_id = parameter_id

        grad_a.add(m, alpha=beta_eff)
        tensor.add_(grad_a, alpha=self.current_step_lr * lr_scalar)

        del grad_a
        del grad_b
        del flat_A
        del flat_B
        del noise_buffer
        del m
    
    def get_update_history(self) -> Any:
        return self.update_history

    def restore_from_history(self, history, backend):
        raise NotImplementedError("TwoHalvesEstimator does not support restoring from history.")