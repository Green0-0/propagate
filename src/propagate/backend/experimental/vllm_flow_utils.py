import torch
import math
import re
from typing import Dict, List
from propagate.genome import Genome
from propagate.backend.vllm_lorautils import WorkerExtension as BaseWorkerExtension

MODULE_ORDER = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "o_proj": 3,
    "gate_proj": 4,
    "up_proj": 5,
    "down_proj": 6,
}

def tinylora_sort_key(name: str):
    m = re.search(r"layers\.(\d+)\.", name)
    layer_idx = int(m.group(1)) if m else 10**9
    suffix = name.split(".")[-1]
    return layer_idx, MODULE_ORDER.get(suffix, 99), name

class FlowWorkerExtension(BaseWorkerExtension):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tinylora_layer_names = []
        self.tinylora_U = []
        self.tinylora_S = []
        self.tinylora_V = []
        self.tinylora_P = []
        self.tinylora_orientations = []
        self.tinylora_u_dim = 0
        self.tinylora_n_tie = 1
        self.tinylora_lora_rank = 0
        self.tinylora_num_groups = 0
        self.tinylora_normalize_svd = False
        self._gpu_cache = None

    @torch.inference_mode()
    def compute_tinylora_svd(self, lora_rank: int, normalize_svd: bool = False):
        """Computes truncated SVD of base weights for TinyLoRA and caches it on the worker."""
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager
        adapters_dict = adapter_manager.list_adapters()
        if not adapters_dict:
            return []
        aid, _ = sorted(adapters_dict.items(), key=lambda x: x[0])[0]

        lora_weights = self._collect_gpu_lora_tensors(aid)
        model = self.model_runner.model

        named_modules = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                named_modules[name] = module
                if hasattr(module, 'base_layer') and hasattr(module.base_layer, 'weight'):
                    named_modules[name + '.base_layer'] = module.base_layer

        module_dims = {}
        for layer_name, (lora_a, lora_b) in lora_weights.items():
            clean_name = layer_name
            for prefix in ['base_model.model.', 'model.', 'base_model.']:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
                    break
            parent_name = '.'.join(clean_name.split('.')[:-1])
            suffix = clean_name.split('.')[-1]
            if suffix in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']:
                if lora_a.shape[0] == lora_rank and lora_b.shape[1] == lora_rank:
                    d_out = lora_b.shape[0]
                elif lora_a.shape[1] == lora_rank and lora_b.shape[0] == lora_rank:
                    d_out = lora_a.shape[0]
                else:
                    continue
                if parent_name not in module_dims:
                    module_dims[parent_name] = {}
                module_dims[parent_name][suffix] = d_out

        self.tinylora_layer_names = []
        self.tinylora_U = []
        self.tinylora_S = []
        self.tinylora_V = []
        self.tinylora_orientations = []
        self.tinylora_normalize_svd = normalize_svd
        self.tinylora_lora_rank = lora_rank
        self._gpu_cache = None
        svd_info = []

        # Natural sort to ensure contiguous chunks correspond to logical transformer blocks
        for layer_name, (lora_a, lora_b) in sorted(lora_weights.items(), key=lambda kv: tinylora_sort_key(kv[0])):
            if lora_a.shape[0] == lora_rank and lora_b.shape[1] == lora_rank:
                d_in = lora_a.shape[1]
                d_out = lora_b.shape[0]
                orientation = "peft"
            elif lora_a.shape[1] == lora_rank and lora_b.shape[0] == lora_rank:
                d_in = lora_b.shape[1]
                d_out = lora_a.shape[0]
                orientation = "vllm_swapped"
            else:
                raise RuntimeError(f"Cannot infer LoRA orientation for {layer_name}")

            base_weight = None
            clean_name = layer_name
            for prefix in ['base_model.model.', 'model.', 'base_model.']:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
                    break

            target_module = None
            if clean_name in named_modules:
                target_module = named_modules[clean_name]
            else:
                for name, mod in named_modules.items():
                    if name.endswith('.' + clean_name) or name == clean_name:
                        target_module = mod
                        break

            if target_module is not None:
                if hasattr(target_module, 'base_layer') and hasattr(target_module.base_layer, 'weight'):
                    base_weight = target_module.base_layer.weight.data
                elif hasattr(target_module, 'weight'):
                    base_weight = target_module.weight.data

            if base_weight is None:
                base_suffix = clean_name.split('.')[-1]
                parent_name = '.'.join(clean_name.split('.')[:-1])
                merged_name = None
                order = []
                if base_suffix in ['q_proj', 'k_proj', 'v_proj']:
                    merged_name = parent_name + '.qkv_proj'
                    order = ['q_proj', 'k_proj', 'v_proj']
                elif base_suffix in ['gate_proj', 'up_proj']:
                    merged_name = parent_name + '.gate_up_proj'
                    order = ['gate_proj', 'up_proj']
                if merged_name and merged_name in named_modules:
                    merged_mod = named_modules[merged_name]
                    W_merged = merged_mod.weight.data
                    offset = 0
                    for s in order:
                        if s == base_suffix:
                            break
                        offset += module_dims.get(parent_name, {}).get(s, 0)
                    chunk = d_out
                    base_weight = W_merged[offset : offset + chunk, :]

            if base_weight is None:
                raise RuntimeError(f"Could not find base weight for {layer_name}")

            W = base_weight.float()
            if W.shape[0] != d_out or W.shape[1] != d_in:
                if W.shape[0] == d_in and W.shape[1] == d_out:
                    W = W.T
                else:
                    raise RuntimeError(f"Shape mismatch for {layer_name}: W={W.shape}, expected ({d_out}, {d_in})")

            q = min(lora_rank + 4, min(W.shape[0], W.shape[1]))
            U, S, V = torch.svd_lowrank(W, q=q, niter=5)
            U = U[:, :lora_rank].contiguous().cpu()
            S = S[:lora_rank].contiguous().cpu()
            V = V[:, :lora_rank].contiguous().cpu()

            if normalize_svd and S.norm() > 0:
                S = S / S.norm()

            self.tinylora_layer_names.append(layer_name)
            self.tinylora_U.append(U)
            self.tinylora_S.append(S)
            self.tinylora_V.append(V)
            self.tinylora_orientations.append(orientation)

            svd_info.append({
                'name': layer_name,
                'orientation': orientation,
                'a_shape': tuple(lora_a.shape),
                'b_shape': tuple(lora_b.shape),
            })

        return svd_info

    def get_tinylora_svd_data(self):
        return {
            "names": self.tinylora_layer_names,
            "U": self.tinylora_U,
            "S": self.tinylora_S,
            "V": self.tinylora_V,
            "orientations": self.tinylora_orientations,
            "normalize_svd": self.tinylora_normalize_svd,
            "lora_rank": self.tinylora_lora_rank,
        }

    def set_tinylora_svd_data(self, data):
        self.tinylora_layer_names = data["names"]
        self.tinylora_U = data["U"]
        self.tinylora_S = data["S"]
        self.tinylora_V = data["V"]
        self.tinylora_orientations = data["orientations"]
        self.tinylora_normalize_svd = data["normalize_svd"]
        self.tinylora_lora_rank = data["lora_rank"]
        self._gpu_cache = None

    @torch.inference_mode()
    def init_tinylora(self, u_dim: int, n_tie: int, lora_rank: int):
        self.tinylora_u_dim = u_dim
        self.tinylora_n_tie = max(1, n_tie)
        self.tinylora_P = []
        self._gpu_cache = None
        self.tinylora_num_groups = math.ceil(len(self.tinylora_layer_names) / self.tinylora_n_tie)

        gen = torch.Generator().manual_seed(42)
        for _ in self.tinylora_layer_names:
            P = torch.randn(self.tinylora_u_dim, lora_rank, lora_rank, generator=gen) / math.sqrt(self.tinylora_u_dim)
            self.tinylora_P.append(P)

    def _ensure_gpu_cache(self, device, dtype):
        if self._gpu_cache is None or self._gpu_cache['device'] != device or self._gpu_cache['dtype'] != dtype:
            self._gpu_cache = {
                'device': device,
                'dtype': dtype,
                'U': [u.to(device, dtype=dtype) for u in self.tinylora_U],
                'S': [s.to(device, dtype=dtype) for s in self.tinylora_S],
                'V': [v.to(device, dtype=dtype) for v in self.tinylora_V],
                'P': [p.to(device, dtype=dtype) for p in self.tinylora_P],
            }
        return self._gpu_cache

    def update_original_weights_cache(self):
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager
        adapters_dict = adapter_manager.list_adapters()
        for aid, _ in adapters_dict.items():
            weights = self._collect_gpu_lora_tensors(aid)
            if not hasattr(self, '_original_weights_cache'):
                self._original_weights_cache = {}
            self._original_weights_cache[aid] = {k: (v[0].clone(), v[1].clone()) for k, v in weights.items()}

    @torch.inference_mode()
    def apply_mode_permanently(self, mode_v: torch.Tensor):
        adapters_dict = self.model_runner.lora_manager._adapter_manager.list_adapters()
        sorted_adapters = sorted(adapters_dict.items(), key=lambda x: x[0])
        num_adapters = len(sorted_adapters)
        
        dummy_genomes = [Genome() for _ in range(num_adapters)]
        for g in dummy_genomes:
            g.special_metadata["flow_v"] = mode_v.cpu()
            
        self.set_explicit_weights_multi(dummy_genomes, is_restore=False)
        self.update_original_weights_cache()

    @torch.inference_mode()
    def set_explicit_weights_multi(self, genomes: List[Genome], is_restore: bool = False):
        assert len(self.tinylora_layer_names) > 0, \
            "TinyLoRA SVD not computed. Call compute_tinylora_svd first."

        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager
        adapters_dict = adapter_manager.list_adapters()
        sorted_adapters = sorted(adapters_dict.items(), key=lambda x: x[0])

        if len(genomes) > len(sorted_adapters):
            raise ValueError(f"Received {len(genomes)} genomes but only {len(sorted_adapters)} adapters.")

        for i, genome in enumerate(genomes):
            aid, _ = sorted_adapters[i]
            weights = self._collect_gpu_lora_tensors(aid)

            if not hasattr(self, '_original_weights_cache'):
                self._original_weights_cache = {}
            if aid not in self._original_weights_cache:
                self._original_weights_cache[aid] = {k: (v[0].clone(), v[1].clone()) for k, v in weights.items()}

            if is_restore:
                for layer_name, (lora_a, lora_b) in weights.items():
                    orig_a, orig_b = self._original_weights_cache[aid][layer_name]
                    lora_a.copy_(orig_a)
                    lora_b.copy_(orig_b)
            else:
                v = genome.special_metadata.get("flow_v")
                if v is None:
                    raise ValueError(f"Genome {i} is missing 'flow_v'")
                v = v.cpu()
                
                expected_len = self.tinylora_u_dim * self.tinylora_num_groups
                assert len(v) == expected_len, f"flow_v length mismatch: got {len(v)}, expected {expected_len}"

                first_layer = next(iter(weights.keys()))
                device = weights[first_layer][0].device
                dtype = weights[first_layer][0].dtype
                
                # Keep SVD math in float32 for precision, cast only the final A/B candidates
                gpu_cache = self._ensure_gpu_cache(device, torch.float32)

                for module_idx, layer_name in enumerate(self.tinylora_layer_names):
                    lora_a, lora_b = weights[layer_name]

                    group_idx = module_idx // self.tinylora_n_tie
                    group_v = v[group_idx * self.tinylora_u_dim : (group_idx + 1) * self.tinylora_u_dim].to(device, dtype=torch.float32)

                    P = gpu_cache['P'][module_idx]
                    R = torch.einsum('u,urs->rs', group_v, P)

                    U = gpu_cache['U'][module_idx]
                    S = gpu_cache['S'][module_idx]
                    V = gpu_cache['V'][module_idx]
                    orientation = self.tinylora_orientations[module_idx]

                    U_sigma = U * S.unsqueeze(0)

                    if orientation == "peft":
                        B_cand = U_sigma @ R
                        A_cand = V.T
                    else:
                        A_cand = U_sigma @ R
                        B_cand = V.T

                    lora_a.copy_(A_cand.to(lora_a.dtype))
                    lora_b.copy_(B_cand.to(lora_b.dtype))

        if torch.cuda.is_available():
            torch.cuda.synchronize()