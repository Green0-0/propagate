import gc
import re
import time
from typing import Dict, List, Tuple

import torch
from ray.util import collective

from libs.genome import Genome

class WorkerExtension:
    def init_collective_group(self, world_size: int, rank: int, backend: str = "nccl"):
        self.collective_group_name = "weight_sync_group"
        self.world_size = world_size
        self.rank = rank
        
        if collective.is_group_initialized(self.collective_group_name):
            collective.destroy_collective_group(self.collective_group_name)
        
        collective.init_collective_group(
            world_size=world_size,
            rank=rank,
            backend=backend,
            group_name=self.collective_group_name,
        )
        print(f"#-- Worker {rank} collective group initialized. --#")
        return True

    def perform_all_reduce_sync(self):
        """Performs AllReduce to average weights across all workers."""
        if not collective.is_group_initialized(self.collective_group_name) or self.world_size <= 1:
            return True
            
        with torch.no_grad():
            for name, p in self.model_runner.model.named_parameters():
                if p.requires_grad:
                    collective.allreduce(
                        p.data, 
                        group_name=self.collective_group_name
                    )
                    p.data.div_(self.world_size)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def destroy_collective_group(self):
        if collective.is_group_initialized(self.collective_group_name):
            collective.destroy_collective_group(self.collective_group_name)
        return True

    def __del__(self):
        self.destroy_collective_group()

    def save_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True
    
    def self_report_lora_params_sanity_check(self):
        print("#-- LoRA Parameters Sanity Check --#")
        lora_manager = self.model_runner.lora_manager
        adapter_manager = self.model_runner.lora_manager._adapter_manager
        if lora_manager is None or adapter_manager is None:
            raise RuntimeError("LoRA manager or adapter manager not found in model runner.")
        adapters_dict = adapter_manager.list_adapters()
        if not adapters_dict:
            raise RuntimeError("No adapters found in adapter manager.")
        modules = adapter_manager.modules
        if not modules:
            raise RuntimeError("No modules found in adapter manager.")
        
        adapters_found = []
        lora_modules_found = []
        lora_tensors_found = []
        for aid, lora_model in sorted(adapters_dict.items(), key=lambda x: x[0]):
            adapters_found.append(aid)
            for mod in modules:
                lora = lora_model.get_lora(mod)
                lora_modules_found.append(mod)
                lora_tensors_found.append((str(lora.lora_a)[0:20], str(lora.lora_b)[0:20]))
        print(f"#-- {len(adapters_found)} LoRA Adapters Found: {adapters_found} --#")
        print(f"#-- {len(lora_modules_found)} LoRA Modules Found, ie. {lora_modules_found[0]} --#")
        print(f"#-- {len(lora_tensors_found)} LoRA Tensors Found, ie. {lora_tensors_found[0]} --#")

    # --- lazy init since Ray may not call __init__ on this extension ---
    def _init_lora_groups(self):
        if hasattr(self, "_lora_groups_cache"):
            return
        # Cache: lora_id -> { dtype: {"cpu": flat_cpu_pinned, "gpu": flat_gpu, "tmp": tmp_gpu} }
        self._lora_groups_cache: Dict[int, Dict[torch.dtype, Dict[str, torch.Tensor]]] = {}
        # One reusable CUDA RNG generator (or CPU as fallback)
        self._rng = torch.Generator(device="cuda") if torch.cuda.is_available() else torch.Generator()
        self._lora_groups_initialized = True

    # ---------- utilities ----------
    def _get_adapter_manager(self):
        lora_manager = getattr(self.model_runner, "lora_manager", None)
        if lora_manager is None:
            raise RuntimeError("WorkerExtension: lora_manager not found on model_runner.")
        adapter_manager = getattr(lora_manager, "_adapter_manager", None)
        if adapter_manager is None:
            raise RuntimeError("WorkerExtension: _adapter_manager not found on lora_manager.")
        return adapter_manager

    def _list_lora_tensors_by_dtype(self, lora_model, module_names):
        """
        Return mapping dtype -> list of entries (list_ref, idx, shape, numel, tensor)
        in a stable order across modules and A/B lists.
        """
        groups: Dict[torch.dtype, List] = {}
        for mod in module_names:
            lora = lora_model.get_lora(mod)
            for lst in (lora.lora_a, lora.lora_b):
                for i, t in enumerate(lst):
                    if not isinstance(t, torch.Tensor):
                        continue
                    dtype = t.dtype
                    if dtype not in groups:
                        groups[dtype] = []
                    tt = t.detach().contiguous()
                    groups[dtype].append((lst, i, tuple(tt.shape), tt.numel(), tt))
        return groups

    def _build_lora_groups_for_adapter(self, lora_id: int):
        """
        Build flat pinned CPU buffers and matching GPU scratch buffers for a given adapter.
        Replace adapter tensors with views into the pinned CPU buffers.
        """
        adapter_manager = self._get_adapter_manager()
        try:
            lora_model = adapter_manager.get_adapter(lora_id)
        except KeyError:
            # No adapter by that id
            return

        module_names = adapter_manager.modules
        groups = self._list_lora_tensors_by_dtype(lora_model, module_names)
        if not groups:
            # No tensors to manage for this adapter
            self._lora_groups_cache[lora_id] = {}
            return

        cache_entry: Dict[torch.dtype, Dict[str, torch.Tensor]] = {}
        for dtype, entries in groups.items():
            total = sum(numel for (_, _, _, numel, _) in entries)
            if total == 0:
                continue

            # Create one pinned CPU flat buffer
            cpu_flat = torch.empty(total, dtype=dtype, device="cpu", pin_memory=True)

            # Pack original data into pinned flat buffer and replace adapter tensors with views
            offset = 0
            for (lst, idx, shape, numel, tt) in entries:
                cpu_flat[offset:offset + numel].copy_(tt.view(-1), non_blocking=False)
                lst[idx] = cpu_flat[offset:offset + numel].view(shape)
                offset += numel

            # Create GPU buffers for processing
            if torch.cuda.is_available():
                gpu_flat = torch.empty_like(cpu_flat, device="cuda")
                tmp_flat = torch.empty_like(cpu_flat, device="cuda")
            else:
                # CPU fallback (will be slower)
                gpu_flat = torch.empty_like(cpu_flat)
                tmp_flat = torch.empty_like(cpu_flat)

            cache_entry[dtype] = {"cpu": cpu_flat, "gpu": gpu_flat, "tmp": tmp_flat}

        self._lora_groups_cache[lora_id] = cache_entry

    def _ensure_lora_groups(self, lora_ids: List[int]):
        adapter_manager = self._get_adapter_manager()
        existing = adapter_manager.list_adapters()  # {id: lora_model}
        for lora_id in lora_ids:
            if lora_id not in existing:
                # Skip silently if not present; matches your original behavior
                continue
            if lora_id not in self._lora_groups_cache:
                self._build_lora_groups_for_adapter(lora_id)

    # ---------- high-level APIs called by your backend ----------

    @torch.inference_mode()
    def perturb_self_weights_multi(self, genomes: List[Genome]):
        """
        Perturb multiple adapters (one per genome) efficiently.
        Expects vLLM LoRA adapter ids to be 1..N for the chunk assigned to this engine.
        """
        if not genomes:
            return True
        if not hasattr(self, "_lora_groups_cache"):
            self._init_lora_groups()

        # Map each genome to its local adapter id (local_idx + 1)
        lora_ids = [i + 1 for i in range(len(genomes))]
        self._ensure_lora_groups(lora_ids)

        for local_idx, genome in enumerate(genomes):
            lora_id = local_idx + 1
            groups = self._lora_groups_cache.get(lora_id, {})
            if not groups:
                continue

            # Stage current CPU weights onto GPU once per dtype
            for g in groups.values():
                g["gpu"].copy_(g["cpu"], non_blocking=True)

            # Apply noise: for each seed, fill tmp and add scaled noise
            for seed, weight in zip(genome.seeds, genome.seed_weights):
                self._rng.manual_seed(int(seed))
                for g in groups.values():
                    # Fill tmp in-place using 'out=' to avoid allocation
                    torch.randn(g["tmp"].shape, generator=self._rng, out=g["tmp"])
                    g["gpu"].add_(g["tmp"], alpha=float(weight))

            # Copy back to CPU pinned buffers
            for g in groups.values():
                g["cpu"].copy_(g["gpu"], non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    @torch.inference_mode()
    def restore_self_weights_multi(self, genomes: List[Genome]):
        """
        Restore multiple adapters (inverse of perturb) efficiently.
        """
        if not genomes:
            return True
        if not hasattr(self, "_lora_groups_cache"):
            self._init_lora_groups()

        lora_ids = [i + 1 for i in range(len(genomes))]
        self._ensure_lora_groups(lora_ids)

        for local_idx, genome in enumerate(genomes):
            lora_id = local_idx + 1
            groups = self._lora_groups_cache.get(lora_id, {})
            if not groups:
                continue

            for g in groups.values():
                g["gpu"].copy_(g["cpu"], non_blocking=True)

            for seed, weight in zip(genome.seeds, genome.seed_weights):
                self._rng.manual_seed(int(seed))
                for g in groups.values():
                    torch.randn(g["tmp"].shape, generator=self._rng, out=g["tmp"])
                    g["gpu"].add_(g["tmp"], alpha=-float(weight))  # subtract

            for g in groups.values():
                g["cpu"].copy_(g["gpu"], non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    @torch.inference_mode()
    def perturb_self_weights_all(self, genome: Genome):
        """
        Apply perturbation to ALL known adapters on this engine.
        Used by Backend.update().
        """
        if not hasattr(self, "_lora_groups_cache"):
            self._init_lora_groups()

        adapter_manager = self._get_adapter_manager()
        adapters = adapter_manager.list_adapters()  # dict-like {lora_id: lora_model}
        if not adapters:
            return True

        lora_ids = sorted(list(adapters.keys()))
        self._ensure_lora_groups(lora_ids)

        for lora_id in lora_ids:
            groups = self._lora_groups_cache.get(lora_id, {})
            if not groups:
                continue

            for g in groups.values():
                g["gpu"].copy_(g["cpu"], non_blocking=True)

            for seed, weight in zip(genome.seeds, genome.seed_weights):
                self._rng.manual_seed(int(seed))
                for g in groups.values():
                    torch.randn(g["tmp"].shape, generator=self._rng, out=g["tmp"])
                    g["gpu"].add_(g["tmp"], alpha=float(weight))

            for g in groups.values():
                g["cpu"].copy_(g["gpu"], non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    """
    def perturb_self_weights_multi(self, genomes: List[Genome]):
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager
        
        adapters_dict = adapter_manager.list_adapters() # {aid: lora_model}
        modules = adapter_manager.modules
        
        sorted_adapters = sorted(adapters_dict.items(), key=lambda x: x[0])
        
        if len(genomes) > len(sorted_adapters):
            raise ValueError(
                f"Received {len(genomes)} genomes but only {len(sorted_adapters)} adapters are available."
            )
        
        for i, genome in enumerate(genomes):
            _aid, lora_model = sorted_adapters[i]

            adapter_params_wrapper = lora_model_parameters(lora_model, modules)
            
            genome.update_tensor(model=adapter_params_wrapper)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
    """