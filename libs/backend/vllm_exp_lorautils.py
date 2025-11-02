import gc
import re
import time
from typing import Dict, List, Tuple

import torch
from ray.util import collective

from libs.genome import Genome

class WorkerExtension:
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

    def inspect_lora(self, msg: str, lora_id: int = 1):
        """
        Inspect one adapter using the same adapter_manager/modules that
        self_report_lora_params_sanity_check references. Safe against
        missing modules or None returns from get_lora().
        """
        output = ""
        output += f"#-- inspect_lora called: {msg} on id {lora_id} --#\n"
        # follow the same lookup pattern as your sanity check
        lora_manager = getattr(self.model_runner, "lora_manager", None)
        adapter_manager = getattr(lora_manager, "_adapter_manager", None) if lora_manager is not None else None
        adapters = adapter_manager.list_adapters()
        modules = adapter_manager.modules

        lora_model = adapters.get(lora_id) if isinstance(adapters, dict) else adapter_manager.get_adapter(lora_id)
        output += f"Inspecting adapter id {lora_id}. Modules to check: {len(modules)} (showing up to 50)\n"
        for mod in sorted(modules):
            l = None
            try:
                l = lora_model.get_lora(mod)
            except Exception as e:
                output += (f"    {mod} error retrieving LoRA: {e}\n")
                continue

            if l is None:
                output += (f"    {mod} LoRA is None\n")
                continue

            # support attribute name variations: lora_a/lora_b or a/b
            a_list = l.lora_a
            b_list = l.lora_b

            # print info for each entry in the a/b lists (usually small)
            for idx, (ta, tb) in enumerate(zip(a_list, b_list)):
                try:
                    if isinstance(ta, torch.Tensor):
                        a_dev = ta.device
                        a_sum = float(ta.detach().float().sum())
                    else:
                        a_dev = type(ta)
                        a_sum = None

                    if isinstance(tb, torch.Tensor):
                        b_dev = tb.device
                        b_sum = float(tb.detach().float().sum())
                    else:
                        b_dev = type(tb)
                        b_sum = None

                    output += (f"    {mod}[{idx}] a: device={a_dev} sum={a_sum}  |  b: device={b_dev} sum={b_sum}\n")
                except Exception as e:
                    output += (f"    {mod}[{idx}] error inspecting tensors: {e}\n")
                if idx > 3:
                    break
            break
        output += ("#-- inspect_lora done --#\n")
        print(output)

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
        if not genomes:
            return True
        if not hasattr(self, "_lora_groups_cache"):
            self._init_lora_groups()

        lora_ids = [i + 1 for i in range(len(genomes))]
        self._ensure_lora_groups(lora_ids)

        adapter_manager = self._get_adapter_manager()

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
                    torch.randn(g["tmp"].shape, generator=self._rng, out=g["tmp"])
                    g["gpu"].add_(g["tmp"], alpha=float(weight))

            # Copy all dtype groups back to CPU (do this for *all* groups first)
            for g in groups.values():
                g["cpu"].copy_(g["gpu"], non_blocking=True)

            # Now, perform one activate/deactivate step (if needed) per adapter,
            # not once per dtype. This avoids re-creating adapter mid-write.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            try:
                if hasattr(adapter_manager, "deactivate_adapter"):
                    adapter_manager.deactivate_adapter(lora_id)
                elif hasattr(adapter_manager, "remove_adapter"):
                    adapter_manager.remove_adapter(lora_id)
            except Exception as e:
                raise
            try:
                ok = adapter_manager.activate_adapter(lora_id)
            except Exception as e:
                raise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    @torch.inference_mode()
    def restore_self_weights_multi(self, genomes: List[Genome]):
        if not genomes:
            return True
        if not hasattr(self, "_lora_groups_cache"):
            self._init_lora_groups()

        lora_ids = [i + 1 for i in range(len(genomes))]
        self._ensure_lora_groups(lora_ids)

        adapter_manager = self._get_adapter_manager()

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

            # copy back all groups
            for g in groups.values():
                g["cpu"].copy_(g["gpu"], non_blocking=True)

            # single activate/deactivate step (if needed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            try:
                if hasattr(adapter_manager, "deactivate_adapter"):
                    adapter_manager.deactivate_adapter(lora_id)
                elif hasattr(adapter_manager, "remove_adapter"):
                    adapter_manager.remove_adapter(lora_id)
            except Exception as e:
                raise
            try:
                adapter_manager.activate_adapter(lora_id)
            except Exception as e:
                raise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True


    @torch.inference_mode()
    def perturb_self_weights_all(self, genome: Genome):
        if not hasattr(self, "_lora_groups_cache"):
            self._init_lora_groups()

        adapter_manager = self._get_adapter_manager()
        adapters = adapter_manager.list_adapters()
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

            # activate/deactivate once per adapter after full copy-back
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            try:
                if hasattr(adapter_manager, "deactivate_adapter"):
                    adapter_manager.deactivate_adapter(lora_id)
                elif hasattr(adapter_manager, "remove_adapter"):
                    adapter_manager.remove_adapter(lora_id)
            except Exception as e:
                raise
            try:
                adapter_manager.activate_adapter(lora_id)
            except Exception as e:
                raise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True