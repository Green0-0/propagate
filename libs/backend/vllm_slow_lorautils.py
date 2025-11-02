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
        
        print("#-- CPU Check (_registered_adapters) --#")
        adapters_found = []
        lora_modules_found = []
        lora_tensors_found = []
        for aid, lora_model in sorted(adapters_dict.items(), key=lambda x: x[0]):
            adapters_found.append(aid)
            for mod in modules:
                lora = lora_model.get_lora(mod)
                lora_modules_found.append(mod)
                lora_tensors_found.append((str(lora.lora_a)[0:20], str(lora.lora_b)[0:20]))
            adapter_manager.pin_adapter(aid)
        print(f"#-- {len(adapters_found)} LoRA Adapters Found: {adapters_found} --#")
        print(f"#-- {len(lora_modules_found)} LoRA Modules Found, ie. {lora_modules_found[0]} --#")
        print(f"#-- {len(lora_tensors_found)} LoRA Tensors Found, ie. {lora_tensors_found[0]} --#")

        print("#-- GPU Check after pinning (_active_adapters) --#")
        active_adapters = adapter_manager._active_adapters
        print(active_adapters)
        for aid, adapter in active_adapters.items():
            print(f"Adapter ID: {aid}, Adapter: {adapter}")
            weights = self._collect_gpu_lora_tensors(aid)
            
            items = list(weights.items())
            n = len(items)
            if not n:
                print("    Adapter has no weights.")
                continue
                
            indices_to_print = [("First", 0), ("Middle", n // 2), ("Last", n - 1)]
            printed = set()
            for label, idx in indices_to_print:
                if idx not in printed:
                    mod_name, (lora_a, lora_b) = items[idx]
                    print(f"    {label} Module: {mod_name}, LoRA A sum: {lora_a.sum().item()}, LoRA B sum: {lora_b.sum().item()}, Device A: {lora_a.device}, Device B: {lora_b.device}")
                    printed.add(idx)

    def _collect_gpu_lora_tensors(self, adapter_id: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Walk through every LoRA-enabled module and collect the two weight tensors
        that belong to the given slot. The returned dict maps the *module name*
        → (lora_a, lora_b).

        This routine works for both ordinary LoRA layers and packed LoRA layers.
        """
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager

        slot = adapter_manager.lora_index_to_id.index(adapter_id)
        gpu_tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        for mod_name, mod in adapter_manager.modules.items():
            a_sub_layers = mod.lora_a_stacked
            b_sub_layers = mod.lora_b_stacked

            if not isinstance(a_sub_layers, (list, tuple)) or not isinstance(b_sub_layers, (list, tuple)):
                print(f"Skipping module {mod_name}: lora_a_stacked/b_stacked is not a list or tuple.")
                continue

            is_packed = len(a_sub_layers) > 1

            for i in range(len(a_sub_layers)):
                a_stacked_tensor = a_sub_layers[i]
                b_stacked_tensor = b_sub_layers[i]

                if a_stacked_tensor is None or b_stacked_tensor is None:
                    continue
                
                if not (isinstance(a_stacked_tensor, torch.Tensor) and a_stacked_tensor.dim() > 0 and a_stacked_tensor.shape[0] > slot):
                    print(f"  WARNING: Skipping {mod_name} sub-layer {i}. LoRA A tensor is invalid or has wrong shape: {a_stacked_tensor.shape if isinstance(a_stacked_tensor, torch.Tensor) else type(a_stacked_tensor)}")
                    continue
                if not (isinstance(b_stacked_tensor, torch.Tensor) and b_stacked_tensor.dim() > 0 and b_stacked_tensor.shape[0] > slot):
                    print(f"  WARNING: Skipping {mod_name} sub-layer {i}. LoRA B tensor is invalid or has wrong shape: {b_stacked_tensor.shape if isinstance(b_stacked_tensor, torch.Tensor) else type(b_stacked_tensor)}")
                    continue

                lora_a = a_stacked_tensor[slot]
                lora_b = b_stacked_tensor[slot]
                
                current_mod_name = mod_name
                if is_packed:
                    current_mod_name = f"{mod_name}#sub{i}"
                
                gpu_tensors[current_mod_name] = (lora_a, lora_b)

        return gpu_tensors

    def inspect_lora(self, msg: str, lora_id: int = 1):
        """
        Inspect one adapter using the same adapter_manager/modules that
        self_report_lora_params_sanity_check references. Safe against
        missing modules or None returns from get_lora().
        """
        output = ""
        output += f"#-- inspect_lora called: {msg} on id {lora_id} --#\n"
        # follow the same lookup pattern as your sanity check
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager
        adapters = adapter_manager.list_adapters()
        modules = adapter_manager.modules

        lora_model = adapters.get(lora_id) if isinstance(adapters, dict) else adapter_manager.get_adapter(lora_id)
        output += f"Inspecting adapter id {lora_id}. Modules to check: {len(modules)} (showing up to 50)\n"
        active_adapters = adapter_manager._active_adapters
        output += f"Active adapters: {list(active_adapters.keys())}\n"
        for aid, adapter in active_adapters.items():
            output += f"Adapter ID: {aid}, Adapter: {adapter}\n"
            weights = self._collect_gpu_lora_tensors(aid)
            
            items = list(weights.items())
            n = len(items)
            if not n:
                output += ("    Adapter has no weights.")
                continue
                
            indices_to_print = [("First", 0), ("Middle", n // 2), ("Last", n - 1)]
            printed = set()
            for label, idx in indices_to_print:
                if idx not in printed:
                    mod_name, (lora_a, lora_b) = items[idx]
                    output += f"    {label} Module: {mod_name}, LoRA A sum: {lora_a.sum().item()}, LoRA B sum: {lora_b.sum().item()}\n"
                    printed.add(idx)
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

    @torch.inference_mode()
    def perturb_self_weights_all(self, genome: Genome):
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager

        adapters_dict = adapter_manager.list_adapters()  # {aid: lora_model}

        sorted_adapters = sorted(adapters_dict.items(), key=lambda x: x[0])

        for aid, _lora_model in sorted_adapters:
            weights = self._collect_gpu_lora_tensors(aid)
            for seed, weight in zip(genome.seeds, genome.seed_weights):
                rand_counter = 0
                for name, (lora_a, lora_b) in weights.items():

                    gen = torch.Generator(device=lora_a.device)
                    gen.manual_seed(int(seed) + rand_counter)
                    rand_counter += 1

                    noise = torch.randn(lora_a.shape, generator=gen, device=lora_a.device, dtype=lora_a.dtype)
                    noise.mul_(weight)
                    lora_a.data.add_(noise)
                    del noise

                    gen = torch.Generator(device=lora_b.device)
                    gen.manual_seed(int(seed) + rand_counter)
                    rand_counter += 1

                    noise = torch.randn(lora_b.shape, generator=gen, device=lora_b.device, dtype=lora_b.dtype)
                    noise.mul_(weight)
                    lora_b.data.add_(noise)
                    del noise
                
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def perturb_self_weights_multi(self, genomes: List[Genome]):
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager

        adapters_dict = adapter_manager.list_adapters()

        sorted_adapters = sorted(adapters_dict.items(), key=lambda x: x[0])

        if len(genomes) > len(sorted_adapters):
            raise ValueError(
                f"Received {len(genomes)} genomes but only {len(sorted_adapters)} adapters are available."
            )
        for i, genome in enumerate(genomes):
            aid, _lora_model = sorted_adapters[i]
            weights = self._collect_gpu_lora_tensors(aid)

            for seed, weight in zip(genome.seeds, genome.seed_weights):
                rand_counter = 0
                
                for _, (lora_a, lora_b) in weights.items():
                    gen = torch.Generator(device=lora_a.device)
                    gen.manual_seed(int(seed) + rand_counter)
                    rand_counter += 1

                    noise = torch.randn(lora_a.shape, generator=gen, device=lora_a.device, dtype=lora_a.dtype)
                    noise.mul_(weight)
                    lora_a.data.add_(noise)
                    del noise

                    gen = torch.Generator(device=lora_b.device)
                    gen.manual_seed(int(seed) + rand_counter)
                    rand_counter += 1

                    noise = torch.randn(lora_b.shape, generator=gen, device=lora_b.device, dtype=lora_b.dtype)
                    noise.mul_(weight)
                    lora_b.data.add_(noise)
                    del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()


    @torch.inference_mode()
    def restore_self_weights_multi(self, genomes: List[Genome]):
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager

        adapters_dict = adapter_manager.list_adapters()
        modules = adapter_manager.modules

        sorted_adapters = sorted(adapters_dict.items(), key=lambda x: x[0])

        if len(genomes) > len(sorted_adapters):
            raise ValueError(
                f"Received {len(genomes)} genomes but only {len(sorted_adapters)} adapters are available."
            )

        for i, genome in enumerate(genomes):
            aid, _lora_model = sorted_adapters[i]
            weights = self._collect_gpu_lora_tensors(aid)

            for seed, weight in zip(genome.seeds, genome.seed_weights):
                rand_counter = 0

                for _, (lora_a, lora_b) in weights.items():
                    gen = torch.Generator(device=lora_a.device)
                    gen.manual_seed(int(seed) + rand_counter)
                    rand_counter += 1

                    noise = torch.randn(lora_a.shape, generator=gen, device=lora_a.device, dtype=lora_a.dtype)
                    noise.mul_(weight)
                    lora_a.data.sub_(noise)
                    del noise

                    gen = torch.Generator(device=lora_b.device)
                    gen.manual_seed(int(seed) + rand_counter)
                    rand_counter += 1

                    noise = torch.randn(lora_b.shape, generator=gen, device=lora_b.device, dtype=lora_b.dtype)
                    noise.mul_(weight)
                    lora_b.data.sub_(noise)
                    del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()