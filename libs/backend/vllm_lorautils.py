import gc
import re
import time
from typing import Dict, List, Optional

from click import Tuple
import torch

from libs.genome import Genome

from libs.genome import Genome
from ray.util import collective

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

    def perturb_self_weights_all(self, genome: Genome):
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager
        
        adapters_dict = adapter_manager.list_adapters()
        modules = adapter_manager.modules
        
        sorted_adapters = sorted(adapters_dict.items(), key=lambda x: x[0])
        
        for adapter in sorted_adapters:
            _aid, lora_model = adapter

            adapter_params_wrapper = lora_model_parameters(lora_model, modules)
            
            genome.update_tensor(model=adapter_params_wrapper)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
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
            _aid, lora_model = sorted_adapters[i]

            adapter_params_wrapper = lora_model_parameters(lora_model, modules)
            
            genome.restore_tensor(model=adapter_params_wrapper)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

class lora_model_parameters:
    def __init__(self, lora_model, modules_list: List[str]):
        self.lora_model = lora_model
        self.modules_list = modules_list

    def named_parameters(self):
        for mod in self.modules_list:
            lora_layer = self.lora_model.get_lora(mod)
            
            if lora_layer is not None:
                yield f"{mod}.lora_a", lora_layer.lora_a
                yield f"{mod}.lora_b", lora_layer.lora_b