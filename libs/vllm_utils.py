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

    def perturb_self_weights(self, genome: Genome):
        genome.update_tensor(model=self.model_runner.model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def restore_self_weights(self, genome: Genome):
        genome.restore_tensor(model=self.model_runner.model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

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
    
    def report_lora_params(self, expected_adapter_names: Optional[List[str]] = None):
        patterns = [
            re.compile(r"(?P<adapter>lora_[A-Za-z0-9_\-]+)"),   # e.g. lora_sql-lora or lora_000
            re.compile(r"lora\.(?P<adapter>[^.]+)"),            # e.g. lora.adaptername....
            re.compile(r"adapters\.(?P<adapter>[^.]+)"),        # e.g. adapters.name...
            re.compile(r"(?P<adapter>[A-Za-z0-9_\-]+)_lora"),   # e.g. sql-lora_lora
        ]

        grouped = {}
        ungrouped = []

        for name, p in self.model_runner.model.named_parameters(recurse=True):
            lname = name.lower()
            matched = False
            for pat in patterns:
                m = pat.search(lname)
                if m:
                    adapter = m.group("adapter")
                    grouped.setdefault(adapter, []).append((name, p))
                    matched = True
                    break
            if not matched:
                # also try to find any expected adapter substring if provided
                if expected_adapter_names:
                    for expect in expected_adapter_names:
                        if expect.lower() in lname:
                            grouped.setdefault(expect, []).append((name, p))
                            matched = True
                            break
            if not matched:
                # heuristic: common LoRA param suffixes (A/B/up/down)
                if any(s in lname for s in (".lora_a", ".lora_b", "lora_a", "lora_b", "lora_up", "lora_down")):
                    ungrouped.append((name, p))
                # else ignore other params

        # If there are ungrouped LoRA-like params but no groups, put them under __UNGROUPED__
        if ungrouped:
            if not grouped:
                grouped["__UNGROUPED__"] = ungrouped
            else:
                # try to assign ungrouped params by proximity (not perfect) - append to first group
                first_key = next(iter(grouped.keys()))
                grouped[first_key].extend(ungrouped)