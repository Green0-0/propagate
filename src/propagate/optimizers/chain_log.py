from propagate.optimizers.chain import OptimizerChain
from propagate.genome import Genome

from typing import Dict

import torch
import wandb

class Log_Perturb_Norms(OptimizerChain):
    # Note: This logs a scatterplot of hundreds of tensors per step id'd by their parameter id
    # Do log should only be True on rank zero
    # Also, your log_at value should terminate one step before the final step or it won't execute
    def __init__(self, source = "perturbation_norms", log_at = None) -> None:
        self.source = source
        self.log_at = log_at if log_at is not None else [3, 10, 100]
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if do_log == False:
            return
        if "perturb_buffer" not in state:
            raise ValueError("Perturbation buffer does not exist to log?")
        if "step" not in state:
            print("Step not received, the logger won't function!")
        
        key = f"log_perturb_norms_{self.source}"
        data = state.get(key)
        step = state["step"]
        if step-1 > max(self.log_at) + 1:
            return
        if data == None:
            table = wandb.Table(columns=["step", "value", "parameter_id"])
            data = {"start": parameter_id, "logged_at_step":False, "data": table}
            state[key] = data
        if step-1 in self.log_at:
            if data["logged_at_step"] == False:
                data["logged_at_step"] = True
                logged_table = wandb.Table(columns=data["data"].columns, data=data["data"].data)
                wandb.log({f"misc/{self.source}": logged_table})
        else:
            data["logged_at_step"] = False
        norms = torch.linalg.vector_norm(state["perturb_buffer"], dtype=torch.float32).item()
        data["data"].add_data(step, norms, str(parameter_id))
        
class Log_Perturb_Means(OptimizerChain):
    def __init__(self, source = "perturbation_means", log_at = None) -> None:
        self.source = source
        self.log_at = log_at if log_at is not None else [3, 10, 100]
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if do_log == False:
            return
        if "perturb_buffer" not in state:
            raise ValueError("Perturbation buffer does not exist to log?")
        if "step" not in state:
            print("Step not received, the logger won't function!")
            
        key = f"log_perturb_means_{self.source}"
        data = state.get(key)
        step = state["step"]
        if step-1 > max(self.log_at) + 1:
            return
        if data == None:
            table = wandb.Table(columns=["step", "value", "parameter_id"])
            data = {"start": parameter_id, "logged_at_step":False, "data": table}
            state[key] = data
        if step-1 in self.log_at:
            if data["logged_at_step"] == False:
                data["logged_at_step"] = True
                logged_table = wandb.Table(columns=data["data"].columns, data=data["data"].data)
                wandb.log({f"misc/{self.source}": logged_table})
        else:
            data["logged_at_step"] = False
        mean = torch.mean(state["perturb_buffer"], dtype=torch.float32).item()
        data["data"].add_data(step, mean, str(parameter_id))
    
class Log_Perturb_Variances(OptimizerChain):
    def __init__(self, source = "perturbation_variances", log_at = None) -> None:
        self.source = source
        self.log_at = log_at if log_at is not None else [3, 10, 100]
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if do_log == False:
            return
        if "perturb_buffer" not in state:
            raise ValueError("Perturbation buffer does not exist to log?")
        if "step" not in state:
            print("Step not received, the logger won't function!")
            
        key = f"log_perturb_variances_{self.source}"
        data = state.get(key)
        step = state["step"]
        if step-1 > max(self.log_at) + 1:
            return
        if data == None:
            table = wandb.Table(columns=["step", "value", "parameter_id"])
            data = {"start": parameter_id, "logged_at_step":False, "data": table}
            state[key] = data
        if step-1 in self.log_at:
            if data["logged_at_step"] == False:
                data["logged_at_step"] = True
                logged_table = wandb.Table(columns=data["data"].columns, data=data["data"].data)
                wandb.log({f"misc/{self.source}": logged_table})
        else:
            data["logged_at_step"] = False
        var = torch.var(state["perturb_buffer"]).item()
        data["data"].add_data(step, var, str(parameter_id))
        
class Log_RMSProp_Norms(OptimizerChain):
    def __init__(self, source = "rmsprop_norms", log_at = None) -> None:
        self.source = source
        self.log_at = log_at if log_at is not None else [3, 10, 100]
        
    @torch.no_grad()
    def apply(self, source: Genome, state: Dict, parameter_id, tensor: torch.Tensor, random_offset: int, do_log: bool = False):
        if do_log == False:
            return
        if "step" not in state:
            print("Step not received, the logger won't function!")
        step = state["step"]
        if step-1 > max(self.log_at) + 1:
            return
        if (parameter_id, "rmsprop_block") in state:
            key = f"log_rmsprop_block_norms_{self.source}"
            data = state.get(key)
            if data == None:
                table = wandb.Table(columns=["step", "value", "parameter_id"])
                data = {"start": parameter_id, "logged_at_step":False, "data": table}
                state[key] = data
            if step-1 in self.log_at:
                if data["logged_at_step"] == False:
                    data["logged_at_step"] = True
                    logged_table = wandb.Table(columns=data["data"].columns, data=data["data"].data)
                    wandb.log({f"misc/{self.source}": logged_table})
            else:
                data["logged_at_step"] = False
            norms = state[(parameter_id, "rmsprop_block")]
            data["data"].add_data(step, norms, str(parameter_id))

        if (parameter_id, "rmsprop") in state:
            key = f"log_rmsprop_norms_{self.source}"
            data = state.get(key)
            if data == None:
                table = wandb.Table(columns=["step", "value", "parameter_id"])
                data = {"start": parameter_id, "logged_at_step":False, "data": table}
                state[key] = data
            if step-1 in self.log_at:
                if data["logged_at_step"] == False:
                    data["logged_at_step"] = True
                    logged_table = wandb.Table(columns=data["data"].columns, data=data["data"].data)
                    wandb.log({f"misc/{self.source}": logged_table})
            else:
                data["logged_at_step"] = False
            norms = torch.linalg.vector_norm(state[(parameter_id, "rmsprop")], dtype=torch.float32).item()
            data["data"].add_data(step, norms, str(parameter_id))