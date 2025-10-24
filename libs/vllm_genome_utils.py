import gc
import time
import torch

from libs.genome import Genome

class WorkerExtension:
    def save_self_initial_weights(self):
        self.initial_weights = {}
        for name, p in self.model_runner.model.named_parameters():
            self.initial_weights[name] = p.detach().clone().cpu()

    def restore_self_initial_weights(self):
        for name, p in self.model_runner.model.named_parameters():
            if name in self.initial_weights:
                p.data.copy_(self.initial_weights[name].to(p.device))

        del self.initial_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time.sleep(1)

    def perturb_self_weights(self, genome: Genome):
        genome.update_tensor(named_parameters=self.model_runner.model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def restore_self_weights(self, genome: Genome):
        genome.restore_tensor(named_parameters=self.model_runner.model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()