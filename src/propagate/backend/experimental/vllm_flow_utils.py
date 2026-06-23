import torch
from typing import Dict, List, Tuple
from propagate.genome import Genome
from propagate.backend.vllm_lorautils import WorkerExtension as BaseWorkerExtension

class FlowWorkerExtension(BaseWorkerExtension):
    def _build_flat_mapping(self, aid: int):
        if hasattr(self, '_flat_shapes_cache') and aid in self._flat_shapes_cache:
            return self._flat_shapes_cache[aid]
            
        weights = self._collect_gpu_lora_tensors(aid)
        shapes = []
        total_elements = 0
        
        for layer_name, (lora_a, lora_b) in sorted(weights.items()):
            num_a = lora_a.numel()
            num_b = lora_b.numel()
            
            shapes.append({
                "layer": layer_name,
                "a_shape": lora_a.shape,
                "b_shape": lora_b.shape,
                "a_start": total_elements,
                "a_end": total_elements + num_a,
                "b_start": total_elements + num_a,
                "b_end": total_elements + num_a + num_b
            })
            total_elements += num_a + num_b
            
        if not hasattr(self, '_flat_shapes_cache'):
            self._flat_shapes_cache = {}
        self._flat_shapes_cache[aid] = (shapes, total_elements)
        return self._flat_shapes_cache[aid]

    def get_total_lora_params(self, target: str = "ab") -> int:
        """Returns the total number of PERTURBED parameters based on target."""
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager
        adapters_dict = adapter_manager.list_adapters()
        if not adapters_dict:
            return 0
        aid, _ = sorted(adapters_dict.items(), key=lambda x: x[0])[0]
        shapes, _ = self._build_flat_mapping(aid)
        
        target_params = 0
        for shape_info in shapes:
            if "a" in target.lower():
                target_params += shape_info["a_end"] - shape_info["a_start"]
            if "b" in target.lower():
                target_params += shape_info["b_end"] - shape_info["b_start"]
        return target_params

    @torch.inference_mode()
    def set_explicit_weights_multi(self, genomes: List[Genome], target: str, is_restore: bool = False):
        lora_manager = self.model_runner.lora_manager
        adapter_manager = lora_manager._adapter_manager
        adapters_dict = adapter_manager.list_adapters()
        sorted_adapters = sorted(adapters_dict.items(), key=lambda x: x[0])

        if len(genomes) > len(sorted_adapters):
            raise ValueError(f"Received {len(genomes)} genomes but only {len(sorted_adapters)} adapters are available.")

        for i, genome in enumerate(genomes):
            aid, _ = sorted_adapters[i]
            shapes, _ = self._build_flat_mapping(aid)
            weights = self._collect_gpu_lora_tensors(aid)
            
            if not hasattr(self, '_original_weights_cache'):
                self._original_weights_cache = {}
            if aid not in self._original_weights_cache:
                self._original_weights_cache[aid] = {k: (v[0].clone(), v[1].clone()) for k, v in weights.items()}
                
            if is_restore:
                # FIXED: Restore original weights
                for layer_name, (lora_a, lora_b) in weights.items():
                    orig_a, orig_b = self._original_weights_cache[aid][layer_name]
                    if "a" in target.lower():
                        lora_a.copy_(orig_a)
                    if "b" in target.lower():
                        lora_b.copy_(orig_b)
            else:
                flat_candidate = genome.special_metadata.get("flow_candidate")
                if flat_candidate is None:
                    raise ValueError(f"Genome {i} is missing 'flow_candidate' in special_metadata")
                
                device = list(weights.values())[0][0].device
                flat_candidate = flat_candidate.to(device, dtype=list(weights.values())[0][0].dtype)
                
                # FIXED: Consume flat_candidate sequentially based on target parameters
                offset = 0
                for shape_info in shapes:
                    layer_name = shape_info["layer"]
                    lora_a, lora_b = weights[layer_name]
                    orig_a, orig_b = self._original_weights_cache[aid][layer_name]
                    
                    if "a" in target.lower():
                        a_size = shape_info["a_end"] - shape_info["a_start"]
                        a_slice = flat_candidate[offset : offset + a_size].view(shape_info["a_shape"])
                        lora_a.copy_(orig_a + a_slice)
                        offset += a_size
                    if "b" in target.lower():
                        b_size = shape_info["b_end"] - shape_info["b_start"]
                        b_slice = flat_candidate[offset : offset + b_size].view(shape_info["b_shape"])
                        lora_b.copy_(orig_b + b_slice)
                        offset += b_size
                        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()