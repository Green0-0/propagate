from typing import Dict, List
from libs.backend import backend

import os
import ray
import torch
import time 
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

from libs.genome import Genome

class VLLMBackendTP(backend):
    training_actors: List[ray.actor.ActorHandle]
    llm: ray.actor.ActorHandle
    tokenizer: AutoTokenizer
    sampler: SamplingParams

    def __init__(self, model_name: str, NUM_GPUS: int, CPUS_PER_GPU: int, GPU_FRACTION_TRAINING_ACTOR: float, GPU_FRACTION_VLLM_WORKER: float, Sampler: SamplingParams):
        #--------------------------------------------------------#
        #                CUSTOM CLASSES DEFINITION               #
        #--------------------------------------------------------#
        class MyLLM(LLM):
            def __init__(self, *args, bundle_indices: list, **kwargs):
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(GPU_FRACTION_VLLM_WORKER)
                os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
                super().__init__(*args, **kwargs)

        @ray.remote
        class RayTrainingActor:
            def __init__(self):
                self.device = torch.device("cuda:0")
                self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
                self.model.to(self.device)
                self.model.eval()
                torch.cuda.synchronize(self.device)
                from vllm.platforms import current_platform
                self.device_uuid = current_platform.get_device_uuid(0)

            def report_device_id(self):
                return self.device_uuid

            def get_weight_ipc_handles(self):
                from torch.multiprocessing.reductions import reduce_tensor
                return {self.device_uuid: {name: reduce_tensor(p.detach()) for name, p in self.model.named_parameters()}}

            def perturb(self, genome: Genome):
                genome.update_tensor(self.model.named_parameters(), device=self.device)
                torch.cuda.synchronize(self.device)
                return True

            def restore(self, genome: Genome):
                genome.restore_tensor(self.model.named_parameters(), device=self.device)
                torch.cuda.synchronize(self.device)
                return True
        #-----------------------------------------------------#

        ray.init(ignore_reinit_error=True)
        pg = placement_group([{"GPU": 1, "CPU": CPUS_PER_GPU}] * NUM_GPUS)
        ray.get(pg.ready())

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampler = Sampler

        # Spawn training actors
        self.training_actors = []
        for bidx in range(NUM_GPUS):
            a = RayTrainingActor.options(
                num_cpus=2,
                num_gpus=GPU_FRACTION_TRAINING_ACTOR,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bidx,
                    placement_group_capture_child_tasks=True,
                ),
            ).remote()
            self.training_actors.append(a)

        # Spawn a single vLLM engine across all bundles (co-located)
        bundle_indices = list(range(NUM_GPUS))
        self.llm = ray.remote(
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True),
        )(MyLLM).remote(
            model=model_name,
            enforce_eager=True,
            worker_extension_cls="vllm_backend_utils.ColocateWorkerExtension",
            tensor_parallel_size=max(1, NUM_GPUS),
            distributed_executor_backend="ray",
            gpu_memory_utilization=GPU_FRACTION_VLLM_WORKER,
            bundle_indices=bundle_indices,
        )

        # Verify co-location
        training_actor_device_ids = []
        for idx, a in enumerate(self.training_actors):
            dev_id = ray.get(a.report_device_id.remote())
            training_actor_device_ids.append(dev_id)
            print(f"[train-{idx}] device UUID: {dev_id}")

        ids = ray.get(self.llm.collective_rpc.remote("report_device_id", args=tuple()))
        inference_device_ids = ids if isinstance(ids, list) else [ids]
        print(f"[vLLM] device UUID(s): {inference_device_ids}")
        assert set(training_actor_device_ids) <= set(inference_device_ids), \
            "Training actors and vLLM workers are NOT co-located!"
        
        self._push_weights()
        pass

    def _push_weights(self):
        ipc = {}
        for a in self.training_actors:
            ipc.update(ray.get(a.get_weight_ipc_handles.remote()))
        ray.get(self.llm.collective_rpc.remote("update_weights_from_ipc_handles", args=(ipc,)))
        return True

    def update(self, genome: Genome):
        """Update the model permanently with a genome as the source."""
        perturb_tasks = [
            a.perturb.remote(genome)
            for a in self.training_actors
        ]
        ray.get(perturb_tasks)

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[Dict[str, str]]]):
        """
        Generate outputs based on the genome and inputs.
        Updates the genomes with their new outputs.
        """
        prompts = []
        for i in inputs:
            s = self.tokenizer.apply_chat_template(
                i,
                tokenize=False,
                add_generation_prompt=True
            )
            if suffix is not None:
                s = s + suffix
            prompts.append(s)

        for g in genomes:
            perturb_tasks = [
                a.perturb.remote(g)
                for a in self.training_actors
            ]
            ray.get(perturb_tasks)

            gen = ray.get(self.llm.generate.remote(prompts, self.sampler, use_tqdm=False))
            batch_texts = []
            for out in gen:
                text_field = getattr(out, "outputs", None)
                if text_field and len(text_field) > 0 and hasattr(text_field[0], "text"):
                    batch_texts.append(text_field[0].text)
                else:
                    batch_texts.append(getattr(out, "text", str(out)))
                    
            g.latest_outputs = batch_texts

            restore_tasks = [
                a.restore.remote(g)
                for a in self.training_actors
            ]
            ray.get(restore_tasks)
            torch.cuda.empty_cache()

