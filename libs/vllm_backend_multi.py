import math
from typing import Dict, List
from libs.backend import Backend

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

from ray.util import collective
from torch.distributed import ReduceOp

class VLLMBackendMulti(Backend):
    training_actors: List[ray.actor.ActorHandle]
    llm: ray.actor.ActorHandle
    tokenizer: AutoTokenizer
    sampler: SamplingParams

    output_log_file: str
    full_output_log_file: str

    def __init__(self, model_name: str, NUM_GPUS: int, CPUS_PER_GPU: int, GPU_FRACTION_TRAINING_ACTOR: float, GPU_FRACTION_VLLM_WORKER: float, Sampler: SamplingParams, output_log_file: str = "logs/output.log", full_output_log_file: str = "logs/full_output.log", use_tqdm: bool = False, time_self: bool = True):
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

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

                self.world_size = 1

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
            
            def init_collective(self, world_size: int, rank: int, backend: str = "nccl", group_name: str = "actor_sync_group"):
                self.world_size = world_size
                if collective.is_group_initialized(group_name):
                    return True
                collective.init_collective_group(
                    world_size=world_size,
                    rank=rank,
                    backend=backend,
                    group_name=group_name,
                )
                return True
            
            def perform_all_reduce_sync(self):
                if not collective.is_group_initialized("actor_sync_group"):
                    return True
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        collective.allreduce(
                            p.data, 
                            op="SUM", 
                            group_name="actor_sync_group"
                        )
                        
                        p.data.div_(self.world_size)
                
                torch.cuda.synchronize(self.device)
                return True
            
            def __del__(self):
                if collective.is_group_initialized("actor_sync_group"):
                    collective.destroy_collective_group("actor_sync_group")
        #-----------------------------------------------------#

        print("#-- Initializing Backend [VLLMBackendTP] --#")
        print(f"#-- GPUS: {NUM_GPUS}, CPUS per GPU: {CPUS_PER_GPU}, GPU Fraction Training Actor: {GPU_FRACTION_TRAINING_ACTOR}, GPU Fraction VLLM Worker: {GPU_FRACTION_VLLM_WORKER} --#")
        ray.init(address=None, ignore_reinit_error=True)
        pgs = [placement_group([{"GPU": 1, "CPU": CPUS_PER_GPU}]) for _ in range(NUM_GPUS)]
        ray.get([pg.ready() for pg in pgs])

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampler = Sampler
        self.use_tqdm = use_tqdm
        self.time_self = time_self

        self.output_log_file = output_log_file
        self.full_output_log_file = full_output_log_file
        open(self.output_log_file, "w", encoding="utf-8").close()
        open(self.full_output_log_file, "w", encoding="utf-8").close()

        print("#-- Spawning Training Actors with vLLM backends --#")
        # Spawn training actors
        self.training_actors = []
        self.inference_engines = []

        for bidx in range(NUM_GPUS):
            pg = pgs[bidx]
            a = RayTrainingActor.options(
                num_cpus=2,
                num_gpus=GPU_FRACTION_TRAINING_ACTOR,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                ),
            ).remote()
            self.training_actors.append(a)

            eng = ray.remote(
                num_cpus=0,
                num_gpus=0,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True),
            )(MyLLM).remote(
                model=model_name,
                enforce_eager=True,
                worker_extension_cls="libs.vllm_backend_utils.ColocateWorkerExtension",
                tensor_parallel_size=1,
                distributed_executor_backend="ray",
                gpu_memory_utilization=GPU_FRACTION_VLLM_WORKER,
                bundle_indices=[0],
            )
            self.inference_engines.append(eng)

        if len(self.training_actors) > 1:
            print("#-- Initializing Ray Collective group for GPU sync --#")
            world_size = len(self.training_actors)
            ray.get([
                a.init_collective.remote(
                    world_size=world_size,
                    rank=rank,
                    backend="nccl",
                    group_name="actor_sync_group",
                )
                for rank, a in enumerate(self.training_actors)
            ])
            print("#-- Collective group initialized --#")
        else:
            print("#-- Skipping collective group (1 GPU) --#")
        
        print("#-- Verifying Co-location of Training Actors and vLLM Workers --#")
        for bidx in range(NUM_GPUS):
            actor = self.training_actors[bidx]
            engine = self.inference_engines[bidx]

            actor_dev_id = ray.get(actor.report_device_id.remote())
            engine_dev_ids = ray.get(engine.collective_rpc.remote("report_device_id", args=tuple()))

            assert engine_dev_ids and actor_dev_id in engine_dev_ids, \
                f"Training actor and vLLM worker on GPU index {bidx} are NOT co-located!"
        
        self._push_weights()
        print("#-- Backend Initialized --#")
        pass

    def _push_weights(self):
        ipc = {}

        ipc_handles_list = ray.get([a.get_weight_ipc_handles.remote() for a in self.training_actors])
        for handle_dict in ipc_handles_list:
            ipc.update(handle_dict)

        tasks = []
        for eng in self.inference_engines:
            tasks.append(eng.collective_rpc.remote("update_weights_from_ipc_handles", args=(ipc,)))
        ray.get(tasks)
        return True

    def update(self, genome: Genome):
        """Update the model permanently with a genome as the source."""
        perturb_tasks = [
            a.perturb.remote(genome)
            for a in self.training_actors
        ]
        ray.get(perturb_tasks)

        if len(self.training_actors) > 1:
            sync_tasks = [
                a.perform_all_reduce_sync.remote()
                for a in self.training_actors
            ]
            ray.get(sync_tasks)
        self._push_weights()

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

        num_chunks = math.ceil(len(genomes) / len(self.inference_engines))
        for i in range(num_chunks):
            start_time = time.time()

            chunk = genomes[i * len(self.inference_engines):min((i + 1) * len(self.inference_engines), len(genomes))]
            current_actors = self.training_actors[:len(chunk)]
            current_engines = self.inference_engines[:len(chunk)]
            
            perturb_tasks = [
                a.perturb.remote(chunk[i])
                for i, a in enumerate(current_actors)
            ]
            ray.get(perturb_tasks)

            ipc_tasks = [actor.get_weight_ipc_handles.remote() for actor in current_actors]
            ipc_handles_list = ray.get(ipc_tasks)

            push_tasks = [
                engine.collective_rpc.remote("update_weights_from_ipc_handles", args=(ipc,))
                for engine, ipc in zip(current_engines, ipc_handles_list)
            ]
            ray.get(push_tasks)

            gen_tasks = [
                engine.generate.remote(prompts, self.sampler, use_tqdm=self.use_tqdm)
                for engine in current_engines
            ]
            parallel_gens = ray.get(gen_tasks)

            restore_tasks = []
            for i, n_idx in enumerate(range(len(chunk))):
                g = chunk[n_idx]
                gen = parallel_gens[i]
                actor = current_actors[i]

                batch_texts = []
                for out in gen:
                    text_field = getattr(out, "outputs", None)
                    if text_field and len(text_field) > 0 and hasattr(text_field[0], "text"):
                        batch_texts.append(text_field[0].text)
                        with open(self.full_output_log_file, "a", encoding="utf-8") as f:
                            f.write(text_field[0].text + "\n")
                    else:
                        batch_texts.append(getattr(out, "text", str(out)))
                
                g.latest_outputs = batch_texts

                with open(self.output_log_file, "a", encoding="utf-8") as f:
                    f.write(batch_texts[0] + "\n")

                restore_tasks.append(actor.restore.remote(g))
            ray.get(restore_tasks)

            if len(self.training_actors) > 1:
                sync_tasks = [
                    a.perform_all_reduce_sync.remote()
                    for a in self.training_actors
                ]
                ray.get(sync_tasks)

            self._push_weights()
            torch.cuda.empty_cache()

            if self.time_self:
                end_time = time.time()
                print(f"#-- Genome outputs generated in {end_time - start_time:.2f} seconds --#")

