import time
import numpy as np
import ray
from typing import List, Dict
from vllm import SamplingParams
from propagate.backend.vllm_lorabackend import VLLMBackendLoRA
from propagate.genome import Genome
from propagate.optimizers.optimizer import Optimizer
import torch

class VLLMFlowBackendLoRA(VLLMBackendLoRA):
    def __init__(self, model_name: str, NUM_GPUS: int, CPUS_PER_GPU: int, GPU_FRACTION_VLLM_WORKER: float, sampler: SamplingParams, use_tqdm: bool = False, time_self: bool = False, max_model_len: int = 4096, lora_rank: int = 8, lora_perturb_target: str = "b-", init_lora_weights: str = True, lora_model_source: str = None, norm_scale_update: bool = True, repeat_tokens_buffer_count: int = 20, repeat_times_kill: int = 15, rep_check_every: int = 100, repeat_tokens_begin_scan_count: int = 500, repeat_tokens_lookback_length: int = 500):
        super().__init__(
            model_name=model_name, NUM_GPUS=NUM_GPUS, CPUS_PER_GPU=CPUS_PER_GPU, 
            GPU_FRACTION_VLLM_WORKER=GPU_FRACTION_VLLM_WORKER, sampler=sampler, 
            use_tqdm=use_tqdm, time_self=time_self, max_model_len=max_model_len, 
            lora_rank=lora_rank, lora_perturb_target=lora_perturb_target, 
            init_lora_weights=init_lora_weights, lora_model_source=lora_model_source, 
            norm_scale_update=norm_scale_update, repeat_tokens_buffer_count=repeat_tokens_buffer_count, 
            repeat_times_kill=repeat_times_kill, rep_check_every=rep_check_every, 
            repeat_tokens_begin_scan_count=repeat_tokens_begin_scan_count, 
            repeat_tokens_lookback_length=repeat_tokens_lookback_length,
            worker_extension_cls="propagate.backend.experimental.vllm_flow_utils.FlowWorkerExtension"
        )

    def get_total_lora_params(self, target: str):
        """Query a worker for the total number of PERTURBED parameters."""
        return ray.get(self.inference_engines[0].collective_rpc.remote("get_total_lora_params", args=(target,)))

    def update(self, optimizer: Optimizer):
        pass

    def generate_outputs(self, genomes: List[Genome], optimizer: Optimizer, suffix: str, inputs: List[List[Dict[str, str]]]):
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        if len(genomes) > self.population_size:
            raise ValueError(f"Population size {len(genomes)} exceeds max population size {self.population_size}.")
        
        prompts = []
        for idk, i in enumerate(inputs):
            prompt_genome = []
            input_genome_content = []
            for j in i:
                input_genome_content.append(j[-1]['content'])
                s = self.tokenizer.apply_chat_template(j, tokenize=False, add_generation_prompt=True)
                if suffix is not None:
                    s = s + suffix
                prompt_genome.append(s)
            prompts.append(prompt_genome)
            genomes[idk].latest_inputs = input_genome_content

        # WARNING: If NUM_GPUS is used for Tensor Parallelism, you MUST send the full `genomes` 
        # list to all workers. Chunking is only correct for Data Parallelism.
        genome_chunks = np.array_split(genomes, self.NUM_GPUS)
        prompt_chunks = np.array_split(prompts, self.NUM_GPUS)

        if self.time_self:
            start_time = time.time()

        perturb_handles = []
        for eng_idx, llm in enumerate(self.inference_engines):
            my_genomes = genome_chunks[eng_idx]
            if len(my_genomes) == 0:
                continue
            h = llm.collective_rpc.remote(
                "set_explicit_weights_multi", args=(my_genomes.tolist(), self.lora_perturb_target, False)
            )
            perturb_handles.append(h)
            
        ray.get(perturb_handles)
        if self.time_self:
            print(f"#-- All explicit weights injected in {time.time() - start_time:.2f}s --#")
            gen_start_time = time.time()

        all_gen_handles = []
        genome_chunks_kept = []

        for eng_idx, llm in enumerate(self.inference_engines):
            my_genomes = genome_chunks[eng_idx]
            my_prompts = prompt_chunks[eng_idx]
            if len(my_genomes) == 0:
                continue

            lora_specs = []
            for local_idx, _genome in enumerate(my_genomes):
                local_lora_id = local_idx + 1
                local_lora_name = f"lora_{local_idx}"
                lora_path = self.lora_paths[local_lora_name]
                lora_specs.append({"name": local_lora_name, "id": local_lora_id, "path": lora_path})

            h = llm.generate_multi_lora.remote(
                my_prompts,
                self.sampler,
                lora_specs,
                self.use_tqdm
            )
            all_gen_handles.append(h)
            genome_chunks_kept.append(my_genomes)

        all_outputs = ray.get(all_gen_handles)

        if self.time_self:
            end_time = time.time()
            print(f"#-- All genome outputs generated in {end_time - gen_start_time:.2f} seconds --#")

        for chunk_results, genomes_in_chunk in zip(all_outputs, genome_chunks_kept):
            for genome, outputs_for_genome in zip(genomes_in_chunk, chunk_results):
                genome.latest_outputs = outputs_for_genome
        
        restore_handles = []
        for eng_idx, llm in enumerate(self.inference_engines):
            my_genomes = genome_chunks[eng_idx]
            if len(my_genomes) > 0:
                h = llm.collective_rpc.remote(
                    "set_explicit_weights_multi", args=(my_genomes.tolist(), self.lora_perturb_target, True)
                )
                restore_handles.append(h)
                
        ray.get(restore_handles)
            
        if self.time_self:
            print(f"#-- All explicit weights restored in {time.time() - end_time:.2f}s --#")