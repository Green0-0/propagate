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
    def __init__(self, model_name: str, NUM_GPUS: int, CPUS_PER_GPU: int, GPU_FRACTION_VLLM_WORKER: float, sampler: SamplingParams, use_tqdm: bool = False, time_self: bool = False, max_model_len: int = 4096, lora_rank: int = 2, init_lora_weights: str = "zero", lora_model_source: str = None, use_rslora: bool = False, normalize_svd: bool = False, repeat_tokens_buffer_count: int = 20, repeat_times_kill: int = 15, rep_check_every: int = 100, repeat_tokens_begin_scan_count: int = 500, repeat_tokens_lookback_length: int = 500, worker_extension_cls: str = "propagate.backend.experimental.vllm_flow_utils.FlowWorkerExtension"):
        super().__init__(
            model_name=model_name, NUM_GPUS=NUM_GPUS, CPUS_PER_GPU=CPUS_PER_GPU, 
            GPU_FRACTION_VLLM_WORKER=GPU_FRACTION_VLLM_WORKER, sampler=sampler, 
            use_tqdm=use_tqdm, time_self=time_self, max_model_len=max_model_len, 
            lora_rank=lora_rank, lora_perturb_target="ab", 
            init_lora_weights=init_lora_weights, lora_model_source=lora_model_source, 
            norm_scale_update=False, repeat_tokens_buffer_count=repeat_tokens_buffer_count, 
            repeat_times_kill=repeat_times_kill, rep_check_every=rep_check_every, 
            repeat_tokens_begin_scan_count=repeat_tokens_begin_scan_count, 
            repeat_tokens_lookback_length=repeat_tokens_lookback_length,
            worker_extension_cls=worker_extension_cls, use_rslora=use_rslora
        )
        self.backend_name = f"Rank {lora_rank} TinyLoRA vLLM Backend (rslora=False, norm_svd={normalize_svd})"
        self.normalize_svd = normalize_svd

    def compute_tinylora_svd(self, lora_rank: int):
        print(f"#-- Computing TinyLoRA SVD (lora_rank={lora_rank}, normalize_svd={self.normalize_svd}) --#")
        svd_info_list = ray.get(
            self.inference_engines[0].collective_rpc.remote(
                "compute_tinylora_svd", args=(lora_rank, self.normalize_svd)))
        svd_info = svd_info_list[0] if isinstance(svd_info_list, list) else svd_info_list
        print(f"#-- SVD computed for {len(svd_info)} modules on worker 0 --#")

        if len(self.inference_engines) > 1:
            svd_data_list = ray.get(
                self.inference_engines[0].collective_rpc.remote("get_tinylora_svd_data"))
            svd_data = svd_data_list[0] if isinstance(svd_data_list, list) else svd_data_list
            ray.get([
                llm.collective_rpc.remote("set_tinylora_svd_data", args=(svd_data,))
                for llm in self.inference_engines[1:]
            ])
            print(f"#-- SVD broadcast to {len(self.inference_engines) - 1} other workers --#")

        return svd_info

    def init_tinylora(self, u_dim: int, n_tie: int, lora_rank: int):
        print(f"#-- Initializing TinyLoRA (u_dim={u_dim}, n_tie={n_tie}) --#")
        ray.get([
            llm.collective_rpc.remote("init_tinylora", args=(u_dim, n_tie, lora_rank))
            for llm in self.inference_engines
        ])

    def apply_mode_permanently(self, mode_v: torch.Tensor):
        print("#-- Applying mode permanently to all workers --#")
        ray.get([
            llm.collective_rpc.remote("apply_mode_permanently", args=(mode_v,))
            for llm in self.inference_engines
        ])

    def update(self, optimizer: Optimizer):
        raise NotImplementedError("VLLMFlowBackendLoRA does not support standard optimizer updates. The flow trainer handles all updates explicitly.")

    def generate_outputs(self, genomes: List[Genome], optimizer: Optimizer, suffix: str, inputs: List[List[Dict[str, str]]]):
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        if len(genomes) > self.population_size:
            raise ValueError(f"Population size {len(genomes)} exceeds max {self.population_size}.")

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
                "set_explicit_weights_multi", args=(my_genomes.tolist(), False))
            perturb_handles.append(h)
        ray.get(perturb_handles)

        if self.time_self:
            print(f"#-- All TinyLoRA weights injected in {time.time() - start_time:.2f}s --#")
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
                my_prompts, self.sampler, lora_specs, self.use_tqdm)
            all_gen_handles.append(h)
            genome_chunks_kept.append(my_genomes)

        all_outputs = ray.get(all_gen_handles)

        if self.time_self:
            end_time = time.time()
            print(f"#-- All genome outputs generated in {end_time - gen_start_time:.2f}s --#")

        for chunk_results, genomes_in_chunk in zip(all_outputs, genome_chunks_kept):
            for genome, outputs_for_genome in zip(genomes_in_chunk, chunk_results):
                genome.latest_outputs = outputs_for_genome

        restore_handles = []
        for eng_idx, llm in enumerate(self.inference_engines):
            my_genomes = genome_chunks[eng_idx]
            if len(my_genomes) > 0:
                h = llm.collective_rpc.remote(
                    "set_explicit_weights_multi", args=(my_genomes.tolist(), True))
                restore_handles.append(h)
        ray.get(restore_handles)

        if self.time_self:
            print(f"#-- All weights restored in {time.time() - end_time:.2f}s --#")

    def save_weights_to_disk(self, filepath: str):
        ray.get(self.inference_engines[0].collective_rpc.remote("save_weights_to_disk", args=(filepath,)))

    def load_weights_from_disk(self, filepath: str):
        ray.get([llm.collective_rpc.remote("load_weights_from_disk", args=(filepath,))
                 for llm in self.inference_engines])

    def inference(self, conversations: List[List[Dict[str, str]]], suffix: str = None):
        prompts = []
        for c in conversations:
            p = self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            if suffix:
                p += suffix
            prompts.append([p])
        prompt_chunks = np.array_split(prompts, self.NUM_GPUS)

        handles = []
        lora_name = "lora_0"
        lora_path = self.lora_paths.get(lora_name)
        if not lora_path:
            lora_name = list(self.lora_paths.keys())[0]
            lora_path = self.lora_paths[lora_name]

        for i, engine in enumerate(self.inference_engines):
            chunk = prompt_chunks[i]
            if len(chunk) == 0:
                continue
            lora_specs = [{"name": lora_name, "id": 1, "path": lora_path} for _ in chunk]
            chunk_list = chunk.tolist()
            h = engine.generate_multi_lora.remote(chunk_list, self.sampler, lora_specs, self.use_tqdm)
            handles.append(h)

        results_list = ray.get(handles)
        final_outputs = []
        for eng_results in results_list:
            for genome_res in eng_results:
                final_outputs.append(genome_res[0])
        return final_outputs