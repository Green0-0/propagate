import math
from typing import Dict, List
from uuid import uuid4
from libs.backend.backend_abc import Backend

import signal
import sys
import os
from libs.trainer import SimpleTrainer
import ray
import torch
import time 
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

from libs.genome import Genome

from peft import LoraConfig, get_peft_model
from vllm.lora.request import LoRARequest
import shutil
import tempfile
import gc

from libs.optimizers import Optimizer

class VLLMBackendLoRA(Backend):
    def __init__(self, model_name: str, NUM_GPUS: int, CPUS_PER_GPU: int, GPU_FRACTION_VLLM_WORKER: float, Sampler: SamplingParams, use_tqdm: bool = False, time_self: bool = False, max_model_len: int = 4096, lora_rank: int = 8, lora_perturb_target: str = "b-", init_lora_weights: str = True, lora_model_source: str = None, norm_scale_update: bool = True, repeat_tokens_buffer_count: int = 20, repeat_times_kill: int = 15, rep_check_every: int = 100, repeat_tokens_begin_scan_count: int = 500, repeat_tokens_lookback_length: int = 500):
        super().__init__(backend_name=f"Rank {str(lora_rank)} LoRA vLLM Backend, Perturb Target: {lora_perturb_target}, Init Method: {init_lora_weights}", model_name=model_name, NUM_GPUS=NUM_GPUS, CPUS_PER_GPU=CPUS_PER_GPU, GPU_FRACTION_VLLM_WORKER=GPU_FRACTION_VLLM_WORKER, sampler=Sampler, use_tqdm=use_tqdm, max_model_len=max_model_len, time_self=time_self)
        self.lora_model_source = lora_model_source if lora_model_source is not None else model_name
        self.lora_rank = lora_rank
        self.lora_perturb_target: str = lora_perturb_target
        self.init_lora_weights = init_lora_weights
        self.norm_scale_update = norm_scale_update
        if "a" not in lora_perturb_target.lower() and "b" not in lora_perturb_target.lower():
            raise ValueError(f"Invalid lora_perturb_target: {lora_perturb_target}. Must be 'a' or 'b' or 'a-' or 'b-' or 'ab'.")
        
        self.repeat_tokens_buffer_count = repeat_tokens_buffer_count
        self.repeat_times_kill = repeat_times_kill
        self.rep_check_every = rep_check_every
        self.repeat_tokens_begin_scan_count = repeat_tokens_begin_scan_count
        self.repeat_tokens_lookback_length = repeat_tokens_lookback_length

    def startup(self, trainer: SimpleTrainer):
        """Initializes the vLLM backend with Ray actors and placement groups."""
        os.environ.pop("RAY_ADDRESS", None)
        os.environ.pop("RAY_HEAD_IP", None)
        os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
        pass_gpu_fraction = str(self.GPU_FRACTION_VLLM_WORKER)
        #--------------------------------------------------------#
        #                CUSTOM CLASSES DEFINITION               #
        #--------------------------------------------------------#
        class MyLLM(LLM):
            def __init__(self, repeat_tokens_buffer_count, repeat_times_kill, rep_check_every, repeat_tokens_begin_scan_count, repeat_tokens_lookback_length, *args, **kwargs):
                self.repeat_tokens_buffer_count = repeat_tokens_buffer_count
                self.repeat_times_kill = repeat_times_kill
                self.rep_check_every = rep_check_every
                self.repeat_tokens_begin_scan_count = repeat_tokens_begin_scan_count
                self.repeat_tokens_lookback_length = repeat_tokens_lookback_length

                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                os.environ["VLLM_RAY_PER_WORKER_GPUS"] = pass_gpu_fraction
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                super().__init__(*args, **kwargs)

            def generate_multi_lora(self, prompts, sampling_params, lora_specs, use_tqdm=False):
                from uuid import uuid4

                try:
                    from tqdm.auto import tqdm as _tqdm
                    _tqdm_available = True
                except Exception:
                    _tqdm_available = False
                    _tqdm = None

                engine = self.llm_engine

                # Map request_id -> (genome_idx, prompt_idx)
                req_to_idx = {}
                # Results buffer
                results = []

                # Track per-request generated token counts to compute accurate tok/s
                prev_gen_tokens = {}  # rid -> last seen generated token count
                total_generated_tokens = 0
                completed = 0

                # Enqueue all requests
                for g_idx, (spec, genome_prompts) in enumerate(zip(lora_specs, prompts)):
                    results.append([None] * len(genome_prompts))
                    lora_req = LoRARequest(spec["name"], spec["id"], spec["path"])
                    for p_idx, single_prompt_str in enumerate(genome_prompts):
                        rid = f"{g_idx}:{p_idx}:{uuid4().hex}"
                        engine.add_request(
                            request_id=rid,
                            prompt=single_prompt_str,
                            params=sampling_params, 
                            lora_request=lora_req,
                        )
                        
                        req_to_idx[rid] = (g_idx, p_idx)
                        prev_gen_tokens[rid] = 0
                remaining = len(req_to_idx)
                start_t = time.perf_counter()

                pbar = None
                if use_tqdm and _tqdm_available:
                    pbar = _tqdm(total=remaining, desc="vLLM multi-LoRA", leave=True)

                # Drive engine until all requests finish
                global_step_counter = 0
                while remaining > 0:
                    request_outputs = engine.step()
                    global_step_counter += 1

                    step_new_tokens = 0

                    should_check_loops = (global_step_counter % self.rep_check_every == 0)

                    for ro in request_outputs:
                        # Accumulate generated tokens incrementally for throughput stats
                        if ro.request_id not in req_to_idx:
                            continue

                        out = ro.outputs[0] if (hasattr(ro, "outputs") and ro.outputs) else None

                        loop_detected = False
                        if should_check_loops and out is not None and hasattr(out, "token_ids"):
                            curr_tokens = out.token_ids
                            curr_len = len(curr_tokens)
                            
                            if curr_len >= self.repeat_tokens_begin_scan_count and curr_len >= self.repeat_tokens_buffer_count:
                                
                                tail = curr_tokens[-self.repeat_tokens_buffer_count:]
                                seq_len = self.repeat_tokens_buffer_count
                                count = 0
                                limit_threshold = self.repeat_times_kill + 1

                                scan_end_limit = max(-1, curr_len - self.repeat_tokens_lookback_length - 1)
                                
                                for i in range(curr_len - seq_len, scan_end_limit, -1):
                                    if curr_tokens[i : i + seq_len] == tail:
                                        count += 1
                                        if count > limit_threshold:
                                            loop_detected = True
                                            break

                        if out is not None and hasattr(out, "token_ids"):
                            curr = len(out.token_ids)
                            prev = prev_gen_tokens.get(ro.request_id, 0)
                            if curr > prev:
                                step_new_tokens += (curr - prev)
                                prev_gen_tokens[ro.request_id] = curr

                        if loop_detected or getattr(ro, "finished", False):
                            if loop_detected:
                                engine.abort_request(ro.request_id)
                            
                            g_idx, p_idx = req_to_idx.pop(ro.request_id)
                            text = out.text if (out is not None and hasattr(out, "text")) else ""
                            results[g_idx][p_idx] = text

                            remaining -= 1
                            completed += 1
                            if pbar is not None:
                                pbar.update(1)

                    # Update running throughput stats
                    if step_new_tokens > 0:
                        total_generated_tokens += step_new_tokens
                        if pbar is not None:
                            elapsed = max(1e-6, time.perf_counter() - start_t)
                            tok_s = total_generated_tokens / elapsed
                            req_s = completed / elapsed
                            pbar.set_postfix_str(f"tok/s={tok_s:.1f}, req/s={req_s:.2f}, gen_tok={total_generated_tokens}")

                if pbar is not None:
                    # Final stats update
                    elapsed = max(1e-6, time.perf_counter() - start_t)
                    tok_s = total_generated_tokens / elapsed
                    req_s = completed / elapsed
                    pbar.set_postfix_str(f"tok/s={tok_s:.1f}, req/s={req_s:.2f}, gen_tok={total_generated_tokens}")
                    pbar.close()

                return results
        #-----------------------------------------------------#
        print(f"#-- Initializing Backend {self.backend_name} --#")
        print(f"#-- GPUS: {self.NUM_GPUS}, CPUS per GPU: {self.CPUS_PER_GPU}, GPU Fraction VLLM Worker: {self.GPU_FRACTION_VLLM_WORKER} --#")

        print("#-- Spawning Training Actors with vLLM backends --#")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)
        pgs = [placement_group([{"GPU": 1, "CPU": self.CPUS_PER_GPU}]) for _ in range(self.NUM_GPUS)]
        ray.get([pg.ready() for pg in pgs])
        strategies = [PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0) for pg in pgs]
        
        self.population_size = trainer.population_size
        if trainer.mirror:
            self.population_size = self.population_size * 2
        max_loras_per_worker = math.ceil(self.population_size / self.NUM_GPUS)

        self.inference_engines = [
            ray.remote(
                num_cpus=0,
                num_gpus=0,
                scheduling_strategy=strategy,
            )(MyLLM).remote(
                repeat_tokens_buffer_count=self.repeat_tokens_buffer_count,
                repeat_times_kill=self.repeat_times_kill,
                rep_check_every=self.rep_check_every,
                repeat_tokens_begin_scan_count=self.repeat_tokens_begin_scan_count,
                repeat_tokens_lookback_length=self.repeat_tokens_lookback_length,
                model=self.model_name,
                tensor_parallel_size=1,
                distributed_executor_backend="ray",
                worker_extension_cls="libs.backend.vllm_lorautils.WorkerExtension",
                dtype="float16",
                enable_prefix_caching=False,
                enforce_eager=False,
                gpu_memory_utilization=self.GPU_FRACTION_VLLM_WORKER,
                enable_lora=True,
                max_loras=max_loras_per_worker,
                max_lora_rank=max(self.lora_rank, 8),
                max_cpu_loras=1000,
                max_model_len=self.max_model_len,
            )
            for strategy in strategies
        ]

        if self.NUM_GPUS > 1:
            print("#-- Initializing Ray Collective group for GPU sync --#")
            ray.get([llm.collective_rpc.remote("init_collective_group", args=(self.NUM_GPUS, rank,)) for rank, llm in enumerate(self.inference_engines)])
        else:
            print("#-- Skipping collective group (1 GPU) --#")

        def cleanup():
            for llm in self.inference_engines:
                try:
                    ray.kill(llm)
                except Exception:
                    pass
            for pg in pgs:
                try:
                    remove_placement_group(pg)
                except Exception:
                    pass
            ray.shutdown()

        def sig_handler(sig, frame):
            cleanup()
            try:
                if hasattr(self, "_lora_tmp_root") and os.path.exists(self._lora_tmp_root):
                    shutil.rmtree(self._lora_tmp_root, ignore_errors=True)
            except Exception as e:
                print(f"#-- Error cleaning up temp LoRA dir: {e} --#")
            sys.exit(0)

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        print("#-- Creating LoRA adapters --#")
        default_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        base_model = AutoModelForCausalLM.from_pretrained(
            self.lora_model_source,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        
        if self.init_lora_weights == "zero":
            print(f"#-- Initializing rank {self.lora_rank} LoRA adapters with zeros --#")
            lora_cfg = LoraConfig(
                r=self.lora_rank,
                lora_alpha=1,
                target_modules=default_target_modules,
                use_rslora=True,
                init_lora_weights=False
            )
            peft_model = get_peft_model(base_model, lora_cfg)
            with torch.no_grad():
                for name, param in peft_model.named_parameters():
                    if "lora_" in name:
                        param.data.zero_()
        else:
            print(f"#-- Initializing rank {self.lora_rank} LoRA adapters with {self.init_lora_weights} --#")
            lora_cfg = LoraConfig(
                r=self.lora_rank,
                lora_alpha=1,
                target_modules=default_target_modules,
                init_lora_weights=self.init_lora_weights,
                use_rslora=True
            )
            peft_model = get_peft_model(base_model, lora_cfg)

        self._lora_tmp_root = tempfile.mkdtemp(prefix="vllm_loras_")

        lora_names = [f"lora_{i}" for i in range(max_loras_per_worker)]
        self.lora_paths = {} 
        for name in lora_names:
            p = os.path.join(self._lora_tmp_root, name) 
            os.makedirs(p, exist_ok=True)
            peft_model.save_pretrained(p, safe_serialization=True)
            self.lora_paths[name] = p

        del peft_model
        del base_model
        gc.collect()

        print(f"#-- Preloading {len(lora_names)} LoRA adapters into {len(self.inference_engines)} engine(s) --#")
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        dummy_prompt = [" "]
        for llm in self.inference_engines:
            for idx, name in enumerate(lora_names):
                lora_req = LoRARequest(name, idx + 1, self.lora_paths[name])
                ray.get(llm.generate.remote(dummy_prompt, sp, lora_request=lora_req, use_tqdm=False))

        print("#-- Locating LoRA adapters --#")
        ray.get(self.inference_engines[0].collective_rpc.remote("self_report_lora_params_sanity_check"))
        print("#-- Backend Initialized --#")
        
    def update(self, optimizer: Optimizer):
        """Update the model permanently with a genome as the source."""
        ray.get([llm.collective_rpc.remote("update_weights", args=(optimizer, self.lora_perturb_target, self.norm_scale_update)) for llm in self.inference_engines])

        ray.get([llm.collective_rpc.remote("perform_global_average_lora") for llm in self.inference_engines])

        if self.lora_perturb_target == "a-":
            self.lora_perturb_target = "b-"
        elif self.lora_perturb_target == "b-":
            self.lora_perturb_target = "a-"

    def generate_outputs(self, genomes: List[Genome], suffix: str, inputs: List[List[Dict[str, str]]]):
        assert len(genomes) == len(inputs), "Number of genomes must match number of input sets."
        if len(genomes) > self.population_size:
            raise ValueError(f"Population size {len(genomes)} exceeds max population size {self.population_size} for this backend.")
        
        prompts = []
        for i in inputs:
            prompt_genome = []
            for j in i:
                s = self.tokenizer.apply_chat_template(j, tokenize=False, add_generation_prompt=True)
                if suffix is not None:
                    s = s + suffix
                prompt_genome.append(s)
            prompts.append(prompt_genome)

        genome_chunks = np.array_split(genomes, self.NUM_GPUS)
        prompt_chunks = np.array_split(prompts, self.NUM_GPUS)

        if self.time_self:
            start_time = time.time()
        
        perturb_handles = []
        for eng_idx, llm in enumerate(self.inference_engines):
            #ray.get(llm.collective_rpc.remote("inspect_lora", args=("PRE PERTURB INSPECTION", )))
            my_genomes = genome_chunks[eng_idx]
            
            if len(my_genomes) == 0:
                continue
                
            h = llm.collective_rpc.remote(
                "perturb_self_weights_multi", args=(my_genomes.tolist(), self.lora_perturb_target,)
            )
            perturb_handles.append(h)
        ray.get(perturb_handles)
        if self.time_self:
            print(f"#-- All adapters perturbed in {time.time() - start_time:.2f}s --#")
            gen_start_time = time.time()

        all_gen_handles = []
        genome_chunks_kept = []  # to map results back

        for eng_idx, llm in enumerate(self.inference_engines):
            my_genomes = genome_chunks[eng_idx]
            my_prompts = prompt_chunks[eng_idx]
            if len(my_genomes) == 0:
                continue

            lora_specs = []
            for local_idx, _genome in enumerate(my_genomes):
                local_lora_id = local_idx + 1
                local_lora_name = f"lora_{local_idx}"
                try:
                    lora_path = self.lora_paths[local_lora_name]
                except KeyError:
                    print(f"Error: No preloaded path found for {local_lora_name}")
                    print(f"Available paths: {list(self.lora_paths.keys())}")
                    raise
                lora_specs.append({"name": local_lora_name, "id": local_lora_id, "path": lora_path})

            h = llm.generate_multi_lora.remote(
                my_prompts,
                self.sampler,
                lora_specs,
                self.use_tqdm
            )
            all_gen_handles.append(h)
            genome_chunks_kept.append(my_genomes)

        #ray.get(llm.collective_rpc.remote("inspect_lora", args=("POST PERTURB INSPECTION",)))
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
                    "restore_self_weights_multi", args=(my_genomes.tolist(), self.lora_perturb_target,)
                )
                restore_handles.append(h)
        
        ray.get(restore_handles)
        ray.get([llm.collective_rpc.remote("perform_global_average_lora") for llm in self.inference_engines])
        #ray.get(llm.collective_rpc.remote("inspect_lora", args=("POST RESTORE INSPECTION", )))
        if self.time_self:
            print(f"#-- All adapters restored in {time.time() - end_time:.2f}s --#")

    def save_weights_to_disk(self, filepath: str):
        ray.get(self.inference_engines[0].collective_rpc.remote("save_weights_to_disk", args=(filepath,)))