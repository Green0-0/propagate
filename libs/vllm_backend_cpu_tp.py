from typing import Dict, List
from libs.backend import Backend

import os
import torch
import time 
import numpy as np-
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from libs.genome import Genome

class VLLMBackendTP(Backend):
    llm: LLM
    tokenizer: AutoTokenizer
    sampler: SamplingParams

    output_log_file: str
    full_output_log_file: str

    def __init__(self, model_name: str, NUM_GPUS: int, GPU_FRACTION_VLLM: float, Sampler: SamplingParams, output_log_file: str = "logs/output.log", full_output_log_file: str = "logs/full_output.log", use_tqdm: bool = False, time_self: bool = True):
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        print("#-- Initializing Backend [VLLMBackendCPU+TP] --#")
        print(f"#-- GPUS: {NUM_GPUS}, GPU Fraction VLLM Worker: {GPU_FRACTION_VLLM} --#")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampler = Sampler
        self.use_tqdm = use_tqdm
        self.time_self = time_self

        self.output_log_file = output_log_file
        self.full_output_log_file = full_output_log_file
        open(self.output_log_file, "w", encoding="utf-8").close()
        open(self.full_output_log_file, "w", encoding="utf-8").close()

        print(f"#-- Spawning singular vLLM instance with TP --#")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.llm = LLM(model=model_name, enforce_eager=True, tensor_parallel_size=max(1, NUM_GPUS), gpu_memory_utilization=GPU_FRACTION_VLLM)

        print("#-- Backend Initialized --#")
        pass

    def echo_model_runner(self):
        return self.model_runner.model.__class__

    def update(self, genome: Genome):
        """Update the model permanently with a genome as the source."""
        genome.update_tensor(self.model.named_parameters(), device="cpu")
        ret = self.llm.collective_rpc(echo_model_runner)
        self.ret = ret[0].driver_worker.model_runner.model.load_weights(self.model.named_parameters())

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
            start_time = time.time()
            g.update_tensor(self.model.named_parameters(), device="cpu")
            ret = self.llm.collective_rpc(echo_model_runner)
            self.ret = ret[0].driver_worker.model_runner.model.load_weights(self.model.named_parameters())
            
            response = self.llm.generate(prompts, self.sampler, use_tqdm=self.use_tqdm)
            batch_texts = [out.text for out in response.outputs]

            g.latest_outputs = batch_texts
            
            g.restore_tensor(self.model.named_parameters(), device="cpu")
            torch.cuda.empty_cache()

            if self.time_self:
                end_time = time.time()
                print(f"#-- Genome outputs generated in {end_time - start_time:.2f} seconds --#")


