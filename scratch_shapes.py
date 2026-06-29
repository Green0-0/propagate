from vllm.lora.models import LoRAMapping
from vllm.config import LoRAConfig
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", enable_lora=True, max_loras=1, max_lora_rank=2)
model = llm.llm_engine.model_executor.driver_worker.model_runner.model

for name, module in model.named_modules():
    if hasattr(module, 'lora_a_stacked'):
        print(name, module.lora_a_stacked.shape, module.lora_b_stacked.shape)
