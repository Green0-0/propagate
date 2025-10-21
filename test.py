from libs.vllm_backend_tp import VLLMBackendTP
from libs.vllm_backend_multi import VLLMBackendMulti
from libs.countdown_dataset import load_countdown_dataset
from libs.genome import Genome
from libs.trainer import SimpleTrainer
from vllm import SamplingParams

dataset = load_countdown_dataset()
dataset.batch_size = 50

sampler = SamplingParams(
    temperature=0.05,
    top_p=0.99,
    max_tokens=1024
)
backend = VLLMBackendMulti(model_name="Qwen/Qwen2.5-3B-Instruct", NUM_GPUS=2, CPUS_PER_GPU=6, GPU_FRACTION_TRAINING_ACTOR=0.3, GPU_FRACTION_VLLM_WORKER=0.65, Sampler=sampler)
trainer = SimpleTrainer(
    population_size=10,
    learning_rate=0.0005,
    seed_weight=0.001,
    backend=backend,
    dataset=dataset
)

for i in range(250):
    trainer.train_step()