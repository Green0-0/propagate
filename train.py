
from libs.backend.vllm_backend import VLLMBackend


def load_datasets(batch_size: int = 50):
    from datasets import load_dataset, Dataset
    from libs.datasets.hf_dataset_loader import load_hf_dataset
    from libs.datasets.dataset import balanced_merge
    from libs.datasets.reward import MathVerifyRewardGenerator, LastChoiceRewardGenerator
    import re
    
    def is_float(s: str) -> bool:
        if not s:
            return False
        return re.match(r"^-?\d+(\.\d+)?$", str(s).strip()) is not None

    def format_mmlu_row(item):
        choices = item['choices']
        labels = ["A", "B", "C", "D"]
        formatted_question = item['question'] + "\n"
        for label, choice in zip(labels, choices):
            formatted_question += f"{label}. {choice}\n"
        try:
            answer_idx = int(item['answer'])
            letter_answer = labels[answer_idx]
        except (ValueError, IndexError):
            letter_answer = ""
        return {
            "formatted_question": formatted_question,
            "letter_answer": letter_answer
        }
    
    datasets = {}
    ace_hf = load_dataset("nvidia/AceReason-Math", split="train")
    ace_hf = ace_hf.shuffle(seed=42)
    print(len(ace_hf))
    ace_hf = ace_hf.select(range(49000))
    
    datasets["acereason"] = load_hf_dataset(
        batch_size=batch_size,
        hf_data=ace_hf,
        answer_reward=MathVerifyRewardGenerator(target_answer_key="answer"),
        input_column="problem",
        target_column="answer",
        force_reuse_batches=False
    )
    
    mmlu_hf = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    print(len(mmlu_hf))
    mmlu_hf = Dataset.from_list(mmlu_hf['train'])
    mmlu_hf = mmlu_hf.shuffle(seed=42)
    mmlu_hf = mmlu_hf.select(range(49000))
    mmlu_hf = mmlu_hf.map(format_mmlu_row)
    mmlu_hf = mmlu_hf.filter(lambda x: x["formatted_question"] is not None and x["letter_answer"] != "")

    datasets["mmlu"] = load_hf_dataset(
        batch_size=batch_size,
        hf_data=mmlu_hf,
        answer_reward=LastChoiceRewardGenerator(
            choices=["A", "B", "C", "D"], 
            target_answer_key="letter_answer",
            lowercase=False
        ),
        input_column="formatted_question",
        target_column="letter_answer"
    )

    mega_hf = load_dataset("MegaScience/MegaScience", split="train")
    mega_hf = mega_hf.filter(lambda x: is_float(x.get("reference_answer", "")))
    mega_hf = mega_hf.shuffle(seed=42)
    print(len(mega_hf))
    mega_hf = mega_hf.select(range(49000))
    datasets["megascience"] = load_hf_dataset(
        batch_size=batch_size,
        hf_data=mega_hf,
        answer_reward=MathVerifyRewardGenerator(target_answer_key="reference_answer"),
        input_column="question",
        target_column="reference_answer"
    )

    merged_datasets = balanced_merge([datasets["acereason"], datasets["mmlu"], datasets["megascience"]])
    datasets["merged"] = merged_datasets
    return datasets

def do_train(model_source = "Qwen/Qwen3-8B-Base", 
             lora_model_source = None,
             gpu_fraction = 0.5,
             lora_rank = 8,
             ctx_len = 1024,
             batch_size = 50,
             population_size = 30,
             total_steps = 500,
             learning_rate = 3,
             sigma = 0.06,
             momentum = 0.6,
             beta2 = 0.95,
             optimizer_name = "none",
             wandb_project = "propagate_optimizers",
             target_dataset = "merged",
             lora = False):
    from libs.backend.vllm_lorabackend import VLLMBackendLoRA
    from libs.trainer import SimpleTrainer
    from libs.optimizers import SimpleOpt, MomentumOpt, MuonOpt, AdamOpt
    from libs.optimizer_th import TwoHalvesEstimatorOpt
    from libs.optimizer_stein import SteinOpt
    from vllm import SamplingParams

    import gc
    import torch
    import ray
    
    gc.collect()
    torch.cuda.empty_cache()

    try:
        dataset = load_datasets(batch_size=batch_size)[target_dataset]
        dataset.generate_test_split(test_fraction=0.01, fold_index=1)
        if lora_model_source is None or lora_model_source == "":
            lora_model_source = model_source
        sampler = SamplingParams(temperature=0.0, max_tokens=ctx_len)
        
        if lora:
            alt_lora = True
            backend = VLLMBackendLoRA(model_name=model_source, lora_model_source=lora_model_source, NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=gpu_fraction, Sampler=sampler, lora_rank=lora_rank, use_tqdm=False, time_self=True, lora_perturb_target="b-", norm_scale_update=True)
        else:
            alt_lora = False
            backend = VLLMBackend(model_name=model_source, NUM_GPUS=4, CPUS_PER_GPU=6, GPU_FRACTION_VLLM_WORKER=gpu_fraction, sampler=sampler, use_tqdm=False, time_self=True)

        if optimizer_name == "none":
            optimizer = SimpleOpt(total_steps=total_steps, learning_rate=learning_rate, perturb_scale=sigma, norm_by_mean=False, norm_by_stddev=False)
        elif optimizer_name == "momentum":
            optimizer = MomentumOpt(total_steps=total_steps, learning_rate=learning_rate, perturb_scale=sigma, norm_by_mean=False, norm_by_stddev=False, force_lora_alternating=alt_lora, momentum=momentum)
        elif optimizer_name == "muon":
            optimizer = MuonOpt(total_steps=total_steps, learning_rate=learning_rate, perturb_scale=sigma, norm_by_mean=False, norm_by_stddev=False, force_lora_alternating=alt_lora, momentum=momentum)
        elif optimizer_name == "adam":
            optimizer = AdamOpt(total_steps=total_steps, learning_rate=learning_rate, perturb_scale=sigma, norm_by_mean=False, norm_by_stddev=False, force_lora_alternating=alt_lora, momentum=momentum, beta2=beta2)
        elif optimizer_name == "two_halves":
            optimizer = TwoHalvesEstimatorOpt(total_steps=total_steps, learning_rate=learning_rate, perturb_scale=sigma, norm_by_mean=False, norm_by_stddev=False, force_lora_alternating=alt_lora, momentum=momentum, gamma=0.25, ema_decay=beta2)
        elif optimizer_name == "stein":
            optimizer = SteinOpt(total_steps=total_steps, learning_rate=learning_rate, perturb_scale=sigma, norm_by_mean=False, norm_by_stddev=False, force_lora_alternating=alt_lora)

        trainer = SimpleTrainer(population_size=population_size,
                                mirror=True,
                                optimizer=optimizer,
                                backend=backend,
                                dataset=dataset,
                                wandb_project=wandb_project,
                                validate_every=10,
                                print_samples=True,
        )
        
        trainer.train()

        trainer.save_model_seeds("saved_model/saved_model_seeds.json")
        
        print("#-- Training complete --#")

    finally:
        ray.shutdown()
    pass

if __name__ == "__main__":
    do_train(model_source="Qwen/Qwen2.5-3B-Instruct", 
             gpu_fraction=0.8,
             lora_rank=8,
             ctx_len=4096,
             batch_size=50,
             population_size=28,
             total_steps=250,
             learning_rate=3,
             sigma=0.06,
             momentum=0.6,
             beta2=0.95,
             optimizer_name="none",
             wandb_project="propagate_optimizers",
             target_dataset = "acereason",
             lora = True)