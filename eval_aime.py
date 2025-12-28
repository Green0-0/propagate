import os
import re
import argparse
import pandas as pd
from datasets import load_dataset, Dataset
from libs.backend.vllm_lorabackend import VLLMBackendLoRA
from vllm import SamplingParams
from math_verify import parse, verify
import ray
from huggingface_hub import HfApi

class MockTrainer:
    def __init__(self):
        self.population_size = 1
        self.mirror = False

def eval_aime():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="saved_model/G-reen_SmolLM3-3B-SFT.pt", help="Path to the LoRA checkpoint .pt file")
    parser.add_argument("--base_model", type=str, default="G-reen/SmolLM3-3B-SFT", help="Base model name")
    parser.add_argument("--dataset_name", type=str, default="MathArena/aime_2025", help="Dataset to evaluate")
    parser.add_argument("--upload_dataset", type=str, default="G-reen/aime_2025_results_smollm", help="Dataset to upload results to")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset_name}...")
    try:
        dataset = load_dataset(args.dataset_name, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize Backend
    # Note: Using temp 0 for pass@1
    # MATCH TRAIN.PY: Increased max_tokens to 8192 to match training context
    sampler = SamplingParams(temperature=0.0, max_tokens=8192) 
    
    # We use VLLMBackendLoRA as requested
    backend = VLLMBackendLoRA(
        model_name=args.base_model,
        NUM_GPUS=1, # Default to 1 GPU, adjust if needed
        CPUS_PER_GPU=4,
        GPU_FRACTION_VLLM_WORKER=0.8,
        Sampler=sampler,
        lora_rank=args.lora_rank,
        use_tqdm=True,
        max_model_len=8192,
        # MATCH TRAIN.PY: Explicitly matching training backend config
        lora_perturb_target="b-",
        norm_scale_update=True
    )

    try:
        print("Starting VLLM Backend...")
        mock_trainer = MockTrainer()
        backend.startup(trainer=mock_trainer) 
        
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}...")
            backend.load_weights_from_disk(args.checkpoint)
        else:
            print(f"WARNING: Checkpoint {args.checkpoint} not found! Running with base model/random adapter weights.")

        results = []
        
        # Prepare all inputs first to batch
        # MATCH TRAIN.PY: Use the same prompt template as libs/datasets/hf_datasets_loader.py
        prompt_template = "{prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>"
        
        all_conversations = []
        for row in dataset:
            # Apply training prompt template
            content = prompt_template.format(prompt=row['problem'])
            all_conversations.append([{"role": "user", "content": content}])

        print(f"Running inference on {len(all_conversations)} items...")
        
        all_outputs = []
        
        for i in range(0, len(all_conversations), args.batch_size):
            batch = all_conversations[i:i+args.batch_size]
            outputs = backend.inference(batch, suffix="<think>")
            all_outputs.extend(outputs)
            print(f"Processed {len(all_outputs)}/{len(all_conversations)}")

        correct_count = 0
        
        print("Evaluating results...")
        for i, row in enumerate(dataset):
            problem = row['problem']
            source_answer = row['answer'] 
            model_response = all_outputs[i]
            
            # Gold Parse
            # Ensure source_answer is string for parsing
            gold_parsed = parse(str(source_answer))
            
            # Match training reward logic: strictly extract from <answer> tags
            answer_pattern = r"<answer>(.*?)<\/answer>"
            matches = re.findall(answer_pattern, model_response, re.DOTALL)
            if matches:
                text_to_parse = matches[-1]
            else:
                # If no tags, we treat it as a format failure and parse nothing (fail)
                text_to_parse = ""
            
            # Pred Parse
            pred_parsed = parse(text_to_parse)
            
            # Verify
            is_correct = verify(gold_parsed, pred_parsed)
            
            # Extract simple string representation for CSV
            # pred_parsed is a list of distinct extracted answers. 
            # We take the last one as the "final" answer if available, or just the string repr
            model_answer_extracted = str(pred_parsed)

            if is_correct:
                correct_count += 1
            
            results.append({
                "problem": problem,
                "source_answer": source_answer,
                "model_response": model_response,
                "model_answer_extracted": model_answer_extracted,
                "is_correct": is_correct
            })

        score = correct_count / len(dataset)
        print(f"Accuracy: {score:.2%}")

        # Upload
        print(f"Uploading to {args.upload_dataset}...")
        res_df = pd.DataFrame(results)
        res_dataset = Dataset.from_pandas(res_df)
        
        res_dataset.push_to_hub(args.upload_dataset)
        
        # Upload Readme
        readme_content = f"""---
license: mit
---
# Evaluation Results

- **Base Model:** {args.base_model}
- **Dataset:** {args.dataset_name}
- **Accuracy (pass@1):** {score:.2%}

## Details
- **LoRA Rank:** {args.lora_rank}
- **Temperature:** 0.0
"""
        with open("README_EVAL.md", "w") as f:
            f.write(readme_content)
            
        api = HfApi()
        try:
            api.upload_file(
                path_or_fileobj="README_EVAL.md",
                path_in_repo="README.md",
                repo_id=args.upload_dataset,
                repo_type="dataset"
            )
            print("Readme updated.")
        except Exception as e:
            print(f"Could not upload README: {e}")

        print("Done.")

    finally:
        ray.shutdown()

if __name__ == "__main__":
    eval_aime()
