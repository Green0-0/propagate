import argparse
import pandas as pd
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from math_verify import parse, verify
from huggingface_hub import HfApi

def eval_comparison():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B-Instruct", help="Model to evaluate")
    parser.add_argument("--dataset_name", type=str, default="MathArena/aime_2025", help="Dataset to evaluate")
    parser.add_argument("--upload_dataset", type=str, default="G-reen/aime_2025_results_smollm_base", help="Dataset to upload results to")
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset_name}...")
    try:
        dataset = load_dataset(args.dataset_name, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize Tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Initialize VLLM LLM
    # Using tensor_parallel_size=1 and gpu_memory_utilization=0.8 to match previous script environment
    print(f"Initializing LLM: {args.model_name}")
    llm = LLM(
        model=args.model_name, 
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.8,
        max_model_len=32000,
        enforce_eager=False
    )
    
    # Sampling params: Pass@1, Temp 0
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32000)

    # Prepare inputs
    prompts = []
    print(f"Preparing {len(dataset)} prompts...")
    for row in dataset:
        conversation = [{"role": "user", "content": row['problem']}]
        # simple chat template application
        try:
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback if no chat template exists (e.g. base model)
            # This is a basic zero-shot format often used for base models if chat template fails
            prompt = f"Problem: {row['problem']}\nAnswer:"
        prompts.append(prompt)

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Sort outputs by index to ensure alignment (vLLM usually preserves order but good to be safe if async)
    # Actually llm.generate is synchronous and returns in order for list input.
    
    results = []
    correct_count = 0
    
    print("Evaluating results...")
    for i, request_output in enumerate(outputs):
        row = dataset[i]
        problem = row['problem']
        source_answer = row['answer']
        
        # Get generated text
        model_response = request_output.outputs[0].text
        
        # Parse Gold
        gold_parsed = parse(str(source_answer))
        
        # Parse Pred
        pred_parsed = parse(model_response)
        
        # Verify
        is_correct = verify(gold_parsed, pred_parsed)
        
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
    
    try:
        res_dataset.push_to_hub(args.upload_dataset)
        
        # Upload Readme
        readme_content = f"""---
license: mit
---
# Evaluation Results (Baseline)

- **Model:** {args.model_name}
- **Dataset:** {args.dataset_name}
- **Accuracy (pass@1):** {score:.2%}

## Details
- **Type:** Baseline Comparison
- **Temperature:** 0.0
"""
        with open("README_EVAL_BASE.md", "w") as f:
            f.write(readme_content)
            
        api = HfApi()
        api.upload_file(
            path_or_fileobj="README_EVAL_BASE.md",
            path_in_repo="README.md",
            repo_id=args.upload_dataset,
            repo_type="dataset"
        )
        print("Readme updated.")
    except Exception as e:
        print(f"Error uploading results: {e}")

    print("Done.")

if __name__ == "__main__":
    eval_comparison()
