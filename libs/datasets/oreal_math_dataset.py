from typing import Any, Dict, List, Optional
import re
from datasets import load_dataset
from transformers import AutoTokenizer

# Import your new library structures
from libs.datasets.dataset import Dataset
from libs.datasets.reward import FormatRewardGenerator, RegexRewardGenerator

def is_integer_string(s: str) -> bool:
    """Checks if a string represents a simple integer."""
    if s is None:
        return False
    return re.match(r"^-?\d+$", s.strip()) is not None

def load_oreal_rl_prompts_dataset(
    batch_size: int = 50, 
    split: str = "train",
    reward_func_ratio: float = 0.1,
    passk: int = 1,
    passk_proportion: float = 0.1,
    passk_minimum: float = 0.9,
    force_reuse_batches: bool = False
) -> Dataset:
    print("Loading OREAL-RL-Prompts dataset...")
    hf_data = load_dataset("internlm/OREAL-RL-Prompts", split=split)
    
    filtered_data = [
        item for item in hf_data 
        if is_integer_string(item["gold_answer"])
    ]
    print(f"Original size: {len(hf_data)}, Filtered size (integer answers only): {len(filtered_data)}")

    suffix = "<think>"
    pairs = []
    
    for item in filtered_data:
        instruction = (
            item["question"] + 
            "\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>"
        )

        sharegpt_format = [
            {
                "role": "user",
                "content": instruction
            }
        ]
        
        pairs.append({
            "input": sharegpt_format, 
            "gold_answer": item["gold_answer"]
        })

    answer_reward_gen = RegexRewardGenerator(
        target_key="gold_answer", 
        wrapper_regex=r"<answer>(.*?)<\/answer>", 
        lowercase=True
    )
    
    format_reward_gen = FormatRewardGenerator(
        start_think_token="", 
        end_think_token="</think>", 
        start_answer_token="<answer>", 
        end_answer_token="</answer>"
    )

    return Dataset(
        batch_size=batch_size,
        suffix=suffix,
        dataset_input_key="input",
        dataset_pairs=pairs,
        answer_reward=answer_reward_gen,
        format_reward=format_reward_gen,
        force_reuse_batches=force_reuse_batches,
        reward_func_ratio=reward_func_ratio,
        passk=passk,
        passk_proportion=passk_proportion,
        passk_minimum=passk_minimum
    )