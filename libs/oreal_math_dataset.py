from typing import Any, Callable, Dict, List, Optional, Tuple
import re
from datasets import load_dataset
from libs.dataset import Dataset
from libs.generic_rewards import basic_validator_reward, format_reward

def is_integer_string(s: str) -> bool:
    """Checks if a string represents a simple integer."""
    if s is None:
        return False
    # This regex matches optional leading minus and digits, and nothing else.
    return re.match(r"^-?\d+$", s.strip()) is not None


def load_oreal_rl_prompts_dataset(batch_size: int = 200, split: str = "train") -> Dataset:
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
        sharegpt_format = [
            {
                "role": "user",
                "content": item["question"] + "\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>"
            }
        ]

        def create_reward_function(target_answer: str) -> Callable[[str], float]:
            def reward_fn(response: str) -> float:
                format_r = format_reward(response, 0.03, "", "</think>", "<answer>", "</answer>")
                answer_r = basic_validator_reward(response, target=target_answer, wrapper_regex=r"<answer>(.*?)<\/answer>", lowercase=True) * 0.91
                return format_r + answer_r
            return reward_fn
        
        reward_fn = create_reward_function(item["gold_answer"])
        pairs.append((sharegpt_format, reward_fn))

    return Dataset(batch_size, suffix, pairs)


if __name__ == "__main__":
    print("--- Testing OREAL-RL-Prompts Dataset ---")
    ds_oreal = load_oreal_rl_prompts_dataset(batch_size=3)
    
    if len(ds_oreal.pairs_train) > 0:
        print(f"Batch size: {ds_oreal.batch_size}")
        print(f"Suffix: {ds_oreal.suffix}")
        print(f"Total pairs (filtered): {len(ds_oreal.pairs_train)}")
        
        first_prompt_gpt = ds_oreal.pairs_train[0][0]
        reward_fn_oreal = ds_oreal.pairs_train[0][1]
        
        hf_data_test = load_dataset("internlm/OREAL-RL-Prompts", split="train")
        first_good_item = next(
            item for item in hf_data_test 
            if is_integer_string(item["gold_answer"])
        )
        gold_answer = first_good_item["gold_answer"]
        
        print(f"First pair prompt: {first_prompt_gpt}")
        print(f"Corresponding Gold Answer: {gold_answer}")

        print("\nTesting reward function:")
        test_str_1 = "Some random thoughts."
        print(f"Test 1 (Bad): '{test_str_1}' -> Reward: {reward_fn_oreal(test_str_1)}")
        
        test_str_2 = "</think><answer>12345</answer>"
        print(f"Test 2 (Good format, wrong answer): '{test_str_2}' -> Reward: {reward_fn_oreal(test_str_2)}")
        
        test_str_3 = f"</think><answer>{gold_answer}</answer>"
        print(f"Test 3 (Good format, good answer): '{test_str_3}' -> Reward: {reward_fn_oreal(test_str_3)}")
        
        test_str_4 = f"<answer>{gold_answer}</answer>"
        print(f"Test 4 (Missing </think>): '{test_str_4}' -> Reward: {reward_fn_oreal(test_str_4)}")

        test_str_5 = f"</think><answer>1 2 3 4 5 {gold_answer}</answer>"
        print(f"Test 5 (Good format, messed up answers): '{test_str_5}' -> Reward: {reward_fn_oreal(test_str_5)}")

    else:
        print("No samples found with integer gold answers.")