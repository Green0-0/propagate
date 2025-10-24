import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import response
from libs.dataset import Dataset
from libs.generic_rewards import format_reward

def answer_reward_function(response: str, numbers: List[int], target: int) -> float:
    answer_regex = r"<answer>(.*?)<\/answer>"
    all_matches = re.findall(answer_regex, response, re.DOTALL)
    if not all_matches:
        return 0.0
    answer_content = all_matches[-1].strip()
   
    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not answer_content:
        return 0.0
    if not re.match(allowed_chars, answer_content):
        return 0.0
    
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0
    
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 0.91
    except:
        return 0.0
    return 0.0

    
def load_countdown_dataset(batch_size: int = 200, pairs_loaded: int = 200) -> Dataset:
    json_path = "libs/dataset_files/countdown.json"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    suffix = "<think>"
    pairs = []
    
    for item in data[0:pairs_loaded]:
        context = item["context"]
        if context.endswith("Let me solve this step by step.\n<think>"):
            context = context[:-len("Let me solve this step by step.\n<think>")]
        
        # Create ShareGPT format
        sharegpt_format = [
            {
                "role": "user",
                "content": context
            }
        ]

        def create_reward_function(numbers: List[int], target: str) -> Callable[[str], float]:
            def reward_fn(response: str) -> float:
                result = format_reward(response, 0.03, "", "</think>", "<answer>", "</answer>") + answer_reward_function(response, numbers, int(target))
                return result
            return reward_fn
        
        reward_fn = create_reward_function(item["numbers"], item["target"])
        
        pairs.append((sharegpt_format, reward_fn))

    return Dataset(batch_size, suffix, pairs)

if __name__ == "__main__":
    ds = load_countdown_dataset(30)
    print(ds.batch_size)
    print(ds.suffix)
    print(len(ds.pairs))
    print(ds.pairs[0][0])
    reward_fn = ds.pairs[0][1]
    print(reward_fn("First, I will add 25 and 100 to get 125.\nThen, I will subtract 3 to get 122.\nFinally, I will multiply by 2 to get 244.\n<answer>((44 + 19) + 35)"))
    print(reward_fn("First, I will add 25 and 100 to get 125.\nThen, I will subtract 3 to get 122.\nFinally, I will multiply by 2 to get 244.\n</think><answer>((44 + 19) + 35)"))
    print(reward_fn("First, I will add 25 and 100 to get 125.\nThen, I will subtract 3 to get 122.\nFinally, I will multiply by 2 to get 244.\n</think></answer>((44 + 19) + 35)"))
    print(reward_fn("First, I will add 25 and 100 to get 125.\nThen, I will subtract 3 to get 122.\nFinally, I will multiply by 2 to get 244.\n</think><answer>((44 + 19) + 35)</answer>"))