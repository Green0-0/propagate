import json
import re
from reward import RewardGenerator, FormatRewardGenerator
from typing import Callable, List
from libs.datasets.dataset import Dataset

class AnswerRewardGenerator(RewardGenerator):
    """
    Reward function that checks if the final answer in the response matches the target answer,
    and whether only the provided numbers are used in the calculation.
    """
    def __init__(self, numbers_key: str, target_key: str):
        self.numbers_key = numbers_key
        self.target_key = target_key

    def build_reward_function(self, input: dict) -> Callable[[str], float]:
        numbers = input[self.numbers_key]
        target = input[self.target_key]

        def reward_function(response: str) -> float:
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
                    return 1
            except:
                return 0.0
            return 0.0
        return reward_function

def load_countdown_dataset(batch_size: int = 300, reward_func_ratio: float = 0.1, passk: int = 1, passk_proportion: float = 0.1, passk_minimum: float = 0.9) -> Dataset:
    json_path = "libs/datasets/dataset_files/countdown.json"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    suffix = "<think>"
    pairs = []
    
    for item in data:
        context = item["context"]
        if context.endswith("Let me solve this step by step.\n<think>"):
            context = context[:-len("Let me solve this step by step.\n<think>")]
        
        sharegpt_format = [
            {
                "role": "user",
                "content": context
            }
        ]

        pairs.append({"input": sharegpt_format, "numbers": item["numbers"], "target": item["target"]})

    return Dataset(batch_size, suffix, "input", pairs, AnswerRewardGenerator(numbers_key="numbers", target_key="target"), FormatRewardGenerator(start_think_token="", end_think_token="</think>", start_answer_token="<answer>", end_answer_token="</answer>"), force_reuse_batches=False, reward_func_ratio=reward_func_ratio, passk=passk, passk_proportion=passk_proportion, passk_minimum=passk_minimum)
