from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union
from transformers import AutoTokenizer
import re
import requests

from propagate.datasets.postprocessreward import PostProcessReward
from propagate.datasets.reward import RewardGenerator
from propagate.genome import Genome

class RLHFFillerReward(RewardGenerator):
    """
    Basic equality reward function with regex.
    Checks if the response contains the target string within the last instance of the wrapper regex.
    """
    def __init__(self):
        pass

    def build_reward_function(self, input: Dict) -> Callable[[str], float]:
        def reward_function(response: str) -> float:
            return 777
        return reward_function
    
class RLHFJudge(PostProcessReward):
    def __init__(self, response_token_begin: str = "<response>", response_token_end: str = "</response>", model_name_or_path: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B", base_url = "http://127.0.0.1:8000/classify"):
        self.response_token_begin = response_token_begin
        self.response_token_end = response_token_end
        self.model_name_or_path = model_name_or_path
        self.base_url = base_url
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print("Warning: Requires separate RLHF Judge creation...")

    def post_process_rewards(self, genomes: List[Genome]):
        need_scoring = []
        for genome in genomes:
            for i in range(len(genome.latest_outputs)):
                if genome.latest_rewards[i] == 777:
                    processed_response = genome.latest_outputs[i].split(self.response_token_begin)[-1].split(self.response_token_end)[0].strip()
                    conv = [{"role": "user", "content": genome.latest_inputs[i]},{"role": "assistant", "content": processed_response}]
                    conv_tokenized = self.tokenizer.apply_chat_template(conv, tokenize=False)
                    if self.tokenizer.bos_token is not None and conv_tokenized.startswith(self.tokenizer.bos_token):
                        conv_tokenized = conv_tokenized[len(self.tokenizer.bos_token):]
                    need_scoring.append(conv_tokenized)

        payload = {"model": self.model_name_or_path, "text": need_scoring}
        rewards = []
        try:
            responses = requests.post(self.base_url, json=payload).json()
            for response in responses:
                rewards.append(response["embedding"][0])
            assert len(rewards) == len(need_scoring), f"Expected {len(need_scoring)} rewards, got {len(rewards)}"
        except Exception as e:
            print(f"Error: {e}")
            rewards = [0.0] * len(need_scoring)

        reward_idx = 0
        for genome in genomes:
            for i in range(len(genome.latest_outputs)):
                if genome.latest_rewards[i] == 777:
                    genome.latest_rewards[i] = rewards[reward_idx]
                    reward_idx += 1