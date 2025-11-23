from abc import ABC, abstractmethod
from typing import Callable, Dict
import re

class RewardGenerator(ABC):
    @abstractmethod
    def build_reward_function(self, input: Dict) -> Callable[[str], float]:
        """Builds a reward function for the given row of data."""
        pass

class FormatRewardGenerator(RewardGenerator):
    """
    Checks if the response follows the following format: {start_think_token}...{end_think_token}...{start_answer_token}...{end_answer_token}.
    With only exactly one of each token, and nothing after the {end_answer_token}.
    If any of the tokens are none, ignore them in the checking.
    """
    def __init__(self, start_think_token: str = "", end_think_token: str = "", start_answer_token: str = "", end_answer_token: str = ""):
        self.start_think_token = start_think_token
        self.end_think_token = end_think_token
        self.start_answer_token = start_answer_token
        self.end_answer_token = end_answer_token
        self.total_possible_rewards = 0.0 + (start_think_token != None and start_think_token != "") + (end_think_token != None and end_think_token != "") + (start_answer_token != None and start_answer_token != "") + (end_answer_token != None and end_answer_token != "")

    def build_reward_function(self, input: Dict) -> Callable[[str], float]:
        def reward_function(response: str) -> float:
            response = response.strip()
    
            format_rewards = 0.0
            if self.start_think_token != None and self.start_think_token != "" and response.startswith(self.start_think_token) and response.count(self.start_think_token) == 1:
                format_rewards += 1

            if self.end_think_token != None and self.end_think_token != "" and response.count(self.end_think_token) == 1:
                format_rewards += 1

            if self.start_answer_token != None and self.start_answer_token != "" and response.count(self.start_answer_token) == 1:
                if self.end_think_token != None and self.end_think_token != "":
                    if response.find(self.start_answer_token) > response.find(self.end_think_token):
                        format_rewards += 1
                else:
                    if self.start_think_token != None and self.start_think_token != "":
                        if response.find(self.start_answer_token) > response.find(self.start_think_token):
                            format_rewards += 1
                    else:
                        format_rewards += 1

            if self.end_answer_token != None and self.end_answer_token != "" and response.endswith(self.end_answer_token) and response.count(self.end_answer_token) == 1:
                format_rewards += 1
            return format_rewards / self.total_possible_rewards if self.total_possible_rewards > 0 else 0.0
        return reward_function

class RegexRewardGenerator(RewardGenerator):
    """
    Basic equality reward function with regex.
    Checks if the response contains the target string within the last instance of the wrapper regex.
    """
    def __init__(self, target_key: str, wrapper_regex: str, lowercase: bool = True):
        self.wrapper_regex = wrapper_regex
        self.lowercase = lowercase
        self.target_key = target_key

    def build_reward_function(self, input: Dict) -> Callable[[str], float]:
        target = str(input[self.target_key]).strip()
        if self.lowercase:
            target = target.lower()
            
        def reward_function(response: str) -> float:
            try:
                matches = re.findall(self.wrapper_regex, response, re.DOTALL)
            except re.error as e:
                print(f"Regex Error: {e}")
                return 0.0

            if not matches:
                return 0.0

            last_match = matches[-1]

            if isinstance(last_match, tuple):
                extracted = " ".join(last_match)
            else:
                extracted = last_match
            extracted = extracted.strip()
            if self.lowercase:
                extracted = extracted.lower()

            return 1.0 if target == extracted else 0.0
        return reward_function
    
class LastMatchRewardGenerator(RewardGenerator):
    """
    Reward function that extracts the last value which fits the given datatype and checks if it is equal to the target.
    """
    def __init__(self, target_key: str, target_type: type):
        self.target_key = target_key
        self.target_type = target_type

        if self.target_type == int:
            self.pattern = r'-?[\d,]+'
        elif self.target_type == float:
            self.pattern = r'-?[\d,]+(?:\.\d+)?'
        else:
            raise ValueError("LastMatchRewardGenerator only supports int and float target types.")

    def build_reward_function(self, input: Dict) -> Callable[[str], float]:
        target_raw = str(input.get(self.target_key, "")).strip()
        if self.target_type == int:
            target = int(target_raw.replace(",", ""))
        else:
            target = float(target_raw.replace(",", ""))
        def reward_function(response: str) -> float:
            matches = re.findall(self.pattern, response)
            for match in reversed(matches):
                try:
                    clean_match = match.replace(",", "")
                    if self.target_type == int:
                        value = int(clean_match)
                    else:
                        value = float(clean_match)
                    if self.target_type == float:
                        return 1.0 if abs(value - target) < 1e-6 else 0.0
                    else:
                        return 1.0 if value == target else 0.0
                except ValueError:
                    continue
            return 0.0
        return reward_function