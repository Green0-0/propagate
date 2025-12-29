from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, parse, verify, LatexExtractionConfig
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union
import re

class RewardGenerator(ABC):
    """To seamlessly support combining datasets, reward classes serve as builders for reward functions. Each class builds an individual reward function based on a single row of data, which can be shuffled together and stacked to form a varied dataset."""
    @abstractmethod
    def build_reward_function(self, input: Dict) -> Callable[[str], float]:
        """Builds a reward function for the given row of data."""
        pass

class FormatRewardGenerator(RewardGenerator):
    """Reward function for formatting rewards.
    Checks if the response follows the following format: {start_think_token}...{end_think_token}...{start_answer_token}...{end_answer_token}.
    With only exactly one of each token, and nothing after the {end_answer_token}.
    If any of the tokens are passed as none in init, ignore them when checking.
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
    """Basic equality reward function with regex.
    Checks if the response contains the target string within the last instance of the wrapper regex.
    For example, <answer>hello</answer> would match the string "hello" within an <answer>(.*?)</answer> regex.
    """
    def __init__(self, target_key: str, wrapper_regex: str = r"<answer>(.*?)<\/answer>", lowercase: bool = True):
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
    Reward function that extracts the last value which fits the given datatype and checks if it is equal to the target. Useful if you want the last number which usually corresponds to the answer in a COT reasoning trace.
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

class MathVerifyRewardGenerator(RewardGenerator):
    """
    Reward generator that uses the math-verify library to parse and verify mathematical answers.
    
    It supports pre-filtering with a wrapper regex (e.g., to look only inside <answer> tags) and specifying the extraction settings. See the math-verify documentation for more details.
    """
    def __init__(
        self, 
        target_answer_key: str, 
        extraction_config: Optional[List[LatexExtractionConfig]] = None, 
        wrapper_regex: Optional[str] = r"<answer>(.*?)<\/answer>"
    ):
        self.target_key = target_answer_key
        self.wrapper_regex = wrapper_regex
    
        if extraction_config is None:
            self.extraction_config = [
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        basic_latex=True,
                        units=True,
                        malformed_operators=False,
                        nits=False,
                        boxed="all",
                        equations=False
                    ),
                    boxed_match_priority=0 
                ),
                ExprExtractionConfig()
            ]
        else:
            self.extraction_config = extraction_config

    def build_reward_function(self, input: Dict) -> Callable[[str], float]:
        gold_raw = str(input.get(self.target_key, ""))
        gold_raw = f"${gold_raw}$" if not (gold_raw.startswith("$") and gold_raw.endswith("$")) else gold_raw
        try:
            gold_parsed = parse(gold_raw, extraction_config=self.extraction_config)
        except Exception:
            print(f"Warning: Failed to parse gold answer: {gold_raw}. Assigning zero reward.")
            return lambda x: 0.0

        def reward_function(response: str) -> float:
            text_to_parse = response

            if self.wrapper_regex:
                try:
                    matches = re.findall(self.wrapper_regex, response, re.DOTALL)
                    if not matches:
                        return 0.0
                    
                    last_match = matches[-1]
                    if isinstance(last_match, tuple):
                        text_to_parse = " ".join(last_match)
                    else:
                        text_to_parse = last_match
                except re.error:
                    return 0.0
            try:
                pred_parsed = parse(text_to_parse, extraction_config=self.extraction_config)
            except Exception:
                return 0.0
            try:
                is_correct = verify(gold_parsed, pred_parsed)
                return 1.0 if is_correct else 0.0
            except Exception:
                return 0.0

        return reward_function

class LastChoiceRewardGenerator(RewardGenerator):
    """
    Reward function that checks if the last answer correlation to a possible choice in the response matches the target answer. Meant for multiple choice datasets like MMLU.
    """
    def __init__(
        self, 
        choices: Union[List[str], str], 
        target_answer_key: str, 
        lowercase: bool = True, 
        wrapper_regex: Optional[str] = r"<answer>(.*?)<\/answer>"
    ):
        self.choices = choices
        self.target_answer_key = target_answer_key
        self.lowercase = lowercase
        self.wrapper_regex = wrapper_regex

    def build_reward_function(self, input: Dict) -> Callable[[str], float]:
        if isinstance(self.choices, str):
            valid_choices = input.get(self.choices, [])
        else:
            valid_choices = self.choices

        target = str(input.get(self.target_answer_key, "")).strip()

        if self.lowercase:
            valid_choices = [str(c).lower() for c in valid_choices]
            target = target.lower()
        else:
            valid_choices = [str(c) for c in valid_choices]

        if not valid_choices:
            print(f"Warning: No valid choices found for LastChoiceRewardGenerator. Assigning zero reward.")
            return lambda x: 0.0
            
        choices_pattern = "|".join(map(re.escape, sorted(valid_choices, key=len, reverse=True)))
        
        def reward_function(response: str) -> float:
            text_to_scan = response

            if self.wrapper_regex:
                try:
                    matches = re.findall(self.wrapper_regex, response, re.DOTALL)
                    if not matches:
                        return 0.0
                    
                    last_match = matches[-1]
                    if isinstance(last_match, tuple):
                        text_to_scan = " ".join(last_match)
                    else:
                        text_to_scan = last_match
                except re.error:
                    print(f"Regex Error in LastChoiceRewardGenerator: {self.wrapper_regex}")
                    return 0.0

            if self.lowercase:
                text_to_scan = text_to_scan.lower()

            found_choices = re.findall(choices_pattern, text_to_scan)

            if not found_choices:
                return 0.0

            last_choice = found_choices[-1]

            return 1.0 if last_choice == target else 0.0

        return reward_function