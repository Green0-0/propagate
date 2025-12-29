from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, parse, verify, LatexExtractionConfig
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union
import re

class RewardGenerator(ABC):
    """Abstract base class for reward generators.

    Reward classes in this framework serve as factories for reward functions.
    Because each data sample has a unique answer, a specific reward function must be built for each individual row of data.
    
    This enables:
    1. Combining diverse datasets (e.g., math, code, reasoning) into a single training run.
    2. Shuffling data without losing the verification logic for each item.
    """
    @abstractmethod
    def build_reward_function(self, input: Dict) -> Callable[[str], float]:
        """Builds a custom reward function for a specific data sample.

        Args:
            input (Dict): A single row of data from the dataset, containing the
                problem statement, ground truth answer, and other metadata.

        Returns:
            Callable[[str], float]: A function that takes a model's generated response (str)
            and returns a float reward. For consistency and interpretability, please keep the range of rewards between 0.0 and 1.0.
        """
        pass

class FormatRewardGenerator(RewardGenerator):
    """A reward generator that evaluates adherence to a specific output structure.

    This generator checks if the model's response follows a strict template, typically:
    `{start_think_token}...{end_think_token}...{start_answer_token}...{end_answer_token}`.
    
    It verifies:
    1. The presence of start/end tokens.
    2. The correct order of tokens.
    3. That tokens appear exactly once (if required).
    4. That no content trails the final answer token.
    
    Partial credit is awarded for meeting individual criteria.
    Not all the tokens may be required, only the ones specified by the user are.

    Attributes
    ----------
        start_think_token (str): The token marking the start of the thinking block.
        end_think_token (str): The token marking the end of the thinking block.
        start_answer_token (str): The token marking the start of the final answer.
        end_answer_token (str): The token marking the end of the final answer.
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
    """A reward generator that checks for exact string matches using regular expressions.

    This generator is useful for tasks where the answer is a specific string (e.g., a multiple-choice letter,
    a short phrase) that must appear within a specific context.

    It typically extracts a substring using `wrapper_regex` (e.g., content inside `<answer>...</answer>` tags)
    and compares it to the ground truth.

    Attributes
    ----------
        target_key (str): The key in the input dictionary that contains the ground truth string.
        wrapper_regex (str, optional): A regex pattern to extract the potential answer from the model's response.
            Defaults to extracting content within <answer> tags.
        lowercase (bool, optional): Whether to perform case-insensitive matching. Defaults to True.
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
    """A reward generator that extracts numerical answers from a reasoning trace.

    This generator scans the model's response for the *last* occurrence of a number (int or float)
    and compares it to the ground truth. This is a common heuristic for Chain-of-Thought (CoT)
    evaluations where the final answer is stated at the end.
    
    It supports:
    - Integer verification.
    - Float verification (within a small tolerance).

    Attributes
    ----------
        target_key (str): The key in the input dictionary containing the ground truth number.
        target_type (type): The expected type of the answer (int or float).
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
    """A robust reward generator for mathematical reasoning using the `math-verify` library.

    This generator is designed to handle complex mathematical expressions, LaTeX formatting,
    and equivalence checking (e.g., knowing that "1/2" is equal to "0.5").

    It performs a two-step process:
    1. Extraction: optionally isolates the answer using `wrapper_regex`.
    2. Verification: parses both the ground truth and the extracted prediction into symbolic
       representations to check for semantic equivalence.
    
    Attributes
    ----------
        target_answer_key (str): The key in the input dictionary containing the ground truth LaTeX or math string.
        extraction_config (Optional[List[LatexExtractionConfig]], optional): Configuration for `math-verify` to control
            how expressions are parsed and normalized. Defaults to a standard configuration if None.
        wrapper_regex (Optional[str], optional): Regex to limit the search space for the answer (e.g., inside tags).
            Defaults to r"<answer>(.*?)<\/answer>".
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
    """A reward generator specifically for multiple-choice questions (e.g., MMLU).

    This generator identifies which choice key (e.g., A, B, C, D) appears last in the model's
    extracted response and compares it to the correct choice.

    It handles:
    - Fixed choices (passed as a list).
    - Dynamic choices (retrieved from the input dictionary via a key).
    - Case-insensitive matching.

    Attributes
    ----------
        choices (Union[List[str], str]): Either a list of valid choice strings (e.g., ["A", "B", "C", "D"])
            or a key string to look up the choices in the input dictionary.
        target_answer_key (str): The key in the input dictionary containing the correct choice.
        lowercase (bool, optional): Whether to perform case-insensitive matching. Defaults to True.
        wrapper_regex (Optional[str], optional): Regex to limit the search space for the answer (e.g., inside tags).
            Defaults to r"<answer>(.*?)<\/answer>".
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