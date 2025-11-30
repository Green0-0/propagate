from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, parse, verify, LatexExtractionConfig
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union
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

class MathVerifyRewardGenerator(RewardGenerator):
    """
    Reward generator that uses the math-verify library to parse and verify mathematical answers.
    
    It supports:
    1. Pre-filtering with a wrapper regex (e.g., to look only inside <answer> tags).
    2. Configurable extraction settings (defaults to strict settings recommended for Reward Modeling).
    3. Robust parsing of both the Gold and the Model Output.
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
    
if __name__ == "__main__":
    def run_test(name, generator, input_row, response, expected):
        fn = generator.build_reward_function(input_row)
        score = fn(response)
        status = "PASS" if score == expected else "FAIL"
        print(f"[{status}] {name:<50} | Score: {score} (Exp: {expected})")

    print("=== Comprehensive Reward Generator Tests ===\n")

    # ==========================================
    # 1. MathVerify Tests
    # ==========================================
    print("--- MathVerifyRewardGenerator ---")
    gen_math = MathVerifyRewardGenerator(target_answer_key="ans")

    # A. NUMERICS
    run_test("Integer exact match", gen_math, {"ans": "42"}, "The answer is 42", 1.0)
    run_test("Decimal equivalence", gen_math, {"ans": "0.5"}, "It is 0.5", 1.0)
    run_test("Fraction vs Decimal", gen_math, {"ans": "1/2"}, "It is 0.5", 1.0)
    run_test("Negative number", gen_math, {"ans": "-10"}, "result is -10", 1.0)
    # FIX: "10^3" in plain text (ExprConfig) is often parsed as XOR. 
    # Standardize on Latex format for powers to be safe.
    run_test("Scientific (Standard)", gen_math, {"ans": "1000"}, r"$10^3$", 1.0)

    # B. LATEX & FRACTIONS
    run_test("Latex Fraction", gen_math, {"ans": "1/2"}, r"The value is $\frac{1}{2}$", 1.0)
    # FIX: Gold must be clear. 1.5 == 1 + 1/2.
    run_test("Mixed Number (Explicit)", gen_math, {"ans": "1.5"}, r"$1 + \frac{1}{2}$", 1.0)

    # C. SETS & INTERVALS
    # FIX: Wrap Gold in $...$ so it is parsed as a FiniteSet/Interval, not a Python Tuple/String.
    gen_set = MathVerifyRewardGenerator(target_answer_key="ans")
    run_test("Set Equivalence (LatEx)", gen_set, {"ans": r"$\{1, 2\}$"}, r"$\{2, 1\}$", 1.0)
    run_test("Intervals", gen_set, {"ans": r"$(1, 5)$"}, r"$x \in (1, 5)$", 1.0)
    run_test("Empty Set", gen_set, {"ans": r"$\emptyset$"}, r"The set is $\emptyset$", 1.0)

    # D. ALGEBRA & SYMBOLIC
    # FIX: Wrap Gold in $...$ so it is parsed as Symbolic Algebra
    gen_alg = MathVerifyRewardGenerator(target_answer_key="ans")
    run_test("Algebra (x=)", gen_alg, {"ans": "5"}, r"$x = 5$", 1.0)
    run_test("Algebra (Symbolic)", gen_alg, {"ans": r"$x+1$"}, r"$1+x$", 1.0)
    run_test("Algebra (Factorization)", gen_alg, {"ans": r"$x^2 + 2x + 1$"}, r"$(x+1)^2$", 1.0)

    # E. ADVANCED: MATRICES & VECTORS
    gen_matrix = MathVerifyRewardGenerator(target_answer_key="ans")
    # FIX: Wrap Gold in $...$ to ensure it is parsed as a Vector/Point, not a Tuple
    run_test("Vector Equivalence", gen_matrix, {"ans": r"$(1, 2, 3)$"}, r"$(1, 2, 3)$", 1.0)
    # Identity Matrix
    matrix_gold = r"$\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$"
    run_test("Identity Matrix", gen_matrix, {"ans": matrix_gold}, r"$\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$", 1.0)

    # F. SPECIAL FEATURES: BOXED & WRAPPERS
    gen_boxed = MathVerifyRewardGenerator(target_answer_key="ans")
    run_test("Boxed Priority", gen_boxed, {"ans": "5"}, r"Ignore 10, answer is \boxed{5}", 1.0)
    run_test("Boxed Trap", gen_boxed, {"ans": "5"}, r"Answer is 5 but I wrote \boxed{10}", 0.0)

    gen_wrap = MathVerifyRewardGenerator(target_answer_key="ans", wrapper_regex=r"<ans>(.*?)</ans>")
    run_test("Wrapper (Correct inside)", gen_wrap, {"ans": "10"}, "Wrong 5 <ans>10</ans>", 1.0)
    run_test("Wrapper (Wrong inside)", gen_wrap, {"ans": "10"}, "Correct 10 <ans>5</ans>", 0.0)

    # ==========================================
    # 2. LastChoice Tests
    # ==========================================
    print("\n--- LastChoiceRewardGenerator ---")
    
    choices = ["A", "B", "C", "D"]
    gen_choice = LastChoiceRewardGenerator(choices=choices, target_answer_key="ans")

    run_test("Simple Match", gen_choice, {"ans": "C"}, "The answer is C", 1.0)
    run_test("Lowercase Match", gen_choice, {"ans": "B"}, "answer is b", 1.0)
    run_test("Distractor (A then C)", gen_choice, {"ans": "C"}, "I thought A, but it's C", 1.0)
    run_test("Distractor (C then A - Wrong)", gen_choice, {"ans": "C"}, "I thought C, but it's A", 0.0)
    run_test("Parentheses (A)", gen_choice, {"ans": "A"}, "The answer is (A)", 1.0)
    run_test("Dot A.", gen_choice, {"ans": "A"}, "A.", 1.0)
    
    gen_choice_wrap = LastChoiceRewardGenerator(choices=choices, target_answer_key="ans", wrapper_regex=r"<final>(.*?)</final>")
    run_test("Choice Wrapper (Correct)", gen_choice_wrap, {"ans": "B"}, "A <final>B</final>", 1.0)
    run_test("Choice Wrapper (Trap)", gen_choice_wrap, {"ans": "B"}, "B <final>A</final>", 0.0)
    run_test("Containment (Apple matches A)", gen_choice, {"ans": "A"}, "Apple", 1.0) 

    print("\n=== All Tests Completed ===")