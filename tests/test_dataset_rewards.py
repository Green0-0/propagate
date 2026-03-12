import pytest
from propagate.datasets.reward import (
    RewardGenerator,
    FormatRewardGenerator, 
    RegexRewardGenerator, 
    LastMatchRewardGenerator, 
    MathVerifyRewardGenerator
)
import re

# Helper for testing format reward
def test_format_reward_structure():
    gen = FormatRewardGenerator(
        start_think_token="<think>", 
        end_think_token="</think>", 
        start_answer_token="<answer>", 
        end_answer_token="</answer>"
    )
    
    # Perfect format
    reward_fn = gen.build_reward_function({})
    score_perfect = reward_fn("<think>foo</think><answer>bar</answer>")
    assert score_perfect == 1.0

    # Missing parts
    score_partial = reward_fn("<think>foo</think>")
    # Has start/end think (2 points), missing start/end answer. But check order logic too...
    # total possible: 4
    # Points: start T (1) + end T (1) + start A (0) + end A (0) = 2/4 = 0.5
    assert score_partial == 0.5

    # Wrong order
    score_wrong_order = reward_fn("<answer>bar</answer><think>foo</think>")
    # Start T (0 - not at start?), End T (1), Start A (0 - before end T?), End A (1)
    # The current implementation checks startswith for start_think.
    # So: Start T=0. End T=1. Start A: checks against End T index. 
    # "<answer>" index < "</think>" index? 0 < 15. So Start A logic fails if it expects specific order.
    # The code says if end_think exists: find(start_answer) > find(end_think)
    # Here: 0 < 15 -> False. So Start A point is 0.
    # End A point is 1.
    # Total: 2/4 = 0.5
    # Score 4:
    # <answer>bar</answer><think>foo</think>
    # Start T: False (0).
    # End T: True (1).
    # Start A: False (0).
    # End A: False (0) - strict endswith check.
    # Total: 1/4 = 0.25.
    
    assert score_wrong_order == 0.25

def test_regex_reward():
    gen = RegexRewardGenerator(target_key="tgt")
    reward_fn = gen.build_reward_function({"tgt": "Correct"})
    
    # Exact match within tags
    assert reward_fn("...<answer>Correct</answer>...") == 1.0
    # Case insensitive by default
    assert reward_fn("<answer>correct</answer>") == 1.0
    # Wrappers missing
    assert reward_fn("Correct") == 0.0
    # Wrong content
    assert reward_fn("<answer>Wrong</answer>") == 0.0

def test_last_match_reward_int():
    gen = LastMatchRewardGenerator(target_key="tgt", target_type=int)
    reward_fn = gen.build_reward_function({"tgt": "42"})
    
    # Find last number
    assert reward_fn("The answer is 42") == 1.0
    assert reward_fn("10, 20, 30... 42") == 1.0
    assert reward_fn("42 is the answer, not 43") == 0.0 # 43 is last

def test_last_match_reward_float():
    gen = LastMatchRewardGenerator(target_key="tgt", target_type=float)
    reward_fn = gen.build_reward_function({"tgt": "3.14"})
    
    # Tolerance check
    assert reward_fn("Pi is about 3.14") == 1.0
    assert reward_fn("3.14159") == 0.0 # Exact match required on the string "3.14"? No, within 1e-6
    # 3.14 vs 3.14159 -> diff > 1e-6 -> 0.0
    
    assert reward_fn("Value: 3.140000001") == 1.0

def test_math_verify_reward():
    # Placeholder: MathVerify relies on external libraries.
    # We test initialization and basic string matching logic if possible.
    try:
        gen = MathVerifyRewardGenerator(target_answer_key="tgt")
        # If imports fail inside, skip.
    except ImportError:
        pytest.skip("Math verify dependencies not installed")
        
    # Mock input
    input_data = {"tgt": "x^2"}
    # The class usually parses LaTeX.
    # We'll just verify it doesn't crash on build.
    fn = gen.build_reward_function(input_data)
    assert callable(fn)
