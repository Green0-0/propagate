import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from libs.dataset import Dataset

def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]
    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"
    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)
    if full_format_match:
        return 1.0
    reward = 0.0
    if think_match:
        reward += 0.1
    if answer_match:
        reward += 0.5
    return reward


def answer_reward_function(
    response: str, numbers: List[int] = None, target: int = None
) -> float:
    """
    Checks if the last <answer>...</answer> uses all numbers exactly once and evaluates to the target.
    Returns 1.0 if the last one is correct, else 0.0.
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    all_matches = re.findall(answer_regex, response, re.DOTALL)
    if not all_matches:
        return 0.0
    # Only check the last answer
    answer_content = all_matches[-1]
   
    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not answer_content:
        return 0.0
    if not re.match(allowed_chars, answer_content):
        return 0.0
    # Check numbers used
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0
    # Try evaluating
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        return 0.0
    return 0.0


def reward_function(
    response: str,
    numbers: List[int] = None,
    target: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Reward function for Countdown Tasks.
    Total reward = 0.1 * format_reward + answer_reward
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }


def load_countdown_dataset() -> Dataset:
    """
    Load the countdown dataset and convert it to the Dataset format.
    
    Returns
    -------
    Tuple containing:
        - batch_size: int (size of the full dataset)
        - suffix: str (the suffix to use, "<think>")
        - pairs: List[Tuple[List[Dict[str, str]], Callable[[str], float]]]
    """
    json_path = "libs/dataset_files/countdown.json"
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    batch_size = len(data)
    suffix = "<think>"
    pairs = []
    
    for item in data:
        # Clean the context by removing the trailing part
        context = item["context"]
        # Remove "Let me solve this step by step.\n<think>" from the end
        if context.endswith("Let me solve this step by step.\n<think>"):
            context = context[:-len("Let me solve this step by step.\n<think>")]
        
        # Create ShareGPT format
        sharegpt_format = [
            {
                "role": "user",
                "content": context
            }
        ]
        
        # Create reward function that uses the countdown reward logic
        def create_reward_function(numbers: List[int], target: str) -> Callable[[str], float]:
            def reward_fn(response: str) -> float:
                result = reward_function(
                    response=response,
                    numbers=numbers,
                    target=int(target),
                    end_token=None
                )
                return result["reward"]
            return reward_fn
        
        reward_fn = create_reward_function(item["numbers"], item["target"])
        
        pairs.append((sharegpt_format, reward_fn))

    return Dataset(batch_size, suffix, pairs)


if __name__ == "__main__":
    # Example usage
    dataset = load_countdown_dataset()

    print(f"Batch size: {dataset.batch_size}")
    print(f"Suffix: {dataset.suffix}")
    print(f"Number of pairs: {len(dataset.pairs)}")

    # Get first batch
    sharegpt_lists, reward_fns = dataset.next()
    print(f"\nFirst item in batch:")
    print(sharegpt_lists[0])
    
    # Test the reward function with different responses
    print(f"\nReward function tests:")
    
    # Test 1: Correct format and correct answer
    test_response_1 = "I need to find a way to combine 44, 19, and 35 to get 98.\n</think>\n<answer>(44 + 19) + 35</answer>"
    print(f"Correct format & answer: {reward_fns[0](test_response_1)}")

    # Test 2: Correct format and different correct answer
    test_response_1 = "I need to find a way to combine 44, 19, and 35 to get 98.\n</think>\n<answer>44 + 19 + 35</answer>"
    print(f"Correct format & answer: {reward_fns[0](test_response_1)}")
    
    # Test 3: Correct format but wrong answer
    test_response_2 = "Let me try something.\n</think>\n<answer>(44 + 19) + 36</answer>"
    print(f"Correct format, wrong answer: {reward_fns[0](test_response_2)}")
    
    # Test 4: Only has answer tag (partial format)
    test_response_3 = "<answer>(44 + 19) + 35</answer>"
    print(f"Only answer tag, correct answer: {reward_fns[0](test_response_3)}")
    
    # Test 5: No tags
    test_response_4 = "(44 + 19) + 35"
    print(f"No tags: {reward_fns[0](test_response_4)}")

    print(f"\nSecond item in batch:")
    print(sharegpt_lists[1])

    # Test the reward function for the second item
    print(f"\nReward function tests for second item:")

    # Test 1: Previous format and previous answer
    test_response_1 = "I need to find a way to combine 44, 19, and 35 to get 98.\n</think>\n<answer>(44 + 19) + 35</answer>"
    print(f"Previous incorrect format & answer: {reward_fns[1](test_response_1)}")

    # Test 2: Correct format and correct answer
    test_response_2 = "Let me try something.\n</think>\n<answer>(63 - 95) + 96</answer>"
    print(f"Correct format & answer: {reward_fns[1](test_response_2)}")