from typing import Any, Callable, Dict, List, Optional
from libs.datasets.dataset import Dataset
from libs.datasets.reward import FormatRewardGenerator, RewardGenerator

def load_hf_dataset(
    hf_data,
    answer_reward: RewardGenerator,
    input_column: str,
    target_column: str,
    format_reward: Optional[RewardGenerator] = None,
    prompt_template: str = "{{prompt}}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>",
    suffix: str = "<think>",
    batch_size: int = 50,
    reward_func_ratio: float = 0.1,
    passk: int = 1,
    passk_proportion: float = 0.1,
    passk_minimum: float = 0.9,
    force_reuse_batches: bool = False
) -> Dataset:
    """
    A generalized loader for Hugging Face datasets that prepares data for RL training.
    
    Args:
        hf_data: The pre-loaded Hugging Face dataset object.
        answer_reward: The specific reward generator for checking answer correctness.
        input_column: The column name in the HF dataset containing the user prompt/question.
        target_column: The column name containing the ground truth answer (used by reward gen).
        format_reward: Custom format reward generator. Defaults to the standard <think>/<answer> format.
        prompt_template: String containing '{{prompt}}' where the input text should be injected.
        suffix: The string suffix to append to inputs (default: "<think>").
        batch_size, reward_func_ratio, etc.: Standard arguments passed to the Dataset class.
    """
    
    print(f"Building HF dataset from {hf_data}...")
    pairs = []
    
    for item in hf_data:
        user_input = item.get(input_column)
        if user_input is None:
            continue
            
        instruction = prompt_template.replace("{{prompt}}", str(user_input))

        sharegpt_format = [
            {
                "role": "user",
                "content": instruction
            }
        ]
        
        pair = {
            "input": sharegpt_format
        }
        if target_column in item:
            pair[target_column] = item[target_column]
            
        pairs.append(pair)

    if format_reward is None:
        format_reward = FormatRewardGenerator(
            start_think_token="", 
            end_think_token="</think>", 
            start_answer_token="<answer>", 
            end_answer_token="</answer>"
        )

    return Dataset(
        batch_size=batch_size,
        suffix=suffix,
        dataset_input_key="input",
        dataset_pairs=pairs,
        answer_reward=answer_reward,
        format_reward=format_reward,
        force_reuse_batches=force_reuse_batches,
        reward_func_ratio=reward_func_ratio,
        passk=passk,
        passk_proportion=passk_proportion,
        passk_minimum=passk_minimum
    )