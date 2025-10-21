import re


def format_reward(response: str, reward_amount = 0.1, start_think_token: str = "", end_think_token: str = "", start_answer_token: str = "", end_answer_token: str = "") -> float:
    """
    Checks if the response follows the following format: {start_think_token}...{end_think_token}...{start_answer_token}...{end_answer_token}.
    With only exactly one of each token, and nothing after the {end_answer_token}.
    If any of the tokens are none, ignore them in the checking.
    Returns the reward, which increments by ``reward_amount`` for each correct section.
    """
    response = response.strip()
    
    format_rewards = 0.0
    if start_think_token != None and start_think_token != "" and response.startswith(start_think_token) and response.count(start_think_token) == 1:
        format_rewards += reward_amount
        
    if end_think_token != None and end_think_token != "" and response.count(end_think_token) == 1:
        format_rewards += reward_amount
        
    if start_answer_token != None and start_answer_token != "" and response.count(start_answer_token) == 1:
        if end_think_token != None and end_think_token != "":
            if response.find(start_answer_token) > response.find(end_think_token):
                format_rewards += reward_amount
        else:
            if start_think_token != None and start_think_token != "":
                if response.find(start_answer_token) > response.find(start_think_token):
                    format_rewards += reward_amount
            else:
                format_rewards += reward_amount

    if end_answer_token != None and end_answer_token != "" and response.endswith(end_answer_token) and response.count(end_answer_token) == 1:
        format_rewards += reward_amount
    return format_rewards

def basic_validator_reward(response: str, target: str, wrapper_regex: str, lowercase: bool = True) -> float:
    """
    Basic validator reward function.
    Checks if the response contains the target string within the last instance of the wrapper regex.
    Returns 1.0 if found, else 0.0.
    """
    all_matches = re.findall(wrapper_regex, response, re.DOTALL)
    if not all_matches:
        return 0.0
    
    last_instance = all_matches[-1].strip()
    if lowercase:
        if target.strip().lower() == last_instance.lower():
            return 1.0
    if target.strip() == last_instance:
        return 1.0
    return 0.0