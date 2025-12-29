from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union
import re
import math

from propagate.genome import Genome

class PostProcessReward(ABC):
    @abstractmethod
    def post_process_rewards(self, genomes: List[Genome]):
        """Applies post processing rewards to the given batch of genomes. Assumes the genomes store their answer rewards, but not the format rewards."""
        pass

class DynamicLengthReward(PostProcessReward):
    """
    A post processing reward that penalizese long correct answers, while rewarding long incorrect answers. This is based on the idea that if the model outputs a long thinking trace while being correct, it is overthinking, but if the model outputs a short thinking trace while being incorrect, it is overconfident. 
    """
    def __init__(self, length_penalty_percent: float = 0.3, length_reward_percent: float = 0.3, words_target: int = 2000, correct_threshold: float = 0.9):
        """
        Args:
            length_penalty: The penalty factor applied per token in the response.
        """
        self.length_penalty_percent = length_penalty_percent
        self.length_reward_percent = length_reward_percent
        self.words_target = words_target
        self.correct_threshold = correct_threshold

    def post_process_rewards(self, genomes: List[Genome]):
        # Assume that genomes.latest_outputs is populated and genomes.latest_rewards contains only answer rewards corresponding to the respective output
        # If the length of the output is greater than words_target, assume that it is equal to words_target for penalty purposes

        for genome in genomes:
            for i in range(len(genome.latest_outputs)):
                output = genome.latest_outputs[i]
                answer_reward = genome.latest_rewards[i]

                output_len = min(len(output.split()), self.words_target)
                output_usage_proportion = output_len / self.words_target

                if answer_reward > self.correct_threshold:
                    # Correct answer, apply length penalty
                    length_penalty = output_usage_proportion * self.length_penalty_percent
                    genome.latest_rewards[i] = answer_reward * (1 - length_penalty)
                else:
                    # Incorrect answer, apply length reward (weighted average between answer reward and length reward)
                    length_reward = output_usage_proportion * self.length_reward_percent
                    genome.latest_rewards[i] = answer_reward + (1 - answer_reward) * length_reward

class NormalizedLengthReward(PostProcessReward):
    def __init__(self, length_penalty_percent: float = 0.3, length_reward_percent: float = 0.3, correct_threshold: float = 0.9):
        self.length_penalty_percent = length_penalty_percent
        self.length_reward_percent = length_reward_percent
        self.correct_threshold = correct_threshold

    def post_process_rewards(self, genomes: List[Genome]):
        all_lengths = []
        for genome in genomes:
            for output in genome.latest_outputs:
                all_lengths.append(len(output.split()))

        mean_length = sum(all_lengths) / len(all_lengths)
        variance = sum((x - mean_length) ** 2 for x in all_lengths) / (len(all_lengths) - 1)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return

        for genome in genomes:
            for i in range(len(genome.latest_outputs)):
                output = genome.latest_outputs[i]
                answer_reward = genome.latest_rewards[i]
                current_len = len(output.split())

                z_score = (current_len - mean_length) / std_dev
                relative_length = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

                if answer_reward > self.correct_threshold:
                    penalty_factor = relative_length * self.length_penalty_percent
                    genome.latest_rewards[i] = answer_reward * (1 - penalty_factor)
                
                else:
                    reward_factor = relative_length * self.length_reward_percent
                    genome.latest_rewards[i] = answer_reward + (1 - answer_reward) * reward_factor