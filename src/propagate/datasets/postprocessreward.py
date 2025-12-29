from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union
import re
import math

from propagate.genome import Genome

class PostProcessReward(ABC):
    """Abstract base class for post-processing rewards.

    This class defines the interface for modifying rewards after they have been initially calculated.
    This allows for adjustments based on factors like output length, formatting, or other heuristics
    that are not captured by the primary reward function.
    """
    @abstractmethod
    def post_process_rewards(self, genomes: List[Genome]):
        """Applies post-processing adjustments to the rewards of the given batch of genomes.

        This method should modify the 'latest_rewards' attribute of the genomes in place.
        It assumes that the genomes already contain the primary answer rewards.

        Args:
            genomes (List[Genome]): A list of Genome objects whose rewards should be post-processed.
        """
        pass

class DynamicLengthReward(PostProcessReward):
    """A post-processing reward that penalizes long correct answers and rewards long incorrect answers.

    This strategy is based on the heuristic that if a model generates a long thinking trace and arrives at the correct answer, it may be "overthinking" or inefficient. 
    Conversely, if it generates a short thinking trace and is incorrect, it may be "overconfident".
    Therefore, this class lightly penalizes verbose correct answers to encourage conciseness and rewards verbose incorrect answers to encourage deeper thinking steps when the model is struggling.

    Warning: words_target is in words, not in tokens, and only serves a proxy for the length of the thinking trace.

    Attributes
    ----------
        length_penalty_percent (float): The maximum percentage to penalize a correct answer based on length.
        length_reward_percent (float): The maximum percentage to boost an incorrect answer based on length.
        words_target (int): The target word count. Lengths beyond this are capped for calculation purposes.
        correct_threshold (float): The reward threshold above which an answer is considered correct.
    """
    def __init__(self, length_penalty_percent: float = 0.3, length_reward_percent: float = 0.3, words_target: int = 2000, correct_threshold: float = 0.9):
        self.length_penalty_percent = length_penalty_percent
        self.length_reward_percent = length_reward_percent
        self.words_target = words_target
        self.correct_threshold = correct_threshold

    def post_process_rewards(self, genomes: List[Genome]):
        """Applies dynamic length-based reward adjustments to a list of genomes.

        For each genome and each of its latest outputs:
        - If the answer is correct (reward > threshold): Apply a penalty proportional to the length.
        - If the answer is incorrect: Apply a reward boost proportional to the length.

        The length is normalized against `words_target`. If the length of the output is greater than `words_target`, it is capped at `words_target`.

        Args:
            genomes (List[Genome]): The list of genomes to process.
        """
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
    """A post-processing reward that adjusts rewards based on the statistical distribution of output lengths.

    This class calculates the mean and standard deviation of output lengths across the entire batch of genomes. It then calculates a Z-score for each output length to determine its relative length. This means you do not need to specify an arbitrary target length.
    
    Similar to DynamicLengthReward:
    - Long correct answers are penalized (encouraging efficiency).
    - Long incorrect answers are rewarded (encouraging detailed reasoning).
    
    The magnitude of adjustment is based on the cumulative distribution function (CDF) of the Z-score.

    Attributes
    ----------
        length_penalty_percent (float): The maximum percentage to penalize correct answers.
        length_reward_percent (float): The maximum percentage to boost incorrect answers.
        correct_threshold (float): The reward threshold above which an answer is considered correct.
    """
    def __init__(self, length_penalty_percent: float = 0.3, length_reward_percent: float = 0.3, correct_threshold: float = 0.9):
        """Initialize the NormalizedLengthReward.

        Args:
            length_penalty_percent (float, optional): Maximum penalty for correct answers. Defaults to 0.3.
            length_reward_percent (float, optional): Maximum reward boost for incorrect answers. Defaults to 0.3.
            correct_threshold (float, optional): Threshold for considering an answer correct. Defaults to 0.9.
        """
        self.length_penalty_percent = length_penalty_percent
        self.length_reward_percent = length_reward_percent
        self.correct_threshold = correct_threshold

    def post_process_rewards(self, genomes: List[Genome]):
        """Applies statistics-based length reward adjustments to a list of genomes.

        Calculates batch-wide statistics (mean, variance, std_dev) for output lengths.
        Then adjusts rewards based on how far each output's length deviates from the mean.

        Args:
            genomes (List[Genome]): The list of genomes to process.
        """
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