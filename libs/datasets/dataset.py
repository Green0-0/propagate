import random as rand
from typing import Callable, Dict, List, Tuple

from libs import genome
from libs.genome import Genome
from libs.datasets.reward import RewardGenerator

class Dataset:
    batch_size: int

    # A list of data pairs: (input in ShareGPT format, answer reward function, format reward function)
    pairs_train: List[Tuple[List[Dict[str, str]], Callable[[str], float], Callable[[str], float]]]
    pairs_test: List[Tuple[List[Dict[str, str]], Callable[[str], float], Callable[[str], float]]]
    suffix: str
    
    i: int
    last_batch: List[Tuple[List[List[Dict[str, str]]], List[Callable[[str], float]], List[Callable[[str], float]]]]
    
    def __init__(
        self,
        batch_size: int,
        suffix: str,
        dataset_input_key: str,
        dataset_pairs: List[Dict],
        answer_reward: RewardGenerator,
        format_reward: RewardGenerator,
        force_reuse_batches: bool = False,
        reward_func_ratio: float = 0.1,
        passk: int = 1,
        passk_proportion: float = 0.1,
        passk_minimum: float = 0.9,
        postprocess_reward: Callable[[List[Genome]], None] = None                                                           
    ):
        self.batch_size = batch_size
        self.suffix = suffix
        self.i = 0
        self.force_reuse_batches = force_reuse_batches
        self.reward_func_ratio = reward_func_ratio

        """Generalized pass@k with reward shaping works as follows:
        For each genome, we create 'passk' outputs for each question.
        For each genome where the answer reward exceeds 'passk_minimum', we consider this as correct.
        If the proportion of correct genomes is at least 'passk_proportion', we consider the batch as passed.
        If the genome passes that question, then its score on each output is the maximum reward it received on that question. 
        Ie. if it received a partial reward of 0.95, 0.6, 0.6, 0.6, 0.1 across passk@5 outputs with a minimum passing score of 0.9 and a passing proportion of 0.1, then all 5 outputs receive a score of 0.95.
        Note: Pass@k is ONLY used for training pairs, not for testing pairs, which are always tested zero-shot."""
        self.passk = passk
        self.passk_proportion = passk_proportion
        self.passk_minimum = passk_minimum
        self.postprocess_reward = postprocess_reward

        self.pairs_train = []
        self.pairs_test = []
        for pair in dataset_pairs:
            self.pairs_train.append((pair[dataset_input_key], answer_reward.build_reward_function(pair), format_reward.build_reward_function(pair)))

    def _get_next_batch(self) -> List[Tuple[List[Dict[str, str]], Callable[[str], float], Callable[[str], float]]]:
        """
        Return the next `batch_size` entries from the dataset, wrapping around
        when the end of the list is reached.

        Output
        ------
        The subset of pairs in ShareGPT format.
        """
        n_pairs = len(self.pairs_train)
        if n_pairs == 0:
            return []

        start = self.i
        stop = start + self.batch_size

        if stop <= n_pairs:
            batch = self.pairs_train[start:stop]
        else:
            stop_mod = stop % n_pairs
            batch = self.pairs_train[start:] + self.pairs_train[:stop_mod]

        self.i = stop % n_pairs

        # Duplicate each entry 'passk' times for pass@k
        expanded_batch = []
        for entry in batch:
            for _ in range(self.passk):
                expanded_batch.append(entry)
        batch = expanded_batch
        return batch
    
    def next(self, population_size: int, mirror: bool) -> List[List[List[Dict[str, str]]]]:
        """
        Return a separate batch for each member of the population.

        Output
        ------
        A list of batches, each batch being a list of input texts in ShareGPT format.
        """
        self.last_batch = []
        all_inputs = []
        self.last_batch_is_train = True
        if self.force_reuse_batches:
            batch = self._get_next_batch()
            dict_lists, answer_funcs, reward_funcs = zip(*batch)
            all_inputs = [list(dict_lists) for _ in range(population_size)]
            self.last_batch = [(list(dict_lists), list(answer_funcs), list(reward_funcs)) for _ in range(population_size)]
        else:
            for _ in range(population_size):
                batch = self._get_next_batch()
                dict_lists, answer_funcs, reward_funcs = zip(*batch)
                self.last_batch.append((list(dict_lists), list(answer_funcs), list(reward_funcs)))
                all_inputs.append(list(dict_lists))
        if mirror:
            # Double the number of batches because the effective population size is doubled, ie. [Genome List | Mirror Genome List] so we need [Batch List | Batch List]
            all_inputs = all_inputs * 2
            self.last_batch = self.last_batch * 2
        return all_inputs

    def get_test_set(self) -> List[List[List[Dict[str, str]]]]:
        """
        Return the entire test set.
        """
        self.last_batch_is_train = False
        if not self.pairs_test or len(self.pairs_test) == 0:
            raise ValueError("Test set is empty or not defined. Generate test split first.")
        dict_lists, answer_funcs, format_funcs = zip(*self.pairs_test)
        self.last_batch = [(list(dict_lists), list(answer_funcs), list(format_funcs))]
        return [list(dict_lists)]
    
    def score_all(self, genomes: List[Genome]):
        """
        Compute the scores of all genomes based on the last batch. Update each genome's reward history, and the set of its latest rewards.
        """
        if not self.last_batch or len(self.last_batch) == 0:
            raise ValueError("No last batch available. Score should be done on a batch after outputs are generated.")
        all_rewards = []
        for i, genome in enumerate(genomes):
            if not genome.latest_outputs:
                raise ValueError("Genome does not have outputs for the last batch.")
            if len(genome.latest_outputs) != len(self.last_batch[i][0]):
                raise ValueError("Genome outputs do not match the last batch size.")
            _, answer_funcs, format_funcs = self.last_batch[i]
            latest_rewards = [
                (answer_func(output), format_func(output))
                for output, answer_func, format_func in zip(genome.latest_outputs, answer_funcs, format_funcs)
            ]
            if self.last_batch_is_train:
                # Compute pass@k adjusted rewards per question for the total batch
                # Count the number of passing outputs per question, and if that exceeds the passk_proportion, assign max reward to all outputs for that question
                # Apply the format function afterwards
                adjusted_rewards = []
                for j in range(0, len(latest_rewards), self.passk):
                    question_rewards = latest_rewards[j:j+self.passk]
                    n_passing = sum(1 for r in question_rewards if r[0] >= self.passk_minimum)
                    proportion_passing = n_passing / self.passk
                    adjusted_rewards.extend(question_rewards)
                    if proportion_passing >= self.passk_proportion:
                        max_reward = max(r[0] for r in question_rewards)
                        for k in range(self.passk):
                            adjusted_rewards[j + k] = (max_reward, question_rewards[k][1])
            else:
                adjusted_rewards = latest_rewards
            all_rewards.append(adjusted_rewards)
        if self.last_batch_is_train:
            for i, genome in enumerate(genomes):
                adjusted_rewards = all_rewards[i]
                genome.latest_rewards = [ans_reward for ans_reward, fmt_reward in adjusted_rewards]
            if self.postprocess_reward is not None:
                self.postprocess_reward.post_process_rewards(genomes)
            for i, genome in enumerate(genomes):
                adjusted_rewards = all_rewards[i]
                genome.latest_rewards = [
                        genome.latest_rewards[j] * (1 - self.reward_func_ratio) + fmt_reward * self.reward_func_ratio
                        for j, (_, fmt_reward) in enumerate(adjusted_rewards)
                    ]
        else:
            for i, genome in enumerate(genomes):
                adjusted_rewards = all_rewards[i]
                genome.latest_rewards = [
                    ans_reward
                    for ans_reward, fmt_reward in adjusted_rewards
                ]
        for genome in genomes:
            mean_reward = sum(genome.latest_rewards) / len(genome.latest_rewards)
            genome.historical_rewards.append(mean_reward)

    def generate_test_split(self, test_fraction: float, fold_index: int = 1):
        """Generate a test split from the dataset.

        Args:
            test_fraction (float): The fraction of the dataset to use for testing.
            fold_index (int): The index of the fold to use for testing. Use with cross validation. Defaults to 1.
        """
        n_pairs = len(self.pairs_train)
        n_test = int(n_pairs * test_fraction)
        start_idx = (fold_index - 1) * n_test
        end_idx = start_idx + n_test

        pairs_test = self.pairs_train[start_idx:end_idx]
        pairs_train = self.pairs_train[:start_idx] + self.pairs_train[end_idx:]

        self.pairs_train = pairs_train
        self.pairs_test = pairs_test

def _copy_dataset_config(source: Dataset) -> Dataset:
    new_dataset = Dataset.__new__(Dataset)
    new_dataset.batch_size = source.batch_size
    new_dataset.suffix = source.suffix
    new_dataset.force_reuse_batches = source.force_reuse_batches
    new_dataset.reward_func_ratio = source.reward_func_ratio
    new_dataset.passk = source.passk
    new_dataset.passk_proportion = source.passk_proportion
    new_dataset.passk_minimum = source.passk_minimum
    
    new_dataset.i = 0
    new_dataset.last_batch = []
    new_dataset.last_batch_is_train = True
    new_dataset.pairs_train = []
    new_dataset.pairs_test = []
    
    new_dataset.postprocess_reward = source.postprocess_reward
    return new_dataset

def merge_dataset(datasets: List[Dataset], shuffle: bool = True) -> Dataset:
    """
    Merges a list of datasets by concatenating them.
    Inherits configuration properties from the first dataset in the list.
    """
    if not datasets:
        raise ValueError("Dataset list cannot be empty.")

    new_dataset = _copy_dataset_config(datasets[0])
    
    new_dataset.pairs_train = []
    for d in datasets:
        new_dataset.pairs_train.extend(d.pairs_train)
    
    new_dataset.pairs_test = []
    for d in datasets:
        test_pairs = getattr(d, 'pairs_test', [])
        if test_pairs:
            new_dataset.pairs_test.extend(test_pairs)
    
    if shuffle:
        rand.shuffle(new_dataset.pairs_train)
        
    return new_dataset

def balanced_merge(datasets: List[Dataset]) -> Dataset:
    """
    Merges a list of datasets by interleaving them as evenly as possible based on their lengths.
    This prevents sequences of the same dataset type which can cause overfitting.
    
    Algorithm:
    Assigns each item a 'normalized position' in [0, 1] based on its index 
    and the total length of its source dataset, then sorts by this position.
    """
    if not datasets:
        raise ValueError("Dataset list cannot be empty.")

    new_dataset = _copy_dataset_config(datasets[0])
    
    weighted_items = []
    
    for d in datasets:
        length = len(d.pairs_train)
        if length == 0:
            continue
            
        for i, item in enumerate(d.pairs_train):
            pos = (i + 0.5) / length
            weighted_items.append((pos, item))
        
    weighted_items.sort(key=lambda x: x[0])
    
    new_dataset.pairs_train = [item for _, item in weighted_items]
    
    weighted_test = []
    has_test_data = False
    
    for d in datasets:
        test_pairs = getattr(d, 'pairs_test', [])
        length = len(test_pairs)
        if length == 0:
            continue
        
        has_test_data = True
        for i, item in enumerate(test_pairs):
            pos = (i + 0.5) / length
            weighted_test.append((pos, item))
    
    if has_test_data:
        weighted_test.sort(key=lambda x: x[0])
        new_dataset.pairs_test = [item for _, item in weighted_test]
        
    return new_dataset