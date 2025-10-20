from typing import Callable, Dict, List, Tuple

from libs.genome import Genome

import random

class Dataset:
    batch_size: int
    pairs: List[Tuple[List[Dict[str, str]], Callable[[str], float]]]
    suffix: str
    
    i: int
    last_batch: Tuple[List[List[Dict[str, str]]], List[Callable[[str], float]]]
    
    def __init__(
        self,
        batch_size: int,
        suffix: str,
        pairs: List[Tuple[List[Dict[str, str]], Callable[[str], float]]],
    ):
        """Initialize the Dataset.
        Warning: Scores should always be positive or zero, or logical errors will occur.

        Args:
            batch_size (int): The number of pairs to include in each batch.
            suffix (str): The suffix to append to each input text.
            pairs (List[Tuple[List[Dict[str, str]], Callable[[str], float]]]): The input-output pairs for the dataset, along with an associated scoring function.
        """
        self.batch_size = batch_size
        self.pairs = pairs
        self.suffix = suffix
        self.i = 0

    def next(
        self
    ) -> List[List[Dict[str, str]]]:
        """
        Return the next `batch_size` entries from the dataset, wrapping around
        when the end of the list is reached.

        Output
        ------
        A list of input texts in ShareGPT format.
        """
        n_pairs = len(self.pairs)
        if n_pairs == 0:
            return [], []

        start = self.i
        stop = start + self.batch_size

        if stop <= n_pairs:
            batch = self.pairs[start:stop]
        else:
            stop_mod = stop % n_pairs
            batch = self.pairs[start:] + self.pairs[:stop_mod]

        self.i = stop % n_pairs

        dict_lists, funcs = zip(*batch)
        self.last_batch = (list(dict_lists), list(funcs))
        return self.last_batch[0]

    def score(self, genome: Genome):
        """
        Compute the score of the given genome based on the last batch. Update the genome's reward history, and the set of its latest rewards.
        """
        if not self.last_batch:
            raise ValueError("No last batch available. Score should be done on a batch after outputs are generated.")
        if not genome.latest_outputs:
            raise ValueError("Genome does not have outputs for the last batch.")
        if len(genome.latest_outputs) != len(self.last_batch[0]):
            raise ValueError("Genome outputs do not match the last batch size.")
        _, funcs = self.last_batch
        
        genome.latest_rewards = [func(output) for func, output in zip(funcs, genome.latest_outputs)]
        mean_reward = sum(genome.latest_rewards) / len(genome.latest_rewards)
        genome.historical_rewards.append(mean_reward)

def combine_datasets(datasets: List[Dataset], shuffle: bool = False) -> Dataset:
    """Combine multiple datasets into a single dataset.

    Args:
        datasets (List[Dataset]): The datasets to combine.
        shuffle (bool, optional): Whether to shuffle the combined dataset. Defaults to False.

    Returns:
        Dataset: The combined dataset.
    """
    combined_pairs = []
    for dataset in datasets:
        combined_pairs.extend(dataset.pairs)
    if shuffle:
        random.shuffle(combined_pairs)
    return Dataset(
        batch_size=datasets[0].batch_size,
        suffix=datasets[0].suffix,
        pairs=combined_pairs
    )
    
#def combine_datasets_smart(datasets: List[Dataset], shuffle: bool = False) -> Dataset:    
#    pass