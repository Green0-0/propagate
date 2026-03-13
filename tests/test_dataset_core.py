import pytest
from propagate.genome import Genome
from propagate.datasets.dataset import Dataset, merge_dataset, balanced_merge
from propagate.datasets.reward import RewardGenerator

# Mock RewardGenerators for testing
class MockAnswerReward(RewardGenerator):
    def build_reward_function(self, input_data):
        target = input_data.get("target", "")
        def reward_func(response):
            return 1.0 if response.strip() == target else 0.0
        return reward_func

class MockFormatReward(RewardGenerator):
    def build_reward_function(self, input_data):
        def reward_func(response):
            return 1.0 if response.startswith("FORMAT:") else 0.0
        return reward_func

@pytest.fixture
def basic_dataset():
    pairs = [
        {"input": [{"role": "user", "content": "q1"}], "target": "a1"},
        {"input": [{"role": "user", "content": "q2"}], "target": "a2"},
        {"input": [{"role": "user", "content": "q3"}], "target": "a3"},
    ]
    ds = Dataset(
        batch_size=2,
        suffix="",
        dataset_input_key="input",
        dataset_pairs=pairs,
        answer_reward=MockAnswerReward(),
        format_reward=MockFormatReward(),
        passk=1
    )
    return ds

@pytest.fixture
def dataset_for_merge_1():
    pairs = [
        {"input": [{"role": "user", "content": "1-q1"}], "target": "1-a1"},
        {"input": [{"role": "user", "content": "1-q2"}], "target": "1-a2"},
    ]
    return Dataset(
        batch_size=2,
        suffix="",
        dataset_input_key="input",
        dataset_pairs=pairs,
        answer_reward=MockAnswerReward(),
        format_reward=MockFormatReward()
    )

@pytest.fixture
def dataset_for_merge_2():
    pairs = [
        {"input": [{"role": "user", "content": "2-q1"}], "target": "2-a1"},
    ]
    return Dataset(
        batch_size=2,
        suffix="",
        dataset_input_key="input",
        dataset_pairs=pairs,
        answer_reward=MockAnswerReward(),
        format_reward=MockFormatReward()
    )

def test_dataset_initialization(basic_dataset):
    assert basic_dataset.batch_size == 2
    assert len(basic_dataset.pairs_train) == 3
    assert basic_dataset.i == 0

def test_get_next_batch(basic_dataset):
    # First batch: items 0 and 1
    batch1 = basic_dataset._get_next_batch()
    assert len(batch1) == 2
    # Verify content
    assert batch1[0][0] == [{"role": "user", "content": "q1"}]
    assert batch1[1][0] == [{"role": "user", "content": "q2"}]
    
    # Second batch: item 2 and wrap around to item 0
    batch2 = basic_dataset._get_next_batch()
    assert len(batch2) == 2
    assert batch2[0][0] == [{"role": "user", "content": "q3"}]
    assert batch2[1][0] == [{"role": "user", "content": "q1"}]

def test_next_population_no_reuse(basic_dataset):
    pop_size = 2
    # Without force_reuse_batches, each genome gets a slice
    # Total available items = 3. 
    # Genome 0 gets [0, 1]
    # Genome 1 gets [2, 0] (wrap around)
    inputs = basic_dataset.next(population_size=pop_size, mirror=False, center=False)
    
    assert len(inputs) == pop_size
    assert len(inputs[0]) == 2
    assert len(inputs[1]) == 2
    
    assert inputs[0][0] == [{"role": "user", "content": "q1"}]
    assert inputs[1][0] == [{"role": "user", "content": "q3"}]
    
    # Verify internal state tracking
    assert len(basic_dataset.last_batch) == pop_size

def test_next_population_force_reuse(basic_dataset):
    basic_dataset.force_reuse_batches = True
    pop_size = 3
    
    # All genomes should get the SAME batch
    inputs = basic_dataset.next(population_size=pop_size, mirror=False, center=False)
    
    assert len(inputs) == pop_size
    first_batch_content = inputs[0]
    
    for i in range(1, pop_size):
        assert inputs[i] == first_batch_content

def test_next_population_mirror(basic_dataset):
    pop_size = 2
    inputs = basic_dataset.next(population_size=pop_size, mirror=True, center=False)
    
    # Mirror doubles the output list
    assert len(inputs) == pop_size * 2
    
    # The first half should match the second half
    half = pop_size
    assert inputs[:half] == inputs[half:]

def test_generate_test_split(basic_dataset):
    # 3 items total. Split 0.33 -> 1 item for test
    basic_dataset.generate_test_split(test_fraction=0.34, fold_index=1)
    
    assert len(basic_dataset.pairs_test) == 1
    assert len(basic_dataset.pairs_train) == 2
    
    # Verify test set retrieval
    test_set = basic_dataset.get_test_set()
    assert len(test_set) == 1 # One batch of test data
    assert len(test_set[0]) == 1 # Contains 1 item

def test_score_all(basic_dataset):
    # Setup a single genome interaction
    genome = Genome()
    inputs = basic_dataset.next(population_size=1, mirror=False, center=False)
    # Batch is [q1, q2] -> targets are a1, a2
    
    # Set outputs manually
    genome.latest_outputs = ["a1", "wrong"]
    
    # Calculate scores
    basic_dataset.score_all([genome])
    
    # Rewards are weighted sum of answer and format
    # reward_func_ratio default is 0.1
    # Item 1: Ans(1.0), Fmt(0.0) -> 1.0*0.9 + 0.0*0.1 = 0.9
    # Item 2: Ans(0.0), Fmt(0.0) -> 0.0
    
    assert len(genome.latest_rewards) == 2
    assert abs(genome.latest_rewards[0] - 0.9) < 1e-6
    assert abs(genome.latest_rewards[1] - 0.0) < 1e-6
    
    # Check history
    assert len(genome.historical_rewards) == 1
    assert abs(genome.historical_rewards[0] - 0.45) < 1e-6  # mean of 0.9 and 0.0

def test_passk_logic(basic_dataset):
    # Configure for pass@2
    basic_dataset.passk = 2
    basic_dataset.passk_proportion = 0.1 # Need 1/2 correct (50%) to pass
    basic_dataset.passk_minimum = 0.9    # Threshold for correct answer
    
    genome = Genome()
    # next() automatically expands the batch by passk
    inputs = basic_dataset.next(population_size=1, mirror=False, center=False)
    
    # Original batch size 2. Expanded by passk=2 -> 4 items.
    # Logic: [Item1_Try1, Item1_Try2, Item2_Try1, Item2_Try2]
    # Targets: [a1, a1, a2, a2]
    
    # Scenario: 
    # Item 1: Try1 is Correct, Try2 is Wrong -> Should trigger pass (1/2 >= 0.1) -> Both get max score
    # Item 2: Try1 is Wrong, Try2 is Wrong -> No pass -> regular scores
    genome.latest_outputs = ["a1", "wrong", "wrong", "wrong"]
    
    basic_dataset.score_all([genome])
    
    # Calculate Expected:
    # Item 1: Max reward is 1.0 (from "a1"). Both outputs get answer reward 1.0.
    #      Final Score = 1.0 * 0.9 (weight) + 0.0 (format) = 0.9
    # Item 2: Max reward 0.0.
    
    rewards = genome.latest_rewards
    assert len(rewards) == 4
    # Both attempts for Item 1 should have high score
    assert abs(rewards[0] - 0.9) < 1e-6
    assert abs(rewards[1] - 0.9) < 1e-6
    # Item 2 failed
    assert abs(rewards[2] - 0.0) < 1e-6

def test_merge_dataset(dataset_for_merge_1, dataset_for_merge_2):
    merged = merge_dataset([dataset_for_merge_1, dataset_for_merge_2], shuffle=False)
    
    assert len(merged.pairs_train) == 3
    # Order should be preserved if shuffle=False
    assert merged.pairs_train[0][0][0]["content"] == "1-q1"
    assert merged.pairs_train[-1][0][0]["content"] == "2-q1"

def test_balanced_merge(dataset_for_merge_1, dataset_for_merge_2):
    # Balanced merge interleaves based on length ratios
    # DS1 (len 2): indices 0.25, 0.75
    # DS2 (len 1): indices 0.5
    # Expected Sort order: DS1[0], DS2[0], DS1[1]
    
    merged = balanced_merge([dataset_for_merge_1, dataset_for_merge_2])
    
    assert len(merged.pairs_train) == 3
    assert merged.pairs_train[0][0][0]["content"] == "1-q1"
    assert merged.pairs_train[1][0][0]["content"] == "2-q1"
    assert merged.pairs_train[2][0][0]["content"] == "1-q2"
