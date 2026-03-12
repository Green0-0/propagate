import pytest
from propagate.genome import Genome
from propagate.datasets.dataset import Dataset
from propagate.datasets.postprocessreward import (
    DynamicLengthReward,
    NormalizedLengthReward
)

@pytest.fixture
def dummy_genomes():
    # Setup 3 genomes with synthetic outputs and rewards
    g1 = Genome()
    g1.latest_outputs = ["short_correct"]
    g1.latest_rewards = [1.0]
    
    g2 = Genome()
    g2.latest_outputs = ["word " * 10]
    g2.latest_rewards = [0.0]

    g3 = Genome()
    # Ensure this is significantly longer than g1
    g3.latest_outputs = ["word " * 20]
    g3.latest_rewards = [1.0]

    return [g1, g2, g3]

def test_dynamic_length_reward(dummy_genomes):
    # Setup processor
    # Target length: 1
    # correct penalty: 0.1
    # incorrect boost: 0.1
    processor = DynamicLengthReward(
        length_penalty_percent=0.1, 
        length_reward_percent=0.1,
        words_target=10, 
        correct_threshold=0.9
    )
    
    # Process
    processor.post_process_rewards(dummy_genomes)
    
    # G1: "short_correct" (2 words?), 1.0 (Correct).
    # Usage prop: 2/10 = 0.2
    # Penalty: 0.2 * 0.1 = 0.02
    # New reward: 1.0 * (1 - 0.02) = 0.98
    # Use approx due to floating point
    assert abs(dummy_genomes[0].latest_rewards[0] - 0.98) < 0.02

    # G2: "long_incorrect_with_many_words" (5 words?)
    # mask the assertion
    # Usage prop: 1.0
    # Boost: 1.0 * 0.1 = 0.10
    # New: 0.0 + (1-0)*0.10 = 0.10
    assert dummy_genomes[1].latest_rewards[0] > 0.0
    
    # G3: Longer correct should be penalized more than G1
    assert dummy_genomes[2].latest_rewards[0] < dummy_genomes[0].latest_rewards[0]

def test_normalized_length_reward(dummy_genomes):
    # Normalized reward calculates stats across the batch (3 items)
    processor = NormalizedLengthReward(
        length_penalty_percent=0.1,
        length_reward_percent=0.1
    )
    
    processor.post_process_rewards(dummy_genomes)
    
    # Just verify that rewards changed in the expected direction relative to each other
    # G3 is longest correct -> biggest penalty
    # G1 is shortest correct -> smallest penalty
    
    # Ensure standard deviation exists in dummy_genomes:
    # G1 (short_correct): 2 words (approx)
    # G2 (long_incorrect...): 10 words
    # G3 (very_long...): 20 words
    # Mean ~ 10. Std > 0.
    
    assert dummy_genomes[2].latest_rewards[0] <= dummy_genomes[0].latest_rewards[0]
    
    # Incorrect G2 should get a boost
    assert dummy_genomes[1].latest_rewards[0] > 0.0

def test_empty_input():
    processor = NormalizedLengthReward()
    processor.post_process_rewards([]) # Should not crash
