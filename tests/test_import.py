import pytest
import importlib

EXTERNAL_LIBS = [
    "torch",
    "ray",
    "vllm",
    "wandb",
    "datasets",
    "peft",
    "math_verify"
]

INTERNAL_MODULES = [
    # General
    "propagate",
    "propagate.genome",
    "propagate.optimizers",
    "propagate.trainer",
    
    # Backend
    "propagate.backend.backend_abc",
    "propagate.backend.vllm_backend",
    "propagate.backend.vllm_lorabackend",
    "propagate.backend.vllm_lorautils",
    "propagate.backend.vllm_tpu_tp_backend",
    "propagate.backend.vllm_tpu_tp_lora_backend",
    "propagate.backend.vllm_utils",

    # Datasets
    "propagate.datasets.dataset",
    "propagate.datasets.countdown_dataset",
    "propagate.datasets.hf_dataset_loader",
    "propagate.datasets.postprocessreward",
    "propagate.datasets.reward",
    "propagate.datasets.rlhf_reward",
]

@pytest.mark.parametrize("library", EXTERNAL_LIBS)
def test_external_import(library):
    """Test (and implicitly assert) that external dependencies can be imported."""
    try:
        importlib.import_module(library)
    except ImportError as e:
        pytest.fail(f"Failed to import external library '{library}': {e}")

@pytest.mark.parametrize("module", INTERNAL_MODULES)
def test_internal_import(module):
    """Test (and implicitly assert) that internal modules can be imported."""
    if "tpu" in module:
        pytest.skip("Skipping TPU backend tests by default (incompatible with standard vllm).")

    try:
        importlib.import_module(module)
    except ImportError as e:
        pytest.fail(f"Failed to import internal module '{module}': {e}")
    except Exception as e:
        pytest.fail(f"Error while importing '{module}': {e}")
