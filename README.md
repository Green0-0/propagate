# ~~Back~~propagate: Train LLMs with Evolutionary Strategies!

## Quick Start:
1. Clone this repo:
``git clone https://github.com/Green0-0/propagate``
2. Install vllm and wandb:
``pip install vllm wandb``
Note: I have gotten this to work on a windows computer with a 5090 (no multigpu). Look for a fork of https://github.com/SystemPanic/vllm-windows with the appropriate CUDA versions and remove the line ``distributed_executor_backend="ray",`` from ``vllm_backend.py``.
3. Login to wandb, and modify ``test.py`` to your needs. Then run the script!

## Replication results:
Check https://wandb.ai/num110010/propagate_tests.

[What's being worked on](TODO.md)

All work in this repository is based off the work of https://github.com/VsonicV/es-fine-tuning-paper. More papers may be implemented someday.