# ~~Back~~propagate: Train LLMs with Evolutionary Strategies!

## Quick Start:
1. Clone this repo: ``git clone https://github.com/Green0-0/propagate``

2. Install vllm and wandb: ``pip install vllm wandb``

Note: Tested and works on windows with a 5090 (no multigpu). Look for a fork of https://github.com/SystemPanic/vllm-windows with the appropriate CUDA versions and remove ``distributed_executor_backend="ray",`` from ``vllm_backend.py``.

3. Login to wandb, and modify ``test.py`` to your needs. Then run the script!

[Learn what the code does](Docs.md)

[What's being worked on](TODO.md)

## Replication results:
Check https://wandb.ai/num110010/propagate_tests.

All work in this repository is based off the work of https://github.com/VsonicV/es-fine-tuning-paper. More papers may be implemented someday.