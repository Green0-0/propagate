<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="graphics/headerdark.png">
    <source media="(prefers-color-scheme: light)" srcset="graphics/hl.png">
    <img alt="header" src="graphics/hl.png" width="85%">
  </picture>
  <h2>Train thinking models using evolutionary strategies!</h2>
</div>

### 🏃 Quick Start:
1. Clone this repo: ``git clone https://github.com/Green0-0/propagate``

2. Setup your venv and install vllm: ``https://docs.vllm.ai/en/v0.11.2/getting_started/installation/``

3. Install the dependencies: ``cd propagate && pip install -e .``

*Propagate should work wherever vLLM does, including on windows! Look for a fork of https://github.com/SystemPanic/vllm-windows with the appropriate CUDA version, and remove ``distributed_executor_backend="ray",`` from ``vllm_backend.py``.*

4. Run ``python examples/demo_countdown.py``. You should be prompted to login to wandb, and then training will begin!

### 🖊️ [Work in progress](TODO.md)

### 📖 [Guide](GUIDE.md)

### Credits:
- https://openai.com/index/evolution-strategies/
- https://github.com/VsonicV/es-fine-tuning-paper
- https://github.com/ESHyperscale/HyperscaleES