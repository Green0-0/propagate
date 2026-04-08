# TODO:
### In progress:
- Broadcast sync optimizer state
- Cleanup lora backend (Check vLLM async), have it delete the lora adapter
- Give optuna trainer more pruning options, configurable pruning
- Update tutorials
- Direct ES hessian estimator (e^2 - 1) with tensor averaging and ways to reduce cross-tensor variance

### Experimental:
- New optimizer chains, NAdam, Muon, fp32, rank-centering, SignSGD

### Planned:
- Line search genome on previous gradient (Changes to trainer)

- Per layer optimizers, exclude non-trainable params from optimization

- RLHF: implemented, may work, requires a lot of testing
- TPU Support (currently broken because of optimizer update)

- Colab/Kaggle notebooks

- Optimized LoRA backend with round robin adapter inference

- Env training (ie. agentic, coding, multiturn etc), requires dataset input rework
- Proper model saving: currently waiting for official ES github to find a solution, will write one myself if progress stalls

### Possible:
- llamacpp (cpu training) support
- Trained demo models, benchmark suite
- Native quant format training