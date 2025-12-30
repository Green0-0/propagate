# TODO:
### In progress:
- Rescale lr by sigma, automatic sigma sweeping
- Colab/Kaggle notebooks

### Experimental:
- Optimizer rework: implement nesterov momentum/adam (maybe muon too?)
- RLHF: implemented, may work, requires a lot of testing
- TPU Support (exists and tested but is suboptimal; lora doesn't work, may need to wait for vLLM updates)

### Planned:
- Optimized LoRA backend with round robin adapter inference
- Write unit tests
- Random sampling rework (hadamard rachemacher)
- Env training (ie. agentic, coding, multiturn etc), requires dataset input rework
- Proper model saving: currently waiting for official ES github to find a solution, will write one myself if progress stalls

### Possible:
- Sglang support
- LMDeploy support
- Albatross support
- Trained demo models, benchmark suite
- Native quant format training
