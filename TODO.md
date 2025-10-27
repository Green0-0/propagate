# TODO, in order of priority
- The all reduce operation for syncing gpus and the momentum optimizer need review. If the momentum optimizer can be made to work (I'm not sure why its diverging), add more optimizers; the optimizers empirically have zero overhead in terms of both memory and runtime.
- **The model saving function needs to be tested; it should be possible to store a model using only its historical seeds**
- Add new datasets; in particular, difficult math datasets, coding datasets, and logical reasoning datasets. Finish up the dataset merging code.
- Create a trainer that trains using standard GA strategies (ie. crossover, etc) for comparison. Implement speciation based on the rewards. Possibly mix in gradients, https://arxiv.org/pdf/2408.07666?
- Implement pass@k training https://arxiv.org/abs/2508.10751, length penalty, curriculum learning (modifying reward function based on performance)

- Create an optimized LoRA backend

It should be possible to get an extremely optimized implementation like this: Create n loras equal to your population size. Split them amongst the vllm engines (TP has too much overhead). Have vllm load the loras. Get a reference to each lora and have a specific genome apply its update to that lora. Run every genome simultaneously with vllm's native lorarequest, which achieves the greatest batch utilization.

- With an optimized LoRA backend, it becomes possible to train in any quant format (that supports LoRA). In particular, training in AWQ or the neuralmagic formats would be nice.
- Additionally, it should be possible to natively train in a quant format, such as by perturbing the scales and centering.
- Clean up code/add documentation
- Colab/Kaggle notebooks

- Add CMA-ES as tested in https://arxiv.org/abs/2507.04453v1. This will only work with LoRA (+ quant), as it will consume a ridiculous amount of memory otherwise.
- Alignment datasets with RLHF/PPO/DPO to serve as regularizer (prevent overfitting on math) (note: not sure if this will work)