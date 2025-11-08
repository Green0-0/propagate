# TODO, in order of priority
- Update the allreduce operation to match the official code on both backends. If the momentum optimizer can be made to work, add more optimizers; the optimizers empirically have zero overhead in terms of both memory and runtime.
- **The model saving function needs to be tested; it should be possible to store a model using only its historical seeds**
- Add new datasets; in particular, difficult math datasets, coding datasets, and logical reasoning datasets. Finish up the dataset merging code.
- Implement pass@k training https://arxiv.org/abs/2508.10751, length penalty, curriculum learning (modifying reward function based on performance)

- Create a trainer that trains using standard GA strategies (ie. crossover, etc) for comparison. Implement speciation based on the rewards. Possibly mix in gradients, https://arxiv.org/pdf/2408.07666?
- It should be possible to natively train in a quant format, such as by perturbing the scales and centering.
- Clean up code/add documentation
- Colab/Kaggle notebooks

- Add CMA-ES as tested in https://arxiv.org/abs/2507.04453v1. This will only work with LoRA (+ quant), as it will consume a ridiculous amount of memory otherwise.
- Alignment datasets with RLHF/PPO/DPO to serve as regularizer (prevent overfitting on math) (note: not sure if this will work)
- Multi-turn training