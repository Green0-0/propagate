# TODO, in order of priority
- Update lora backend allreduce
- Model seed saving and full model saving
- Test smollm-sft model and mistral-3 3b model, test optimizers on aime

- Dynamic length penalty/length reward through a third reward function on datasets
- Remove format rewards on validation to have it measure pure performance

- Automatic perturbation scale sweeping, readd sigma to step size

- Colab/Kaggle notebooks

- Alignment datasets with RLHF/PPO/DPO to serve as regularizer (prevent overfitting on math) (note: not sure if this will work)
- Clean up code/add documentation
- Optimized lora implementation (max lora limit)

- Improve optimizers
- Speciation

- It should be possible to natively train in a quant format, such as by perturbing the scales and centering.

- Multi-turn training