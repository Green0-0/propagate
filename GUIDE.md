Before reading the guide, try one of the example scripts, wait an hour, and look through the logs/graphs! It's better to observe the code in action than to read about it.

---
## Where am I getting my numbers?
I have ran 150+ experiments to determine the best hyperparameters for a variety of configs.

See them here:

- [Current experiments](https://wandb.ai/num110010/propagate_optimizers)
- [Previous experiments](https://wandb.ai/num110010/propagate_tests?nw=nwusernum110010)


## Should I use LoRA?
If you are hardware constrained (less compute than a 5090) you should NOT use LoRA. LoRA maximizes GPU utilization but increases compute/memory overhead. Unlike standard training setups, LoRA is only used as a proxy for the perturbations, so it does not save memory.

However, if you are extremely memory constrained, you might be forced to use LoRA, but it will be slower. LoRA enables quantized training of models in any format that vLLM can load (GPTQ, GGUF, AWQ, etc). Beware that vLLM will not necessarily compute samples faster just because you are running a quantized model. However, performance loss with quantized models was negligible in my testing, so feel free to experiment.

## Sample Size Tuning:
The number of samples you must calculate in a single step is equal to ``population size * batch size``.

The duration of training is directly proportional to the number of samples multiplied by the number of training steps.

You almost never want to change the number of training steps (100-250 is okay); the model converges very quickly and doesn't improve afterwards.

There are two possible ways to set the number of samples:

1. You are NOT using LoRA, and thus have weak hardware. Follow the standard training setup from https://arxiv.org/abs/2509.24372: high batch size, low population size. This maximizes GPU utilization and minimizes overhead, at the cost of model capability.

Recommended numbers are 100-300 batch size and 10-30 population size. Aim for a sample size of at least 3000, ideally 5000+.

2. You are using LoRA, and thus have high-end hardware and/or multiple GPUs. Follow the training setup from https://arxiv.org/abs/2511.16652: low batch size, high population size. This maximizes model capability and GPU utilization, at the cost of high overhead. It is also more sample efficient.

Recommended numbers are 25-100 batch size and 40-300 population size. Aim for a sample size of at least 2500. Capability increases with sample size.

Note that mirrored doubles your population size. All the numbers above are for the effective population size, not the number you set the hyperparameter to.

## LR Tuning:
Warning: A major change will be made to the LR calculation soon.

Warning 2: If you train a different model other than qwen or smollm, I will be unable to help you with the hyperparameter tuning, and you will have to manually sweep the perturb scale and LR.

For now, follow the hyperparameters that are used in the example scripts. These have been pretuned to work well. You should NOT use any of the optimizers, they have been found to decrease performance. 

## Update Rule
The basic ES loop is quite simple, so I'll explain it below:

First, we need a trained model ``M`` to finetune, and a function ``R(m)`` that evaluates our model's performance. 

We begin by generating a set of seeds ``s1``, ``s2``, ...

For each seed, we generate gaussian noise with the same shape as ``M``, call it ``ϵ1``, ``ϵ2``, ...

We then calculate ``r1 = R(M + ϵ1)``, ``r2 = R(M + ϵ2)``, ...

This is the score of the model after we add some random noise to each of its parameters.

Note: To restore the ``M`` after adding ``ϵi`` to it, we can just subtract ``ϵi``.

Now, we have one reward per seed.

We calculate the mean of the rewards (``rmean``) and the standard deviation (``rstd``).

Then, the model parameters are updated as follows:

``M = M + (learning rate) * ϵi * (ri - rmean) / rstd`` for every seed ``i``.

In other words, we adjust the rewards such that seeds with bigger rewards yield a more positive contribution to the gradient, and vice versa.
Then we just repeat the process! And somehow, almost magically, our model learns.

For mirrored sampling, we copy each seed ``si`` to ``-si`` with ``-ϵi``. The update rule is equivalent, but you should drop ``rmean`` and ``rstd`` to yield the equation:

``M = M + (learning rate) * ϵi * ri``

where ``ϵi`` might be positive or negative.

This equation is perhaps more intuitive: the bigger the reward, the more weight we give the gradient. The update naturally cancels itself out because we have both ``ϵi`` and ``-ϵi`` in opposite directions.

## Update Rule (LoRA)

A couple changes are required to make LoRA work. You are free to test yourself, but below was my intuition.

First of all, observe the perturbation: ``M + AB``.

If we perturb ``A`` and ``B`` simultaneously, we get a quadratic ``(A + ϵai)(B + ϵbi)``.

However, in the update, we will do something like ``(A + ϵa1 + ϵa2 + ...)(B + ϵb1 + ϵb2 + ...)``.

Unfortunately, this does not translate directly from the quadratic, and introduces cross terms of the form ``ϵai * ϵbi``.

We can thus instead perturb the matrices in alternating fashion: ``(A + ϵai)(B)`` then ``(A)(B + ϵbi)``.

(We could also use the eggroll update which does not have this issue, but I haven't gotten around to implementing it yet.)

Now, if we look at ``(A + ϵai)(B)``:

notice that the result is ``AB + Bϵai``.

In other words, the magnitude of your perturbation is scaled by B.

This can cause divergance, so its best to normalize the magnitude of the perturbation by the magnitudes of A, or B, or both.

