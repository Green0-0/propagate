### 3 METHOD

This section introduces the basic algorithmic structure of ES, followed by a detailed description of its implementation for LLM fine-tuning.

#### 3.1 BASIC ES ALGORITHM

The ES implementation in this paper is an algorithmically simplified variant of natural evolution strategies (NES) (Wierstra et al., 2008; 2014). The overall design is similar to OpenAI ES (Salimans et al., 2017), which simplified NES with fixed covariance for perturbation noise.

Given a pretrained LLM with initial parameters $\theta_0$ and a target reward function $R(\cdot)$, the task is to fine-tune the parameters so that the reward function is optimized (Algorithm 1). In each iteration, $N$ perturbed models are sampled by adding random Gaussian noise $\epsilon_n$ to their parameters. The noise is i.i.d. in the parameter space, and it is scaled by the hyperparameter $\sigma$. The perturbed models are evaluated to obtain their reward scores $R_n$. The final update of the model parameters aggregates the sampled perturbations by weighting them using their normalized reward scores. The standard update equation $\theta_t \leftarrow \theta_{t-1} + \alpha \cdot \frac{1}{\sigma} \frac{1}{N} \sum_{n=1}^{N} R_n \epsilon_n$ is simplified to $\theta_t \leftarrow \theta_{t-1} + \alpha \cdot \frac{1}{N} \sum_{n=1}^{N} R_n \epsilon_n$ by digesting the term $\frac{1}{\sigma}$ into the learning rate $\alpha$.

To improve scalability, a number of modifications to this basic algorithm were made as detailed in the next section.

#### 3.2 IMPLEMENTATION DETAILS

Algorithm 2, the actual implementation of ES for this paper, expands on the above algorithm in seven ways:

(1) **Noise retrieval with random seeds:** Similar to Salimans et al. (2017); Such et al. (2017), only the random seeds are stored to reduce GPU memory usage. The perturbation noise used during sampling can be retrieved exactly by resetting the random number generator with specific random seeds.
(2) **Parallel evaluations:** In each iteration, the perturbed models can be evaluated fully in parallel by assigning a separate random seed to each process.
(3) **Layer-level in-place perturbation and restoration:** To reduce the peak GPU memory usage, the model parameters are perturbed in-place layer by layer, with corresponding random seeds archived. After evaluation of the perturbed model, the model parameters are restored by subtracting the same noise perturbations using the archived random seeds. For each evaluation process, apart from the model parameters, the only additional memory needed is to store a tensor the size of a layer temporarily.
(4) **Reward normalization:**

---

**Algorithm 2** ES Implementation for LLM Fine-Tuning
```psuedocode
Require: Pretrained LLM with initial parameters $\theta_0$, reward function $R(\cdot)$, total iterations $T$, population size $N$, noise scale $\sigma$, learning rate $\alpha$, number of parallel process $P$.
1: Create $P$ processes, each instantiates a model with the same initial parameters $\theta_0$, with one process as the main process
2: for $t=1$ to $T$ do                           $\triangleright$ ES iterations
3:   Sample random seeds $s_1, s_2, \dots, s_N$
4:   Assign random seeds to $P$ processes
5:   for $n=1$ to $N$ do
6:     For the process handling $s_n$, reset its random number generator using random seed $s_n$
7:     for each LLM layer do                   $\triangleright$ perturbation within current process
8:       Sample noise $\epsilon_{n,l} \sim \mathcal{N}(0, I)$, which has the same shape as the $l$th layer's parameters
9:       Perturb the $l$th layer's parameters in-place: $\theta_{t-1, l} \leftarrow \theta_{t-1, l} + \sigma \cdot \epsilon_{n,l}$
10:    end for
11:    Compute reward for perturbed parameters $R_n = R(\theta_{t-1})$    $\triangleright$ within current process
12:    For the process handling $s_n$, reset its random number generator using random seed $s_n$
13:    for each LLM layer do                   $\triangleright$ restoration within current process
14:      Sample noise $\epsilon_{n,l} \sim \mathcal{N}(0, I)$, which has the same shape as the $l$th layer's parameters
15:      Restore the $l$th layer's parameters in-place: $\theta_{t-1, l} \leftarrow \theta_{t-1, l} - \sigma \cdot \epsilon_{n,l}$
16:    end for
17:  end for
18:  Normalize the reward scores by calculating the $z$-score for each $R_n$: $Z_n = \frac{R_n - R_{mean}}{R_{std}}$,
19:  where $R_{mean}$ and $R_{std}$ are the mean and standard deviation of $R_1, R_2, \dots, R_N$.
20:  for $n=1$ to $N$ do                         $\triangleright$ in main process only
21:    Reset current random number generator using random seed $s_n$
22:    for each LLM layer do
23:      Sample noise $\epsilon_{n,l} \sim \mathcal{N}(0, I)$, which has the same shape as the $l$th layer's parameters
24:      Update $l$th layer's parameters in-place as $\theta_{t,l} \leftarrow \theta_{t-1, l} + \alpha \cdot Z_n \epsilon_{n,l}$
25:    end for
26:  Update the model parameters of all processes to $\theta_t$
27: end for
```

The rewards of the perturbed models are normalized using $z$-score within each iteration, so that the normalized rewards for each iteration have a mean of 0 and standard deviation of 1. This normalization makes the reward scale consistent across iterations and tasks.
(5) **Greedy decoding:** The perturbed models use greedy decoding to generate the responses for reward evaluations. As a result, the perturbed models are evaluated deterministically, so that all performance differences come from the exploration in parameter space instead of action space.
(6) **Decomposition of the parameter update:** At the end of each iteration, the aggregated update of model parameters is performed in-place in a decomposed manner, gradually adding up layer by layer and seed by seed, significantly reducing the peak GPU memory needed.
(7) **Learning rate digestion:** The standard update equation $\theta_t \leftarrow \theta_{t-1} + \alpha \cdot \frac{1}{\sigma} \frac{1}{N} \sum_{n=1}^{N} R_n \epsilon_n$ is simplified to $\theta_t \leftarrow \theta_{t-1} + \alpha \cdot \frac{1}{N} \sum_{n=1}^{N} R_n \epsilon_n$ by digesting the term $\frac{1}{\sigma}$ into the learning rate $\alpha$, simplifying the computation and parametric setup.

In order to keep the algorithm simple, common enhancements in OpenAI ES (Salimans et al. (2017) such as rank transformation of rewards (Wierstra et al., 2014), mirrored sampling (Sehnke et al., 2010), weight decay, and virtual batch normalization (Salimans et al., 2016) are not used, and neither are more advanced optimizers like Adam (Kingma & Ba, 2015). They can be included in to improve results in future work.