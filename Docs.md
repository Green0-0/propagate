### dataset.py (``Dataset``)
- Loads a dataset with a corresponding reward function for each row of data
- Generates train/test splits (``generate_test_split``)
- Functions as an iterator and retrieves new data to train on (mini-batches); caches this new data so it can be used to score genome outputs with ``score``
- Saves some formatting information for the prompts

### countdown_dataset.py (see: ``load_countdown_dataset``; returns your dataset)
- Standard countdown dataset, but modifications are made such that the prompt is formatted accordingly
- The reward functions have been slightly adjusted from the official implementation to be more consistent

### generic_rewards.py
- ``format_reward`` generates a reward for each token correctly placed in the output, where the output is formatted as:

{start_think_token}...{end_think_token}...{start_answer_token}...{end_answer_token} and any of the tokens may be excluded. {start_think_token} will conflict with a prompt suffix, use only one.

### genome.py(``Genome``)
- This represents a set of seeds that constitute an updated model, along with their associated weights and historical rewards.
- **Note that the genome stores its own latest outputs and rewards, not the trainer.** In fact, the trainer never even touches the outputs, asides from logging.
- Applies and reverts the noise update to the backend based on its own seeds
- Supports mirrored sampling

### optimizers.py(``Optimizer``)
- These optimize your genome and create the gradient steps. The standard operation to do this is a merge which merges the seeds of all the genomes, calculates the gradients on the new seeds, and performs an average on the old seeds.
- All optimizers come with an lr scheduler and warmup options.
- ``TestMaxOptimizer`` exists only for testing, do not use it
- Use ``SimpleOptimizer``
- ``MomentumOptimizer`` is a no-overhead implementation of momentum. It does not cost extra memory (or rather, the memory cost is marginal and only contains the cost of storing the seeds), and did not empirically increase the runtime (it only does a few simple operations anyways).
- **However momentum caused divergence in the model for unknown reasons; do not use it!**

### trainer.py(``SimpleTrainer``)
- There is currently only one trainer. It mainly does logging with wandb; the train loop is less than 40 lines long.

### test.py
- This is what you run!

### backend.py
- This loads the model for inference, updates it, and produces responses.

### vllm_backend.py / vllm_utils.py (VLLMBackend)
- Copied from the actual vLLM implementation at https://github.com/VsonicV/es-fine-tuning-paper (credits go to them)
- Runs a model using vLLM and ray
- Runs one copy of the model on each GPU (TP adds overhead and doesn't serve a purpose here); when batching sends jobs to each free gpu and waits until a job is finished, then queues the next job
- Performs a sync after finishing the set of jobs by performing an all reduce on the model weights and then dividing by the number of models

*[Note: please ignore the depreciated backends, they are slow and useless]*

The standard training loop looks like this:
1. Generate n genomes that are perturbed. 
2. Get the next mini-batch from the dataset. ``dataset.next()``
3. Sample outputs with the backend and genomes: ``backend.generate_outputs(self.genomes, self.dataset.suffix, inputs)``.
4. Run ``dataset.score(genome)`` on each genome.
5. Update with the optimizer, note that the optimizer returns a genome representing the base genome for the next generation:

``new_genome = self.optimizer.get_step(self.genomes, self.iteration_count)``
6. Update the backend with this new genome (if you don't care about history), or use it as the base to generate new genomes. 