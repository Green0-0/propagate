# TODO
- Check if the two existing backends work properly
- Submit tests of 250 iterations for pop = 10, 20, 30

- Create a backend following the official implementation
- Create a backend for single gpu with no ray so that it will work on windows
- Create a backend for 4 bit bnb (maybe apply modifications to the scales and zeropoints of the quant config, or dequant the whole model and keep a sep. copy on cpu)
- Create a dataset for multiplication, AIME, GSM8K, GPQA
- Write a backup method that backs up the genome by saving all historical seeds with their backprop weight, so that you can restore the model from the seeds
- Write wandb integration, model saving at the end
- Submit many tests

- Write a trainer that performs crossover for n generations, and takes gradient step at n (where n=gradient step counter)
- Write a trainer that performs speciation and does gradient step per species every x generations when x != n (where x=species gradient step counter)
- Submit many tests

- Write google colab notebook
- Write kaggle notebook