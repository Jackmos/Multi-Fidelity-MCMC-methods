# Multifidelity MCMC for parameter estimation 

Politecnico di Milano APSC Course project
# BENCHMARK CASE
In this folder, is contained a preliminary case. After the training of simple Multifidelity Neural Networks, a classical MCMC is used to solve the inverse problem.
The cuqipy library is exploited.

The folder contains: 

`old\` - contains old files  
- `ann_functions` - collection of the functions used to train the NN
- `generalization` - introduces the class which denotes the Multifidelity structure
- `Bench1D_2step_MCMC` - contains the definition of the problems, the forward step using the Neural network and the inversion exploiting MCMC
- `finalModel` - MF surrogate
- `modelLF` - LF surrogate
- `modelHF` - HF surrogate