# Multifidelity MCMC for parameter estimation 

Politecnico di Milano APSC Course project

# Overview 

Each folder but utils contains: 
- 'data_collection' - file which contains a class to define the properties of the considered test case
- 'module_utils' - file contatining some case dependent functions necessary to define the Neural Network or the inverse problem 

Folder 'utils\' is made of functions used by all the analyzed cases, both for the complete definition of Multifidelity Neural Networks and BIPs (Bayesian Inverse Problems). For convenience, the functions are splitted in 3 files
- 'Structure'
- 'Helpers'
- 'BIP_functions'

# Project Explanation

## Usage

## Dependences
- `optuna` [installation guide](https://optuna.readthedocs.io/en/stable/installation.html)
- `cuqi` [git_page](https://github.com/CUQI-DTU/CUQIpy.git)
- `tinyDA` [gitpage](https://github.com/mikkelbue/tinyDA.git)
## Environment
