# Multi-Fidelity MCMC methods for efficient parameter estimation

Politecnico di Milano APSC Course project
Supervisors: Andrea Manzoni and Paolo Conti

This repository contains the code with the Library and the test cases related to implementation and investigation of Multi-fidelity Neural Networks and Markov-chains Monte-Carlo techniques.

# Overview 
`utils\` - folder containing the most important functions for the neural network and bayesian inverse problems routines
It contains:
- `bayesian_utils` - file containing classes necessary for the implementation of MCMC methods. It is related to 2 other libraries, namely CUQIpy and TinyDA. To have more information about the Libraries and how to install them, please of at the voice Dependences
- `functions_to_ray` - file with functions that must be seen as global in order to have a correct execution of parallelization when necessary
- `helper_functions` - file containing some additional classes which collect functions usefull through the whole project
- `MOD_helper` - file containing a class with some functions for Reduced Order Model applications
- `network_utils` - file containing the classes related to neural networks, LSTM networks and Multi-fidelity Neural Networks
`source\` - file containing the classes with data and main function of the test cases. The five files are related with the correspondent five examples described in the report
`test\` - folder containing the most important examples, described in the report

requirements.txt - file containing the libraries and the correspondent versions used to generate the shared results

## Environment
- **Python Version**: This project was developed and tested using Python `3.12.2`
To get a local copy of this project, run the following commands:

```bash
git clone https://github.com/your-username/my-python-project.git
cd my-python-project

## Dependences
- `optuna` [installation guide](https://optuna.readthedocs.io/en/stable/installation.html)
- `cuqi` [git_page](https://github.com/CUQI-DTU/CUQIpy.git)
- `tinyDA` [gitpage](https://github.com/mikkelbue/tinyDA.git)
