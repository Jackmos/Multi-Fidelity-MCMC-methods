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
- `Diffusion_helper` - class for example I
- `Bentchmark_class` - class for example II
- `System_solver` - class for example III
- `Burger_class` - class for example IV
- `Reaction_diffusion_class` - class for example V


`test\` - folder containing the most important examples, described in the report
- `tutorial.py` - the simplest example (first part of example II in the report). Can be used to understand the general implementation of multi-fidelity neural networks and inverse processes
- `benchmark_modified_case.py` - Generalization of example II in the report
- `LV-1par_2step.py` - related to the first part of example III in the report. It contains the first Lotka-Volterra example with the procedures to run Delayed acceptance and adaptive metropolis
- `LV-4par_3step.py` - related to the second part of example III in the report. It contains the second Lotka-Volterra example with the procedures to run MLDA
- `LSTM_burger.py` - related to example IV in the report. LSTM implementation to build the relationship between LF data and HF data
- `Diffusion\` - files to run the different examples presented in example I. ATTENTION: it requires to install in the `test\` folder a folder of data `DATA_reaction_diffusion_test_case`. Since this folder is very heavy, it is not installed on git, but may be available on request
- `Reaction-diffusion\` - contains the files for example V, the folder contains the file to run the code, the data related to that. Since training the network requires much time, a saved version is shared 

`Doc\`- folder containing the final report (will be updated with the presentation very soon!)

requirements.txt - file containing all the libraries and the correspondent versions for the project, used to generate the shared results

## Environment
- **Python Version**: This project was developed and tested using Python `3.12.2`
To get a local copy of this project, run the following commands:

```Bash
git clone https://github.com/Jackmos/Multi-Fidelity-MCMC-methods.git

to run the 

## Dependences
- `optuna` [installation guide](https://optuna.readthedocs.io/en/stable/installation.html)
- `cuqi` [git_page](https://github.com/CUQI-DTU/CUQIpy.git)
- `tinyDA` [gitpage](https://github.com/mikkelbue/tinyDA.git)
