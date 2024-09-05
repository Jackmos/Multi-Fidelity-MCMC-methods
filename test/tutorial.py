# # Tutorial example

# In this notebook, a simple example of comparison of Low-fidelity (LF), High-fidelity (HF) and Multi-fidelity (MF) networks is shown.
# It is possible to notice that the interaction of LF and HF data and structures in a 2-step network can be benefical to build a performant regression model.
# Here, 3 possible simple example can be created

from utils.network_utils import NetworkConfig,NetworkFactory
from utils.bayesian_utils import BayesianInverseProblem_NN
from source.Bentchmark_class import Benchmark_functions
import sys #<-
import os

import numpy as np
import tensorflow as tf
import warnings
import keras

import logging
#from pathlib import Path

# Suppress DeprecationWarnings from TensorFlow
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

# Set TensorFlow logging level to ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Append the utils directory to the system path
#sys.path.append('../utils')

# Import necessary modules from utils

# Load context functions
#current_file_path = Path().resolve()  # Get the current notebook path
#load_context_functions(current_file_path.parent.name)  # Load context functions using the parent folder name

 # Restore the output


# introduction of a seed for reproducibility purposes
def set_seed():
    seed = 10
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


# Main function
def main_function():
# ## Presentation of the model 


    set_seed()
    # Here, the model can be chosen between "Basic_regression", "Discontinuous_regression" and "Oscillatory_regression". 
    example ="Discontinuous_regression"                     
    fidelity_func = Benchmark_functions(example)

    # The plots show how the chosen function varies with respect to a parameter \delta and 
    # the comparison between "high-fidelity" (i.e. very precise) and "low-fidelity" data
    fidelity_func.plot_functions()
    fidelity_func.plot_detailed_functions()

    # data are saved and taken from external file 
    data=fidelity_func.get_parameters(example)  

    ##### Low-fidelity Neural Network #####

    # Example of Neural network built on low fidelity data. It is possible to notice that 
    # the model is not well approximated in None of the 3 cases
    
    # parameters of the Neural Network
    bestLF_params = {'lr' : 0.0255, 
                    'kernel_init' : 'glorot_uniform', 
                    'opt' : 'Adam'}

    # inputs for the definition of a Single level Neural Network
    definition_LF={
            "network_type": "LF",               
            "network_parameters": bestLF_params,    # parameters to define the network
            "dataset_train": data["xlf"],           # training set
            "output_train": data["Ylf"],            # output of the training set    
            "epochs_number": data["NepoLF"],        # Epochs
            "batch_size": data["Nlf"],              # batch size
            "train": True,                          # The NN is trained
            "do_HPO": False,                        # Do not Hyperparameter Optimization, please, notice that the process requires some time. 
                                                    # If parameters with better estimated performance are found, they will substitute the actual parameters 
            "verbose": False                        # reduce printed unnecessary output
        }

    # initialization of the neural network
    config = NetworkConfig(**definition_LF)
    modelLF=NetworkFactory.build_network(config,fidelity_func.getModel)

    # prediction step
    yLF=modelLF.prediction(data["datatest"])
    # performance measurement
    (mse_LF,R2_LF)=modelLF.performance(data["datatest"],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]))



    # plot the real and estimated relation, it allows a qualitative and immediate comparison 
    fidelity_func.save_comparison_plot(data, yLF, 'low_fidelity_model_plot.png',fidelity_level="Low")

    ##### High-fidelity Neural Network #####
    # Example of Neural network built on high fidelity data. The performance does not reach the same level as 
    # the multifideility correspondent in any of the tested cases

    # parameters of the Neural Network
    bestHF_params = {'lr' : 0.0255, 
                    'kernel_init' : 'glorot_uniform', 
                    'opt' : 'Adam'}

    # inputs for the definition of a Single level Neural Network
    # the definition is very similar to its LF counterpart, but with high-fidelity data
    definition_HF={
            "network_type": "LF",
            "network_parameters": bestHF_params,
            "dataset_train": data["xhf"],               # high_fidelity data are imported
            "output_train": data["Yhf"],
            "epochs_number": data["NepoHF"],
            "batch_size": data["Nhf"],
            "train": True,
            "do_HPO": False,
            "verbose": False
        }

    # initialization of the neural network
    config = NetworkConfig(**definition_HF)
    modelHF=NetworkFactory.build_network(config,fidelity_func.getModel) 

    yHF=modelHF.prediction(data["datatest"])
    (mse_HF,R2_HF)=modelHF.performance(data["datatest"],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]))

    fidelity_func.save_comparison_plot(data, yHF, 'high_fidelity_comparison.png')


    # The exploitation of 2 connected neural networks can improve our ability to approximate a target function 

    ### 2 steps NEURAL NETWORK ###

    # parameters for the LF (first) Neural Network
    bestLF_params = {
        'lr': 0.0255, 
        'kernel_init': 'glorot_uniform', 
        'opt': 'Adam'
    }

    # parameters for the HF (second) Neural Network
    best_params = {
        'kernel_init': 'uniform', 
        'l2weight': 0.00010018625799978436, 
        'lr': 0.07093837044166487, 
        'nodes': 74, 
        'opt': 'Adamax'
    }


    # Collection of elements of both networks in a ordered list
    params = [bestLF_params, best_params]
    N = [data["NepoLF"], data["NepoHF"]]
    n = [data["Nlf"], data["Nhf"]]
    # names of the networks 
    names = ['LF', 'HF']


    # collection of the parameters for the "2-step Neural Network"
    # you can notice that the definition is very similar to the one of a classical neural network, but with a list of elements, 
    # which must be in the right element following the sequence of the networks
    # Here a 2 levels case is shown, but this can be generalized to any number of levels after providing 
    # the definition of the structure of the NNs in the support file 
    definition_2steps = {
        "network_type": "2step",
        "names": names,
        "network_parameters": params,
        "dataset_train": [data["xlf"], data["xhf"]],
        "output_train": [data["Ylf"], data["Yhf"]],
        "epochs_number": N,
        "batch_size": n,
        "train": True,
        "do_HPO": False,
        "verbose": False
    }



    # initialization of the neural network
    config = NetworkConfig(**definition_2steps)
    final_model=NetworkFactory.build_network(config, fidelity_func.getModel) 

    # analysis of the results
    y_test=final_model.prediction(data["datatest"])
    (mse_MF,R_MF)=final_model.performance(data["datatest"],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]))



    # plot of the result. It is possible to notice that this system is showing stronger performance with 
    # respect to NNs both trained and tested on Low and high fidelity data
    fidelity_func.save_comparison_plot(data, y_test, 'Multi-Fidelity Model.png')



    #### Bayesian Inverse Problem  #### 

    # In this part, given a dataset with some observations related to the parameter we want to estimate it through a Bayesian approach 
    set_seed()



    parameters =np.array([[3.75]])          # the chosen parameter
    mean_prior =np.array([10.])             # mean of the initial gaussian
    cov_prior=np.diag([ 1])                 # covariance matrix of the prior 

    cov_noise_bounds = (0.25,0.5)           # noise covariance on the observation 
    n_data_bounds = (30,50)                 # range of number of observations 
    sigma_bounds = ( 0.5,1.5)               # standard deviation of the likelihood 
    rwmh_scaling_bounds =  (0.5,1.5)        # rwmh step 
    rwmh_cov = 1
    rwmh_adaptive = True
    iterations = 1500                       # iterations of the markov chains 
    burnin = 1000                           # burn-in steps 
    n_chains = 2                            # number of chains 
    algo = "MH_tiny"                        # Metropolis-Hastings of tinyDA library


    # module_directory = os.path.abspath(os.path.join('..', 'utils'))
    # os.environ['PYTHONPATH'] = module_directory
    # sys.path.append(module_directory)

    # definition of the object for Bayesian Inversion Problem
    BIP=BayesianInverseProblem_NN(algorithm_name=algo,forward_NN=final_model)

    # hyperparameter optimization 
    BIP.hpo(inputs_HF=data["xhf"], 
        domain_bounds=(0.,5.),
        mean_prior=mean_prior, 
        cov_prior=cov_prior, 
        output_HF=data["Yhf"], 
        sigma_noise=cov_noise_bounds, 
        number_data=n_data_bounds,
        real_parameters=parameters, 
        sigma=sigma_bounds,
        rwmh_scaling=rwmh_scaling_bounds, 
        rwmh_covariance=rwmh_cov, 
        rwmh_adaptive=rwmh_adaptive, 
        iterations=iterations, 
        burn_in=burnin, 
        n_chains= n_chains, 
        levels=1,  
        force_sequential=True,
        initial_points_optimizer=3,
        iterations_optimizer=2              #   steps of the optimizer:  initial_points_optimizer + iterations_optimizer 
        )


    # ### with the best found parameters we can rerun a final estimation
    # (alternatively, one can choose the parameters and insert them in the run function)
    set_seed()

    BIP.run(inputs_HF=data["xhf"], 
        domain_bounds=(0.,5.),
        mean_prior=mean_prior, 
        cov_prior=cov_prior, 
        output_HF=data["Yhf"], 
        real_parameters=parameters, 
        rwmh_adaptive=rwmh_adaptive, 
        iterations=iterations, 
        burn_in=burnin, 
        n_chains= n_chains, 
        levels=1,  
        force_sequential=True,
    )




if __name__ == "__main__":
    main_function()