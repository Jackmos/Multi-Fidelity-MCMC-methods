import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
from time import perf_counter
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import tinyDA as tda
from scipy.stats import multivariate_normal, uniform
import arviz as az
import keras
import sys



from module_utils import * 
from data_collection import *
sys.path.append('../utils')
from Structure3 import *
from pathlib import Path

# path to the current notebook
current_file_path = Path().resolve()
# path to the current folder
load_context_functions(current_file_path.parent.name)



def main():
    # introduction of a seed for reproducibility purposes
    seed = 10
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


    example = "Discontinuous"
    fidelity_func = FidelityFunctionModified(example)
    fidelity_func.plot_functions()
    fidelity_func.plot_detailed_functions()

    data=fidelity_func.get_parameters(example)

    # LOW FIDELITY NETWORK

    # parameters of the Neural Network
    bestLF_params = {'lr' : 0.0255, 'kernel_init' : 'glorot_uniform', 'opt' : 'Adam'}
    definition_LF={
            "network_type": "LF",
            "params": bestLF_params,
            "data_train": data["xlf"],
            "output_train": data["Ylf"],
            "N": data["NepoLF"],
            "n": data["Nlf"],
            "train": True,
            "do_HPO": False,
            "verbose": False
        }
    # building of the neural network
    modelLF=NetworkFactory.build_network(**definition_LF)
    # prediction step
    yLF=modelLF.prediction(data["datatest"])
    (mse_LF,R2_LF)=modelLF.performance(data["datatest"],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]))

    # Plot of the estimated results
    plt.figure()
    plt.plot(data["xlf"][:,0],data["Ylf"],'go', label = 'LF training data', markersize = 4)
    plt.plot(data["datatest"][:,0],data["modified_lowfid"](data["datatest"][:,0],data["datatest"][:,1]),'g', label = 'exact LF') 
    plt.plot(data["datatest"][:,0],yLF,'k--', label = 'pred LF')
    plt.legend()
    plt.title('Low fidelity model')
    plt.show()
    
    
    # HIGH FIDELITY NETWORK
    
    # parameters of the Neural Network
    bestHF_params = {'lr' : 0.0255, 'kernel_init' : 'glorot_uniform', 'opt' : 'Adam'}
    definition_HF={
            "network_type": "LF",
            "params": bestLF_params,
            "data_train": data["xhf"],
            "output_train": data["Yhf"],
            "N": data["NepoHF"],
            "n": data["Nhf"],
            "train": True,
            "do_HPO": False,
            "verbose": False
        }
    modelHF=NetworkFactory.build_network(**definition_HF)
    yHF=modelHF.prediction(data["datatest"])
    (mse_HF,R2_HF)=modelHF.performance(data["datatest"],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]))

    # Plot of the estimated results
    plt.figure()
    plt.plot(data["xhf"][:,0],data["Yhf"],'ro', label = 'HF training data', markersize = 4)
    plt.plot(data["datatest"][:,0],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]),'r', label = 'exact HF') 
    plt.plot(data["datatest"][:,0],yHF,'k--', label = 'pred HF')
    plt.legend()
    plt.title('High fidelity model')

    ### 2 steps NEURAL NETWORK ###
    # collection of parameters
    bestLF_params = {
        'lr': 0.0255, 
        'kernel_init': 'glorot_uniform', 
        'opt': 'Adam'
    }

    best_params = {
        'kernel_init': 'uniform', 
        'l2weight': 0.00010018625799978436, 
        'lr': 0.07093837044166487, 
        'nodes': 74.0, 
        'opt': 'Adamax'
    }

    params = [bestLF_params, best_params]

    N = [data["NepoLF"], data["NepoHF"]]
    n = [data["Nlf"], data["Nhf"]]

    # names of the networks 
    names = ['LF', 'HF']

    # collection of the parameters
    definition_2steps = {
        "network_type": "2step",
        "names": names,
        "params": params,
        "data_train": [data["xlf"], data["xhf"]],
        "output_train": [data["Ylf"], data["Yhf"]],
        "N": N,
        "n": n,
        "train": True,
        "do_HPO": False,
        "verbose": False
    }

    # creation of 2 steps model 
    final_model = NetworkFactory.build_network(**definition_2steps)

    if final_model == -1:
        raise ValueError("Failed to build the network. Check the network type and parameters.")

    y_test=final_model.prediction(data["datatest"])

    (_,_)=final_model.performance(data["datatest"],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]))


    ### PLOT of predicted models ###

    plt.figure()
    plt.plot(data["xhf"][:,0],data["Yhf"],'ro', label = 'HF training data', markersize = 4)
    plt.plot(data["datatest"][:,0],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]),'r', label = 'exact HF') 
    plt.plot(data["datatest"][:,0],y_test,'k--', label = 'pred HF')
    plt.legend()
    plt.title('High fidelity model')
    plt.show()
    
    ## INVERSE PROBLEM
    
    (tinyDA_inversion, cuqipy_inversion)=(True,True)
    
    if tinyDA_inversion is True:
        mean_prior = np.array([7.])
        cov_prior = np.diag([1.])

        sigma_noise = [0.05,0.1]
        n_data = [50,30]
        parameters = np.array([15.])
        sigma = np.array([ 4.,2.])
        rwmh_scaling = np.array([ 2.,0.5,1.])
        rwmh_cov = np.eye(1)
        rwmh_adaptive = True
        iterations = 30
        burnin = 20
        n_chains = 2
        algo = "MH"
    
        (_,_)= run_simulation(data["xhf"], mean_prior, cov_prior, data["Yhf"], sigma_noise, n_data, parameters, sigma, rwmh_scaling, rwmh_cov, rwmh_adaptive, iterations, burnin, n_chains, modelLF,algo)

        (_,_)= run_simulation(data["xhf"], mean_prior, cov_prior, data["Yhf"], sigma_noise, n_data, parameters, sigma, rwmh_scaling, rwmh_cov, rwmh_adaptive, iterations, burnin, n_chains, modelHF,algo)
 
        (_,_)= run_simulation(data["xhf"], mean_prior, cov_prior, data["Yhf"], sigma_noise, n_data, parameters, sigma, rwmh_scaling, rwmh_cov, rwmh_adaptive, iterations, burnin, n_chains, final_model,algo)
        
    if cuqipy_inversion is True:
        mean_prior = np.array([7.])
        cov_prior = np.diag([1.])
        Yhf = data["Yhf"]
        sd_noise = [0.1,0.05]
        N = 30
        burn_in = 20
        number_chains = 2
        algo = "MH"
        x_real = np.array([15.])
        proposal_sd = [4.,2.]
        adapt = True
        n_data = [50,30]
            
        # necessary in case off parallelization
        module_directory = os.path.abspath(os.path.join('..', 'utils'))
        os.environ['PYTHONPATH'] = module_directory
        sys.path.append(module_directory)
        
        x_data=  np.linspace(0., 5., n_data[0]).reshape(-1, 1)  # Generate evaluation times
        
        (_,_)=run_simulation_cuqi(data,
        mean_prior,
        x_real,
        N,
        burn_in,
        cov_prior,
        sd_noise,  
        adapt,
        
        proposal_sd,
        number_chains,
        algo,
        x_data,
        n_data,
        final_model,
        parallel=False
        ) 
    
if __name__ == "__main__":
    main()

    