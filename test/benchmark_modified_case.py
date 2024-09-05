from utils.network_utils import NetworkConfig,NetworkFactory
from utils.bayesian_utils import BayesianInverseProblem_NN
from source.Bentchmark_class import Benchmark_functions

import numpy as np

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import keras
import sys
import os

# introduction of a seed for reproducibility purposes
def set_seed():
    seed = 10
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def main_function():

    example = "Basic"
    output_folder = f'modified_{example}_output'

    fidelity_func = Benchmark_functions(example)

    fidelity_func.plot_functions(output_folder)
    fidelity_func.plot_detailed_functions(output_folder)


    data=fidelity_func.get_parameters(example)



    # introduction of a seed for reproducibility purposes
    set_seed()


    ### 2 steps NEURAL NETWORK ###
    # collection of parameters


    bestLF_params = {
        'lr': 0.0255, 
        'kernel_init': 'glorot_uniform', 
        'opt': 'Adam'
    }

    # best parameters found for the HF (second) Neural Network
    if example == "Discontinuous":
        new_parameters={'nodes': 22, 
                        'l2weight': 0.04620166159124063, 
                        'lr': 0.03748985002613576, 
                        'kernel_init': 'uniform', 
                        'opt': 'Adam'}
    elif example == "Basic":

        new_parameters={'nodes': 22,
                        'l2weight': 0.004589276081865305,
                        'lr': 0.04011826688629131,
                        'kernel_init': 'glorot_uniform',
                        'opt': 'Adamax'}
        
    params = [bestLF_params, new_parameters]


    N = [data["NepoLF"], data["NepoHF"]]
    n = [data["Nlf"], data["Nhf"]]

    # names of the networks 
    names = ['LF', 'HF']

    # collection of the parameters
    definition_2steps = {
        "network_type": "2step",
        "network_parameters":params,
        "names": names,
        "dataset_train": [data["xlf"], data["xhf"]],
        "output_train": [data["Ylf"], data["Yhf"]],
        "dataset_validation":[data["x_vallf"],data["x_valhf"]],
        "output_validation":[data["dataval_lf"],data["dataval_hf"]],
        "epochs_number": N,
        "batch_size": n,
        "train": True,
        "do_HPO": False,
        "verbose": False
    }

    # creation of 2 steps model 
    final_model = NetworkFactory.build_network(NetworkConfig(**definition_2steps),fidelity_func.getModel)


    y_test=final_model.prediction(data["datatest"])

    (mse_MF,R_MF)=final_model.performance(data["datatest"],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]))

    ### PLOT of predicted models ###

    fidelity_func.save_comparison_plot(data,y_test, f"multifidelity_{example}", output_folder)


    ### LF NEURAL NETWORK ###

    # introduction of a seed for reproducibility purposes
    set_seed()


    # parameters of the Neural Network
    bestLF_params =  {
        'lr': 0.0255, 
        'kernel_init': 'glorot_uniform', 
        'opt': 'Adam'
    }
    definition_LF={
            "network_type": "LF",
            "network_parameters": bestLF_params,
            "dataset_train": data["xlf"],
            "output_train": data["Ylf"],
            "epochs_number": data["NepoLF"],
            "batch_size": data["Nlf"],
            "train": True,
            "do_HPO": False,
            "verbose": False
        }
    # building of the neural network
    modelLF=NetworkFactory.build_network(NetworkConfig(**definition_LF),fidelity_func.getModel)
    # prediction step
    yLF=modelLF.prediction(data["datatest"])
    (mse_LF,R2_LF)=modelLF.performance(data["datatest"],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]))

    fidelity_func.save_comparison_plot(data,yLF, f"LF_{example}", output_folder)

    ### HF NEURAL NETWORK ###


    # parameters of the Neural Network
    bestHF_params ={
        'lr': 0.0255, 
        'kernel_init': 'glorot_uniform', 
        'opt': 'Adam'
    }
    definition_HF={
            "network_type": "LF",
            "network_parameters": bestHF_params,
            "dataset_train": data["xhf"],
            "output_train": data["Yhf"],
            "epochs_number": data["NepoHF"],
            "batch_size": data["Nhf"],
            "train": True,
            "do_HPO": False,
            "verbose": False
        }
    
    modelHF=NetworkFactory.build_network(NetworkConfig(**definition_HF),fidelity_func.getModel)
    yHF=modelHF.prediction(data["datatest"])
    (mse_HF,R2_HF)=modelHF.performance(data["datatest"],data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]))

    fidelity_func.save_comparison_plot(data,yHF, f"HF_{example}", output_folder)


    # # Prediction of a parameter 2 step nn


    ## (the seeds are always restarted to be sure to get the same results as reported)

    module_directory = os.path.abspath(os.path.join('..', 'utils'))

    # Add the 'utils' directory containing Structure.py to the PYTHONPATH
    os.environ['PYTHONPATH'] = module_directory
    sys.path.append(module_directory)

    # redirected to the examples related to the chosen case

    if example=="Discontinuous":

        mean_prior = np.array([9])
        cov_prior = np.diag([1])
        rwmh_adaptive=True
        iterations=2000
        burnin=1250
        n_chains=2
        parameters = np.array([15.])

        set_seed()

        algo="MH_tiny"
        BIP=BayesianInverseProblem_NN(algorithm_name=algo,forward_NN=final_model)

        BIP.run(inputs_HF=data["datatest"], 
        domain_bounds=(0.,5.),
        mean_prior=mean_prior, 
        cov_prior=cov_prior, 
        output_HF=data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]),
        real_parameters=parameters, 
        rwmh_adaptive=rwmh_adaptive, 
        iterations=iterations, 
        burn_in=burnin, 
        n_chains= n_chains, 
        levels=1,  
        force_sequential=True,
        sigma_noise=0.0511,
        sigma=1.331,
        rwmh_scaling=1.5,
        number_data=37
        )
    
    elif example == "Basic":
        mean_prior = np.array([15])
        cov_prior = np.diag([5.])

        parameters = np.array([21.])

        cov_noise = 0.09649
        n_data= 31
        iterations = 2000
        burnin = 1250
        number_chains = 2
        proposal_sd = 0.6484 
        scaling = 1.
        adapt=True
        set_seed()
        algo="pCN"    # change in pCN to see preconditioned Crank-Nicholson
        BIP=BayesianInverseProblem_NN(algorithm_name=algo,forward_NN=final_model)

        BIP.run(
            inputs_HF=data["datatest"], 
            output_HF=data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]), 
            mean_prior= mean_prior,
            real_parameters=parameters,
            iterations= iterations,
            burn_in= burnin,
            cov_prior= cov_prior,
            sd_noise=cov_noise,
            proposal_sd= proposal_sd,
            num_data=n_data,
            scale= scaling,
            adapt=adapt,
            number_chains= number_chains,
            domain_bounds=(0.,5.),

            parallel=False
        )

        set_seed()

        # tinyDA version

        sigma_noise = 0.15
        n_data = 37
        sigma = 2.5
        rwmh_scaling = 2.911  
        rwmh_cov = np.eye(2)
        rwmh_adaptive = True
        iterations = 2000
        burnin = 1250
        n_chains = 2
        algo = "MH_tiny"
        set_seed()
        BIP=BayesianInverseProblem_NN(algorithm_name=algo,forward_NN=final_model)


        BIP.run(
            inputs_HF=data["datatest"], 
            output_HF=data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]), 
            mean_prior= mean_prior,
            real_parameters=parameters,
            iterations= iterations,
            burn_in= burnin,
            rwmh_adaptive=rwmh_adaptive, 
            cov_prior= cov_prior,
            n_chains= n_chains,
            domain_bounds=(0.,5.),
            sigma_noise=sigma_noise, 
            number_data=n_data,
            sigma=sigma,
            rwmh_scaling=rwmh_scaling, 
            force_sequential=True,
            levels=1
        )
# hpo example
    # mean_prior = np.array([9.])
    # cov_prior = np.diag([1.])

    # sigma_noise = (0.05,0.15)
    # n_data = (30,50)
    # parameter = np.array([15.])
    # sigma_bounds = ( .5,1.5)

    # rwmh_scaling =  (0.5,1.5)
    # rwmh_cov = np.eye(1)
    # rwmh_adaptive = True
    # iterations = 2000
    # burnin = 1000
    # n_chains = 2
    # algo = "MH_tiny"

    # module_directory = os.path.abspath(os.path.join('..', 'utils'))

    # # Add the 'utils' directory containing Structure.py to the PYTHONPATH
    # os.environ['PYTHONPATH'] = module_directory
    # sys.path.append(module_directory)

    # BIP=BayesianInverseProblem_NN(algorithm_name=algo,forward_NN=final_model)


    # BIP.hpo(inputs_HF=data["datatest"], 
    #     domain_bounds=(0.,5.),
    #     mean_prior=mean_prior, 
    #     cov_prior=cov_prior, 
    #     output_HF=data["modified_highfid"](data["datatest"][:,0],data["datatest"][:,1]), 
    #     sigma_noise=sigma_noise, 
    #     number_data=n_data,
    #     real_parameters=parameter, 
    #     sigma=sigma_bounds,
    #     rwmh_scaling=rwmh_scaling, 
    #     rwmh_covariance=rwmh_cov, 
    #     rwmh_adaptive=rwmh_adaptive, 
    #     iterations=iterations, 
    #     burn_in=burnin, 
    #     n_chains= n_chains, 
    #     levels=1,  
    #     force_sequential=True,
    #     initial_points_optimizer=6,
    #     iterations_optimizer=10,
    #     subsampling_rate=1
    #     )


if __name__ == "__main__":
    main_function()