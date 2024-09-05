from utils.functions_to_ray import  *
######### LIBRARIES ############
import numpy as np


import os
import keras
import tensorflow as tf
import sys
import numpy as np

import sys
import os
import tensorflow as tf

from utils. MOD_helper import *
from utils.network_utils import NetworkConfig,NetworkFactory
from utils.bayesian_utils import BayesianInverseProblem_NN
from source.Reaction_diffusion_class import ReactionDiffusionData




def main_function():


    seed = 7

    data_diffusion = {
        "tlf_0":0.,
        "thf_0":0.,
        "Tlf":80.,
        "Thf":40.,
        "dt":0.05,
        "mu_0":0.5,
        "mu_1":1.5,
        "N_mu_train":10}

    data = ReactionDiffusionData(path='test/Reaction_diffusion/data/reaction_diffusion/',**data_diffusion)
    data.visualize_train_data()
    data.interpolate_data()



    n_POD=9
    reaction_rom=ROM(data,64)
    reaction_rom.compute_POD_basis(64)
    reaction_rom.plot_singular_values_threshold(n_POD)

    ulf_train,uhf_train=reaction_rom.project_onto_POD_train(n_POD)

    data.plot_POD_coefficients(ulf_train, uhf_train, n_POD)

    scaling = True

    if scaling:
        scale = np.max(ulf_train) 
        scale_param = 1.
    else:
        scale =  1.
        scale_param = 1.

    ulf_test,uhf_test = reaction_rom.project_onto_POD_test(n_POD)
    
    #train
    t_train_lstm = np.tile(data.t_lf, data_diffusion["N_mu_train"]).T.reshape(data_diffusion["N_mu_train"],-1,1)
    mu_train_lstm = np.repeat(data.mu_train, data.Nt_train).reshape(data_diffusion["N_mu_train"],-1,1) / scale_param
    ulf_train_lstm = ulf_train / scale
    uhf_train_lstm = uhf_train / scale

    #test
    t_test_lstm = np.tile(data.t_hf_test, data.N_mu_test).T.reshape(data.N_mu_test,-1,1)
    mu_test_lstm = np.repeat(data.mu_test, data.Nt_test).reshape(data.N_mu_test,-1,1) / scale_param
    ulf_test_lstm = ulf_test / scale
    uhf_test_lstm = uhf_test / scale

    #Multi-fidelity network input
    #concatenate in a single input: (time, param, LF data)
    train_mf_lstm = np.concatenate((t_train_lstm, mu_train_lstm, ulf_train_lstm), axis=2)
    test_mf_lstm = np.concatenate((t_test_lstm, mu_test_lstm, ulf_test_lstm), axis=2)

    grid1, grid2 = np.meshgrid(data.t_lf,data.mu_train)
    input_train = np.column_stack((grid2.ravel(), grid1.ravel()))
    input_train=input_train[:,[1,0]]

    model = NetworkFactory.build_network(NetworkConfig(network_type="LSTM"),data.getModel)


    train=False

    if train is True:

        params_LSTM = {'batch': 37, 
                    'lay': 1, 
                    'nodes': 69, 
                    'lr':  0.009476420915650432, 
                    'lay_dense': 0, 
                    'nodes_dense': 0, 
                    'opt': 'Adamax', 
                    'sequence_freq': 7, 
                    'sequence_length': 133,
                    'patience' : 200, 
                    'dropout' : 0.3}


        definition_LSTM = {
            "network_type": "LSTM",
            "network_parameters": params_LSTM,
            "dataset_train": train_mf_lstm,
            "output_train": uhf_train_lstm,
            "epochs_number": 3000,
            "train": True,
            "do_HPO": False,
            "verbose": False
        }
        model = NetworkFactory.build_network(NetworkConfig(**definition_LSTM),data.getModel)

    else:

        model.load("test/Reaction_diffusion/fwd_models/model_save_react.keras")



    seed = 10
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


    data.set_n_POD(9)

    mean_prior = np.array([1.])
    cov_prior = np.diag([.5])

    sigma_noise = 0.001
    n_data_bounds = 150
    parameter = np.array([np.unique(input_train[:,1])[0]])
    sigma_bounds = 0.7


    rwmh_scaling_bounds =  1.5443
    rwmh_cov = 0.96
    rwmh_adaptive = True
    iterations = 3000
    burnin = 2000
    n_chains = 2
    algo = "MH_tiny"

    module_directory = os.path.abspath(os.path.join('..', 'utils'))

    # Add the 'utils' directory containing Structure.py to the PYTHONPATH
    os.environ['PYTHONPATH'] = module_directory
    sys.path.append(module_directory)
    BIP=BayesianInverseProblem_NN(algorithm_name=algo,forward_NN=model)

    BIP.run(inputs_HF=input_train, 
        domain_bounds=(0.,2.),
        mean_prior=mean_prior, 
        cov_prior=cov_prior, 
        output_HF=uhf_train_lstm.reshape(-1, 9), 
        real_parameters=parameter, 
        rwmh_adaptive=rwmh_adaptive, 
        iterations=iterations, 
        burn_in=burnin, 
        n_chains= n_chains, 
        levels=1,  
        force_sequential=True,
        rwmh_scaling=rwmh_scaling_bounds,
        rwmh_covariance=rwmh_cov,
        sigma_noise=sigma_noise,
        number_data=n_data_bounds,
        sigma=sigma_bounds,
        fwd_LSTM_folder="test/Reaction_diffusion/fwd_models/model_",
        forward_low_fidelity= data._forward_low_fidelity,
    )


    seed = 10
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


    parameter = np.array([np.unique(input_train[:,1])[-2]])

    
    BIP.run(inputs_HF=input_train, 
        domain_bounds=(0.,2.),
        mean_prior=mean_prior, 
        cov_prior=cov_prior, 
        output_HF=uhf_train_lstm.reshape(-1, 9), 
        real_parameters=parameter, 
        rwmh_adaptive=rwmh_adaptive, 
        iterations=iterations, 
        burn_in=burnin, 
        n_chains= n_chains, 
        levels=1,  
        force_sequential=True,
        rwmh_scaling=rwmh_scaling_bounds,
        rwmh_covariance=rwmh_cov,
        sigma_noise=sigma_noise,
        number_data=n_data_bounds,
        sigma=sigma_bounds,
        fwd_LSTM_folder="test/Reaction_diffusion/fwd_models/model_",
        forward_low_fidelity= data._forward_low_fidelity,
    )


if __name__ == "__main__":
    main_function()