# Example IV: LSTM and Burger
import tensorflow as tf
import numpy as np
import keras
from utils.network_utils import NetworkConfig, NetworkFactory
from utils.bayesian_utils import BayesianInverseProblem_NN
from source.Burger_class import BurgerEquation
from utils.MOD_helper import *

def set_seed():
    seed = 29
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    


def main_function():

    seed=29

    burger_eq = BurgerEquation(nh = 101, nt = 151, T= 2.0, L= 1.0, nre= 20, re_min = 80, re_max= 500, seed=seed)
    burger_eq.generate_data()
    burger_eq.plt_time_instants()
    burger_eq.plot_data()


    n_POD=16

    rom_burger = ROM(burger_eq, n_POD=n_POD)


    _,_, S=rom_burger.compute_POD_basis(16)

    _,_=rom_burger.project_onto_POD_train(n_POD)
    _,_=rom_burger.project_onto_POD_test(n_POD)


    input_train, output_train, input_test, output_test=burger_eq.definitionLSTM_dataset()

    rom_burger.plot_POD_coefficients(ind_re=0)


    params_NN = { 'lay': 1,
                'nodes': 64,
                'lr':  1e-3,
                'opt': 'Adamax',
                'sequence_freq': 2,
                'sequence_length': 10,
                'patience' : 50,
                'dropout' : 0.1}


    definition_LSTM = {
        "network_type": "LSTM",
        "network_parameters": params_NN,
        "dataset_train": input_train,
        "output_train": output_train,
        "epochs_number": 500,
        "train": True,
        "do_HPO": False,
        "verbose": False
    }

    set_seed()
    model = NetworkFactory.build_network(NetworkConfig(**definition_LSTM), burger_eq.getModel)

    output_pred = model.prediction(input_test)

    model.set_parameters(params_NN)



    # plot of the estimated POD coeff
    ind_re = [0]
    rom_burger.plot_POD_output(ind_re=ind_re, output_pred=output_pred)


    # comparison of different fidelity-degree
    burger_eq.plot_comparison(output_pred, rom_burger.get_basis(), ind_re)

    #error plot
    burger_eq.plot_error(output_pred, rom_burger.get_basis(), ind_re)

    set_seed()

    ### BAYESIAN INVERSION 
    # updated number of relelvant modes and basis for the forward model
    burger_eq.set_POD(16,rom_burger.get_basis())

    mean_prior =np.array([.5])
    cov_prior=np.diag([ .1])

    index_re=0
    parameters =np.array([burger_eq.re_grid_lstm_test[index_re,0]])
    rwmh_adaptive = True
    iterations = 3000
    burnin = 2000
    n_chains = 4
    algo = "MH_tiny"


    BB=BayesianInverseProblem_NN(algorithm_name=algo,forward_NN=model)


    BB.run(input_support=burger_eq.x.reshape(-1,1),                 # additional input. x is necessary to evaluate u_LF
        inputs_HF=input_test[:,:,[0,1]].reshape(-1,2),
        domain_bounds=(0.,2.),
        mean_prior=mean_prior, 
        cov_prior=cov_prior, 
        output_HF=output_test[index_re,], 
        real_parameters=parameters, 
        rwmh_adaptive=rwmh_adaptive, 
        iterations=iterations, 
        burn_in=burnin, 
        n_chains= n_chains, 
        number_data=83,
        rwmh_scaling=0.5086,
        sigma=0.3,
        sigma_noise=0.05,
        levels=1,  
        subsampling_rate=1,
        force_sequential=True,
        forward_low_fidelity= burger_eq._forward_low_fidelity       # additional function introduced for the inverse model.
                                                                    # It allows the inverse methods to have acces and use POD basis 
    )




if __name__ == "__main__":
    main_function()