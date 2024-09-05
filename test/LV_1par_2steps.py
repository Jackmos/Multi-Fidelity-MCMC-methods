import numpy as np
from matplotlib import pyplot as plt
import os
import keras
import tensorflow as tf
import sys
from utils.network_utils import *
from utils.bayesian_utils import BayesianInverseProblem_NN
from source.System_solver import System1



# introduction of a seed for reproducibility purposes
def set_seed():
    seed = 10
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

def main_function():

    seed = 7
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

    ### DATASET CREATION 

    solver_HF = System1()
    solver_LF = System1()

    mu_values = np.linspace(1, 5, 9)   # parameter range
    params = [(mu,) for mu in mu_values]

    (t_HF,y_HF,mu_HF)=solver_HF.generate_dataset(T=8.0, params=np.array(params), h=0.02, fidelity='HF')
    #solver_HF.save_dataset('dataset_HF.h5')

    (t_LF,y_LF,mu_LF)=solver_LF.generate_dataset(T=8.0, params=np.array(params), h=0.05, fidelity='LF')
    #solver_LF.save_dataset('dataset_LF.h5')

    # validation set definition
    solver_valHF = System1()
    solver_valLF = System1()
    mu_values_val = np.linspace(1, 5, 6)
    params_val = [(mu,) for mu in mu_values_val]

    (t_valhf,y_valhf,mu_valhf)=solver_valHF.generate_dataset(T=8.0, params=np.array(params_val), h=0.02, fidelity='HF')
    solver_valHF.save_dataset('dataset_HF.h5')

    (t_valLF,y_valLF,mu_valLF)=solver_valLF.generate_dataset(T=8.0, params=np.array(params_val), h=0.05, fidelity='LF')
    solver_valLF.save_dataset('dataset_LF.h5')

    # plot
    solver_HF.plot_dataset()

    solver_LF.plot_dataset()


    # manipulation to give proper shape
    x_HF=np.array(np.meshgrid(mu_HF,t_HF)).T.reshape(-1, 2)[:,[1,0]]
    x_LF=np.array(np.meshgrid(mu_LF,t_LF)).T.reshape(-1, 2)[:,[1,0]]
    x_valHF=np.array(np.meshgrid(mu_valhf,t_valhf)).T.reshape(-1, 2)[:,[1,0]]
    x_valLF=np.array(np.meshgrid(mu_valLF,t_valLF)).T.reshape(-1, 2)[:,[1,0]]


    # from list to array
    y_HF=np.array(y_HF).reshape(-1, 3)
    y_LF=np.array(y_LF).reshape(-1, 3)
    y_valHF=np.array(y_valhf).reshape(-1, 3)
    y_valLF=np.array(y_valLF).reshape(-1, 3)

    # selection data for training
    n_LF=1000
    indices_LF = np.random.permutation(x_LF.shape[0])[:n_LF]
    n_HF=2000
    indices_HF = np.random.permutation(x_HF.shape[0])[:n_HF]
    n_valLF=800
    indices_valLF = np.random.permutation(x_valLF.shape[0])[:n_valLF]
    n_valHF=1600
    indices_valHF = np.random.permutation(x_valHF.shape[0])[:n_valHF]

    # NETWORK DEFINITION
    N=[400,1000]
    n=[200,250]
    names = ['LF', 'HF']

    # parameters 
    bestLF_params =  {'kernel_init': 'uniform',
    'l2weight': 0.0015902185943731835,
    'lr': 0.010824546558751915,
    'nodes': 7,
    'opt': 'Adam'}
    best_params={'kernel_init': 'glorot_uniform',
    'l2weight': 0.00012541532994939157,
    'lr': 0.02007377317859368,
    'nodes': 29,
    'opt': 'Adam'}

    params_NN=[bestLF_params, best_params]


    # collection of the parameters
    definition_2steps = {
        "network_type": "2step",
        "names": names,
        "network_parameters": params_NN,
        "dataset_train": [x_LF[indices_LF,:],x_HF[indices_HF,:]],
        "output_train": [y_LF[indices_LF,:], y_HF[indices_HF,:]],
        "dataset_validation": [x_valLF[indices_valLF,:],x_valHF[indices_valHF,:]],
        "output_validation": [y_valLF[indices_valLF,:], y_valHF[indices_valHF,:]],    
        "epochs_number": N,
        "batch_size": n,
        "train": True,
        "do_HPO": False,
        "verbose": False
    }

    # build NN
    final_model_trained = NetworkFactory.build_network(NetworkConfig(**definition_2steps),solver_HF.getModel)
    
    # creation test set
    solver_test=System1()
    mu_values_test = np.linspace(1, 5, 15)
    params_test = [(mu,) for mu in mu_values_test]
    (t_test,y_test,mu_test)=solver_test.generate_dataset(T=8.0, params=np.array(params_test), h=0.005, fidelity='HF')
    solver_test.save_dataset('dataset_test.h5')

    datatest=np.array(np.meshgrid(mu_test, t_test)).T.reshape(-1, 2)[:,[1,0]]


    (mse_MF,R_MF)=final_model_trained.performance(datatest,np.array(y_test).reshape(-1,3))




#################################plt_per_parameter (LF) *2
#################################plt_per_parameter (HF) *2
    cuqi_example=False
    DA_comp_example=True

    if cuqi_example is True:
        set_seed()

        mean_prior = np.array([2.5])
        cov_prior = np.diag([1.])

        sigma_noise = 0.09649
        n_data= 31
        parameters = np.array([1.])
        sigma = .5
        rwmh_scaling =  0.5
        rwmh_cov = 1
        rwmh_adaptive = True
        iterations = 2000
        burnin = 1250
        number_chains = 2
        algo = "MH_cuqi"
        proposal_sd = 0.6484    # più piccolo, altrimenti scappa da previsione 
        scaling = 1.
        adapt=True
        BB=BayesianInverseProblem_NN(algorithm_name=algo,forward_NN=final_model_trained)



        a=BB.run(
            inputs_HF=x_HF, 
            output_HF=y_HF, 
            mean_prior= mean_prior,
            real_parameters=parameters,
            iterations= iterations,
            burn_in= burnin,
            cov_prior= cov_prior,
            sd_noise=sigma_noise,
            proposal_sd= proposal_sd,
            num_data=n_data,
            scale= scaling,
            adapt=adapt,
            number_chains= number_chains,
            domain_bounds=(0.,8.),

            parallel=False
        )
    

    if DA_comp_example is True:

        set_seed()

        mean_prior = np.array([2.5])
        cov_prior = np.diag([1.])

        sigma_noise = 0.3031          #0.2955
        n_data = 30                  #46
        parameters = np.array([1.])
        sigma = 3.497                #1.637
        rwmh_scaling = 0.5
        rwmh_cov =1.955
        rwmh_adaptive = True
        iterations =2000
        burnin = 1000
        n_chains = 2
        subsampling_rate=2                  #2
        algo = "AM"
        module_directory = os.path.abspath(os.path.join('..', 'utils'))

        os.environ['PYTHONPATH'] = module_directory
        sys.path.append(module_directory)
        BB=BayesianInverseProblem_NN(algorithm_name=algo,forward_NN=final_model_trained)

        # Delayed acceptance, 2 levels
        BB.run(inputs_HF=x_HF, 
            domain_bounds=(0.,8.),
            mean_prior=mean_prior, 
            cov_prior=cov_prior, 
            output_HF=y_HF, 
            sigma_noise=sigma_noise, 
            number_data=n_data,
            real_parameters=parameters, 
            sigma=sigma,
            rwmh_scaling=rwmh_scaling, 
            rwmh_covariance=rwmh_cov, 
            rwmh_adaptive=rwmh_adaptive, 
            iterations=iterations, 
            burn_in=burnin, 
            n_chains= n_chains, 
            levels=2,  
            subsampling_rate=subsampling_rate,
            force_sequential=True,

            )


        set_seed()

        # single level 
        BB.run(inputs_HF=x_HF, 
            domain_bounds=(0.,8.),
            mean_prior=mean_prior, 
            cov_prior=cov_prior, 
            output_HF=y_HF, 
            sigma_noise=sigma_noise, 
            number_data=n_data,
            real_parameters=parameters, 
            sigma=sigma,
            rwmh_scaling=rwmh_scaling, 
            rwmh_covariance=rwmh_cov, 
            rwmh_adaptive=rwmh_adaptive, 
            iterations=iterations, 
            burn_in=burnin, 
            n_chains= n_chains, 
            levels=1,  
            subsampling_rate=1,
            force_sequential=True,

            )



if __name__ == "__main__":
    main_function()