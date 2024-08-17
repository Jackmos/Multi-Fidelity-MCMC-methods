import keras
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import sys
from module_utils import * 
from data_collection import *
sys.path.append('../utils')
from Structure import *
import os 
from pathlib import Path

# path to the current notebook
current_file_path = Path().resolve()
# path to the current folder
load_context_functions(current_file_path.parent.name)


def main():

    # reproducibility
    seed = 7
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)

    ## Dataset definition

    solver_HF = System1()
    solver_LF = System1()

    mu_values = np.linspace(1, 5, 10)
    params = [(mu,) for mu in mu_values]

    (t_HF,y_HF,mu_HF)=solver_HF.generate_dataset(T=8.0, params=np.array(params), h=0.02, fidelity='HF')
    solver_HF.save_dataset('dataset_HF.h5')

    (t_LF,y_LF,mu_LF)=solver_LF.generate_dataset(T=8.0, params=np.array(params), h=0.05, fidelity='LF')
    solver_LF.save_dataset('dataset_LF.h5')

    # plotting of the solution
    solver_HF.plot_results()
    solver_LF.plot_results()

    # input definition
    x_HF=np.array(np.meshgrid(mu_HF,t_HF)).T.reshape(-1, 2)[:,[1,0]]
    x_LF=np.array(np.meshgrid(mu_LF,t_LF)).T.reshape(-1, 2)[:,[1,0]]
    # output definition
    y_HF=np.array(y_HF).reshape(-1, 3)
    y_LF=np.array(y_LF).reshape(-1, 3)

    # parameters introduction
    n_LF=1000
    indices_LF = np.random.permutation(x_LF.shape[0])[:n_LF]

    n_HF=2000
    indices_HF = np.random.permutation(x_HF.shape[0])[:n_HF]
    N=[200,1000]
    n=[150,200]
    names = ['LF', 'HF']

    # test set definition
    solver_test=System1()
    mu_values_test = np.linspace(1, 5, 15)
    params_test = [(mu,) for mu in mu_values_test]
    (t_test,y_test,mu_test)=solver_test.generate_dataset(T=8.0, params=np.array(params), h=0.005, fidelity='HF')
    solver_test.save_dataset('dataset_test.h5')
    datatest=np.array(np.meshgrid(mu_test, t_test)).T.reshape(-1, 2)[:,[1,0]]

    ## Neural Network creation

    # parameters 
    bestLF_params = {
        "lr": 0.0255,
        "kernel_init": "glorot_uniform",
        "opt": "Adam",
    } 
    best_params={
                 'kernel_init': 'uniform', 
                 'l2weight': 0.005395541657336886, 
                 'lr': 0.0025412188591473228, 
                 'nodes': 12.0, 
                 'opt': 'Adamax'
                 }

    params_NN=[bestLF_params, best_params]

    definition_2steps = {
                        "network_type": "2step",
                        "names": names,
                        "params": params_NN,
                        "data_train": [x_LF,x_HF],
                        "output_train": [y_LF, y_HF],
                        "N": N,
                        "n": n,
                        "train": True,
                        "do_HPO": False,            # choose if do Hyperparameter optimization 
                        "verbose": False
                        }
    
    final_model = NetworkFactory.build_network(**definition_2steps)
    y_test_pred=final_model.prediction(datatest)

    (mse_MF,R_MF)=final_model.performance(datatest,np.array(y_test).reshape(-1,3))

    # Plotting results
    components = ['Component 1', 'Component 2', 'Component 3']
    colors_training = ['k', 'b', 'g']  # Nero, Blu scuro, Verde scuro
    colors_pred = ['r', 'orange', 'purple']  # Rosso, Arancione, Viola

    plt.figure(figsize=(18, 6))

    # Loop through each component to create a separate plot
    for i in range(3):
        plt.subplot(1, 3, i + 1)  # Create a subplot (1 row, 3 columns)
        
        # Plotting the training data for the i-th component
        plt.plot(x_HF[:, 0], y_HF[:, i], f'{colors_training[i]}.', label=f'HF training data {i + 1}', markersize=7)

        # Plotting the prediction data for the i-th component
        plt.plot(datatest[:, 0], y_test_pred[:, i], '.', 
                color=colors_pred[i], alpha=0.2, label=f'Pred HF {i + 1}', linewidth=3)
        
        # Adding grid, labels, and title
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Response', fontsize=12)
        plt.title(f'High Fidelity Model: {components[i]}', fontsize=14)
        
        # Add a legend
        plt.legend(loc='best', fontsize='small')

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()


    ## Bayesian inverse problem
    # tinyDA version
    # data
    mean_prior = np.array([2.5])
    cov_prior = np.diag([1.])

    sigma_noise = [0.25,0.5]
    n_data = [50,30]
    parameters = np.array([5.])
    sigma = np.array([ 4.,2.])
    rwmh_scaling = np.array([ 2.,0.5,1.])
    rwmh_cov = np.eye(1)
    rwmh_adaptive = True
    iterations = 300
    burnin = 200
    n_chains = 2
    algo = "MH"

    # Add the 'utils' directory containing Structure.py to the PYTHONPATH
    module_directory = os.path.abspath(os.path.join('..', 'utils'))

    os.environ['PYTHONPATH'] = module_directory
    sys.path.append(module_directory)

    estimateMF, errorMF, param_resultsMF= run_simulation(x_HF, 
                                                         mean_prior, 
                                                         cov_prior, 
                                                         y_HF, 
                                                         sigma_noise,
                                                         n_data, 
                                                         parameters, 
                                                         sigma, 
                                                         rwmh_scaling, 
                                                         rwmh_cov, 
                                                         rwmh_adaptive, 
                                                         iterations, 
                                                         burnin, 
                                                         n_chains, 
                                                         final_model,
                                                         algo
                                                         )
    


