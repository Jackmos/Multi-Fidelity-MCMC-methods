# Example III: MLDA, 3 step case
import numpy as np
import keras
import tensorflow as tf
import itertools
from utils.network_utils import NetworkConfig,NetworkFactory
from utils.bayesian_utils import BayesianInverseProblem_NN
from source.System_solver import System2




# introduction of a seed for reproducibility purposes
def set_seed():
    seed = 10
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

# Main function
def main_function():

    set_seed()
    ### DATASET CREATION 
    # Define the intervals and number of points
    num_points = 5  # Number of points in each linspace (adjust as needed)

    first_interval = np.linspace(2.1, 3, num_points)        # for parameter a
    second_interval = np.linspace(0.9, 1.5, num_points)     # for parameter b
    third_interval = np.linspace(0.9, 1.5, num_points)      # for parameter c
    fourth_interval = np.linspace(1., 2.1, num_points)      # for parameter d

    # Generate all combinations using itertools.product
    combinations = list(itertools.product(first_interval, second_interval, third_interval, fourth_interval))

    # Convert the combinations to a numpy array
    combinations_array = np.array(combinations)
    np.random.shuffle(combinations_array)
    combinations_array=combinations_array[0:30,:]

    # The same for the training set
    # Shuffle the rows of the array
    np.random.shuffle(combinations_array)
    params=combinations_array[0:9,:]


    solver = System2()       

    (t_HF,y_HF,mu_HF)=solver.generate_dataset(T=7.0, params=params, h=0.01, fidelity='HF')
    solver.plot_dataset()

    (t_LF,y_LF,mu_LF)=solver.generate_dataset(T=7.0, params=params, h=0.04, fidelity='LF')   
    solver.plot_dataset()

    (t_valHF,y_valHF,mu_valHF)=solver.generate_dataset(T=7.0, params=params, h=0.01, fidelity='HF')
    (t_valLF,y_valLF,mu_valLF)=solver.generate_dataset(T=7.0, params=params, h=0.01, fidelity='LF')   




    # Create the Cartesian product of the indices
    cartesian_product_indices = np.array(np.meshgrid(np.arange(mu_HF.shape[0]), np.arange(t_HF.shape[0]))).T.reshape(-1, 2)
    # Extract the respective elements from the original vectors
    x_HF = np.hstack((t_HF[cartesian_product_indices[:, 1]].reshape(-1, 1), mu_HF[cartesian_product_indices[:, 0]]))


    # same for LF dataset
    cartesian_product_indices = np.array(np.meshgrid(np.arange(mu_LF.shape[0]), np.arange(t_LF.shape[0]))).T.reshape(-1, 2)
    x_LF = np.hstack((t_LF[cartesian_product_indices[:, 1]].reshape(-1, 1), mu_LF[cartesian_product_indices[:, 0]]))


    # repeat for the validation set
    cartesian_product_indices_val = np.array(np.meshgrid(np.arange(mu_valHF.shape[0]), np.arange(t_valHF.shape[0]))).T.reshape(-1, 2)
    x_valHF = np.hstack((t_valHF[cartesian_product_indices_val[:, 1]].reshape(-1, 1), mu_valHF[cartesian_product_indices_val[:, 0]]))


    cartesian_product_indices_val = np.array(np.meshgrid(np.arange(mu_valLF.shape[0]), np.arange(t_valLF.shape[0]))).T.reshape(-1, 2)
    x_valLF = np.hstack((t_valLF[cartesian_product_indices_val[:, 1]].reshape(-1, 1), mu_valLF[cartesian_product_indices_val[:, 0]]))


    y_HF=np.array(y_HF).reshape(-1, solver.get_dim())
    y_LF=np.array(y_LF).reshape(-1, solver.get_dim())

    y_valHF=np.array(y_valHF).reshape(-1, solver.get_dim())
    y_valLF=np.array(y_valLF).reshape(-1, solver.get_dim())



    # selection data for training and validation
    n_LF=700
    indices_LF = np.random.permutation(x_LF.shape[0])[:n_LF]
    n_Single=1500
    indices_Single = np.random.permutation(x_HF.shape[0])[:n_Single]
    n_HF=2000
    indices_HF = np.random.permutation(x_HF.shape[0])[:n_HF]


    n_valLF=500
    indices_valLF = np.random.permutation(x_valLF.shape[0])[:n_valLF]
    n_valSingle=1250
    indices_valSingle = np.random.permutation(x_valHF.shape[0])[:n_valSingle]
    n_valHF=1700
    indices_valHF = np.random.permutation(x_valHF.shape[0])[:n_valHF]



    ### Initial definition of Neural Network's parameters
    #   3 steps

    N=[1000,500,1000]   # epochs
    n=[75,150,200]      # batch size

    names = ['LF','Single', 'HF']

    # parameters 
    bestLF_params = {'kernel_init': 'glorot_uniform',
    'l2weight': 0.002086252170990476,
    'lr': 0.0006684072410663945,
    'nodes': 19,
    'opt': 'Adam'}
    best_params_single={'kernel_init': 'glorot_uniform',
    'l2weight': 0.0003560479678957748,
    'lr': 0.0067446665215358595,
    'nodes': 6,
    'opt': 'Adam'}

    best_params={'kernel_init': 'glorot_uniform',
    'l2weight': 0.00014877179504142786,
    'lr': 0.03518287483245958,
    'nodes': 18,
    'opt': 'Adam'}

    params_NN=[bestLF_params,best_params_single, best_params]

    # collection of the parameters
    definition_2steps = {
        "network_type": "3step",
        "names": names,
        "network_parameters": params_NN,
        "dataset_train": [x_LF[indices_LF,:],x_HF[indices_Single,:],x_HF[indices_HF,:]],
        "output_train": [y_LF[indices_LF,:],  y_HF[indices_Single,:],y_HF[indices_HF,:]],
        "dataset_validation": [x_valLF[indices_valLF,:],x_valHF[indices_valSingle,:],x_valHF[indices_valHF,:]],
        "output_validation": [y_valLF[indices_valLF,:],  y_valHF[indices_valSingle,:],y_valHF[indices_valHF,:]],
        "epochs_number": N,
        "batch_size": n,
        "train": True,
        "do_HPO": False,
        "verbose": False
        }


    final_model=NetworkFactory.build_network(NetworkConfig(**definition_2steps),solver.getModel)



    ### CREATION OF THE TEST SET
    solver_test=System2()
    params_values_test = combinations_array[0:20,:]
    (t_test,y_test,mu_test)=solver_test.generate_dataset(T=7.0, params=params_values_test, h=0.005, fidelity='HF')


    # Create the Cartesian product of the indices
    cartesian_product_indices = np.array(np.meshgrid(np.arange(mu_test.shape[0]), np.arange(t_test.shape[0]))).T.reshape(-1, 2)
    datatest = np.hstack((t_test[cartesian_product_indices[:, 1]].reshape(-1, 1), mu_test[cartesian_product_indices[:, 0]]))

    # Neural Network test
    y_test_pred=final_model.prediction(datatest)
    (mse_MF,R_MF)=final_model.performance(datatest,np.array(y_test).reshape(-1,solver_test.get_dim()))


    solver.save_plots_per_param(x_HF,y_HF,y_test_pred, final_model)   

    ### BAYESIAN INVERSE PROBLEM ###
 

    set_seed()


    mean_prior =np.array([1.5,1.,1.5,0.6])
    cov_prior=np.diag([ 0.1, 0.01, 0.01, 0.1])

    parameters =np.array([2.325,1.2,1.35,1.])
    rwmh_adaptive = True
    iterations = 2000
    burnin = 1000
    n_chains = 3
    algo = "AM"

    BB=BayesianInverseProblem_NN(algorithm_name=algo,forward_NN=final_model)
    # MLDA
    BB.run(inputs_HF=x_HF, 
        domain_bounds=(0.,7.),
        mean_prior=mean_prior, 
        cov_prior=cov_prior, 
        output_HF=y_HF, 
        real_parameters=parameters, 
        rwmh_adaptive=rwmh_adaptive, 
        iterations=iterations, 
        burn_in=burnin, 
        n_chains= n_chains, 
        levels=3,  
        subsampling_rate=3,
        force_sequential=True,
        number_data=48,
        rwmh_covariance=1.562, 
        sigma=2.062,
        rwmh_scaling =  1.031,
        sigma_noise=0.0985  
        
    )

    set_seed()

    BBB=BayesianInverseProblem_NN(algorithm_name='AM',forward_NN=final_model)

    # AM
    BBB.run(inputs_HF=x_HF, 
        domain_bounds=(0.,7.),
        mean_prior=mean_prior, 
        cov_prior=cov_prior, 
        output_HF=y_HF, 
        real_parameters=parameters, 
        rwmh_adaptive=rwmh_adaptive, 
        iterations=iterations, 
        burn_in=burnin, 
        n_chains= n_chains, 
        levels=1,  
        subsampling_rate=1,
        force_sequential=True,
        number_data=48,
        rwmh_covariance=1.562, 
        sigma=2.062,
        rwmh_scaling =  1.031,
        sigma_noise=0.0985   
        
    )



    set_seed()

    BBB=BayesianInverseProblem_NN(algorithm_name='MH_tiny',forward_NN=final_model)

    # RWMH
    BBB.run(inputs_HF=x_HF, 
        domain_bounds=(0.,7.),
        mean_prior=mean_prior, 
        cov_prior=cov_prior, 
        output_HF=y_HF, 
        real_parameters=parameters, 
        rwmh_adaptive=rwmh_adaptive, 
        iterations=iterations, 
        burn_in=burnin, 
        n_chains= n_chains, 
        levels=1,  
        subsampling_rate=1,
        force_sequential=True,
        number_data=48,
        rwmh_covariance=1.562, 
        sigma=2.062,
        rwmh_scaling =  1.031,
        sigma_noise=0.0985  
        
    )
if __name__ == "__main__":
    main_function()