import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
import tensorflow as tf
import arviz
import logging
from concurrent.futures import ProcessPoolExecutor

from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
import sys
import os
import warnings
import ray
from numba import njit

from scipy.stats import multivariate_normal,beta
import arviz as az
import time 

from itertools import product
from typing import List, Tuple, Union

# import numpy as np
# from joblib import Parallel, delayed, Memory
# from typing import List, Tuple

# def calculate_cov_likelihood(sigma: float, t_eval: np.ndarray) -> np.ndarray:
#     """
#     Calculate the covariance matrix for the likelihood.

#     Parameters:
#     - sigma: Standard deviation for the likelihood.
#     - t_eval: 2D numpy array of evaluation times.

#     Returns:
#     - cov_likelihood: 2D numpy array representing the covariance matrix.
#     """
#     # Assuming a Gaussian process covariance matrix as an example
#     cov_likelihood = np.exp(-0.5 * (t_eval - t_eval.T)**2 / sigma**2)
#     return cov_likelihood

# def process_data(datahf: np.ndarray, parameters: np.ndarray, t_eval: np.ndarray, Yhf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Find the elements of a dataset nearest to the ones given.
    
#     Parameters:
#     - datahf: 2D numpy array where datahf[:,1] contains parameter values.
#     - parameters: 1D numpy array of parameter values to find in datahf.
#     - t_eval: 2D numpy array of evaluation times.
#     - Yhf: 1D numpy array of corresponding y values.
    
#     Returns:
#     - nearest_x: 1D numpy array of x values closest to each t_eval.
#     - y_obs: 1D numpy array of corresponding y values from Yhf.
#     """
#     # Find indices of datahf where the parameter matches the target parameter
#     indices = np.where(datahf[:, 1] == parameters[0])[0]
#     if len(indices) == 0:
#         raise ValueError(f"No observations related to parameter: {parameters[0]}")
    
#     # Extract relevant data and compute differences to t_eval
#     datahf_values = datahf[indices, 0].reshape(-1, 1)
#     differences = np.abs(datahf_values - t_eval.reshape(1, -1))
    
#     # Find the index of the closest data points
#     closest_indices = np.argmin(differences, axis=0)
#     nearest_x = datahf[indices[closest_indices], 0]
#     y_obs = Yhf[indices[closest_indices]]
    
#     return nearest_x, y_obs

# def run_single_simulation(
#     i: int, 
#     k: int, 
#     t: int, 
#     j: int, 
#     noise: float, 
#     n: int, 
#     s: float, 
#     r: float, 
#     datahf: np.ndarray, 
#     parameters: np.ndarray, 
#     Yhf: np.ndarray, 
#     mean_prior: np.ndarray, 
#     cov_prior: np.ndarray, 
#     rwmh_cov: np.ndarray, 
#     rwmh_adaptive: bool, 
#     iterations: int, 
#     burnin: int, 
#     n_chains: int, 
#     final_model, 
#     algo: str, 
#     levels: int, 
#     force_sequential: bool
# ) -> Tuple[np.ndarray, float, np.ndarray, int, int, int, int]:
#     """
#     Run a single simulation for a specific combination of parameters.

#     Parameters:
#     - i, k, t, j: Indices of the parameters in the product loop.
#     - noise: Noise level.
#     - n: Number of data points.
#     - s: Standard deviation for the likelihood.
#     - r: Scaling factor for the RWMH algorithm.
#     - datahf: 2D numpy array containing data.
#     - parameters: 1D numpy array of parameters.
#     - Yhf: 1D numpy array of observed values.
#     - mean_prior: 1D numpy array for the mean of the prior.
#     - cov_prior: 2D numpy array for the covariance of the prior.
#     - rwmh_cov: 2D numpy array for the RWMH covariance.
#     - rwmh_adaptive: Boolean indicating if RWMH is adaptive.
#     - iterations: Integer for the number of iterations.
#     - burnin: Integer for the burn-in period.
#     - n_chains: Integer for the number of chains.
#     - final_model: The model object with the param_inverse method.
#     - algo: String indicating the algorithm to use.
#     - levels: Number of levels for ML MCMC approach.
#     - force_sequential: If True, imposes a sequential approach to the MCMC algorithm.

#     Returns:
#     - est: The estimated parameters.
#     - err: The error corresponding to the estimate.
#     - param_res: Additional parameter results.
#     - i, k, t, j: Indices of the parameters in the product loop for identification.
#     """
    
#     # Generate evaluation times
#     t_eval = np.linspace(np.min(datahf[:, 0]), np.max(datahf[:, 0]), n).reshape(-1, 1)
    
#     # Get nearest x values and corresponding observations
#     _, y_obs = process_data(datahf, parameters, t_eval, Yhf)
    
#     # Compute the covariance matrix for the likelihood
#     cov_likelihood = calculate_cov_likelihood(s, t_eval)
    
#     # Perform parameter estimation and calculate error
#     est, err, param_res = final_model.param_inverse(
#         mean_prior, t_eval, max_par=max(datahf[:, 1]), cov_prior=cov_prior, rmwh_scaling=r, 
#         levels=levels, cov_noise=noise, cov_likelihood=cov_likelihood, y_obs=y_obs, 
#         x_real=parameters, number_chains=n_chains, N=iterations, 
#         burn_in=burnin, diagnostic=True, rwmh_cov=rwmh_cov, 
#         rwmh_adaptive=rwmh_adaptive, algo=algo, force_sequential=force_sequential
#     )
    
#     return est, err, param_res, i, k, t, j

# def run_simulation(
#     datahf: np.ndarray, 
#     mean_prior: np.ndarray, 
#     cov_prior: np.ndarray, 
#     Yhf: np.ndarray, 
#     sigma_noise: List[float], 
#     n_data: List[int],
#     parameters: np.ndarray, 
#     sigma: np.ndarray, 
#     rwmh_scaling: np.ndarray, 
#     rwmh_cov: np.ndarray, 
#     rwmh_adaptive: bool, 
#     iterations: int, 
#     burnin: int, 
#     n_chains: int, 
#     final_model, 
#     algo: str, 
#     levels: int = 1,
#     #force_sequential: bool = False
# ) -> Tuple[np.ndarray, float, List[dict]]:
#     """
#     Run a simulation to estimate parameters and calculate errors.
    
#     Returns:
#     - best_estimate: The best parameter estimate.
#     - best_error: The error corresponding to the best estimate.
#     - param_results: List of dictionaries containing parameter results and their associated parameters.
#     """
    
#     # Initialize the shape of the error array
#     error_shape = (len(sigma_noise), len(n_data), len(sigma), len(rwmh_scaling))
    
#     # Run simulations in parallel for all parameter combinations
#     results = Parallel(n_jobs=-1)(
#         delayed(run_single_simulation)(
#             i, k, t, j, noise, n, s, r, datahf, parameters, Yhf, 
#             mean_prior, cov_prior, rwmh_cov, rwmh_adaptive, iterations, 
#             burnin, n_chains, final_model, algo, levels, True
#         )
#         for (i, noise), (k, n), (t, s), (j, r) in product(
#             enumerate(sigma_noise), enumerate(n_data), 
#             enumerate(sigma), enumerate(rwmh_scaling)
#         )
#     )
    
#     # Unpack results
#     estimates, errors, param_results = [], [], []
#     structured_results = []  # To hold structured results for each simulation
    
#     for est, err, param_res, i, k, t, j in results:
#         estimates.append(est)
#         errors.append(err)
#         param_results.append(param_res)
        
#         # Store structured results including parameter values
#         structured_results.append({
#             'estimate': est,
#             'error': err,
#             'parameters': {
#                 'sigma_noise': sigma_noise[i],
#                 'n_data': n_data[k],
#                 'sigma': sigma[t],
#                 'rwmh_scaling': rwmh_scaling[j]
#             },
#             'param_results': param_res
#         })
    
#     # Reshape errors to match the parameter grid
#     errors = np.array(errors).reshape(error_shape)
    
#     # Identify the index of the minimum error
#     smallest_index = np.unravel_index(np.argmin(errors), errors.shape)
#     best_estimate = estimates[smallest_index[0]]
#     best_error = errors[smallest_index]
    
#     # Print the best parameters
#     print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, "
#           f"number of data={n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, "
#           f"rwmh_scaling={rwmh_scaling[smallest_index[3]]}")

#     return best_estimate, best_error, structured_results



@njit
def process_data(datahf: np.ndarray, parameters: np.ndarray, t_eval: np.ndarray, Yhf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the elements of a dataset nearest to the ones given.
    
    Parameters:
    - datahf: 2D numpy array where datahf[:,1] contains parameter values.
    - parameters: 1D numpy array of parameter values to find in datahf.
    - t_eval: 2D numpy array of evaluation times.
    - Yhf: 1D numpy array of corresponding y values.
    
    Returns:
    - nearest_x: 1D numpy array of x values closest to each t_eval.
    - y_obs: 1D numpy array of corresponding y values from Yhf.
    """
    indices = np.where(datahf[:, 1] == parameters[0])[0]
    if len(indices) == 0:
        raise ValueError(f"No observations related to parameter: {parameters[0]}")
    
    datahf_values = datahf[indices, 0].reshape(-1, 1)
    t_eval_values = t_eval.reshape(1, -1)
    differences = np.abs(datahf_values - t_eval_values)
    closest_indices = np.argmin(differences, axis=0)
    nearest_x = datahf[indices[closest_indices], 0]
    y_obs = Yhf[indices[closest_indices]]
    
    return nearest_x, y_obs


def calculate_cov_likelihood(sigma: float, t_eval: np.ndarray) -> np.ndarray:
    """
    Calculate the covariance matrix for the likelihood.
    
    Parameters:
    - sigma: Standard deviation for the likelihood.
    - t_eval: 2D numpy array of evaluation times.
    
    Returns:
    - cov_likelihood: 2D numpy array representing the covariance matrix.
    """
    return sigma ** 2 * np.eye(t_eval.shape[0])

def run_simulation(
    datahf: np.ndarray, 
    mean_prior: np.ndarray, 
    cov_prior: np.ndarray, 
    Yhf: np.ndarray, 
    sigma_noise: List[float], 
    n_data: List[int],
    parameters: np.ndarray, 
    sigma: np.ndarray, 
    rwmh_scaling: np.ndarray, 
    rwmh_cov: np.ndarray, 
    rwmh_adaptive: bool, 
    iterations: int, 
    burnin: int, 
    n_chains: int, 
    final_model,
    algo: str, 
    subsampling_rate: Union[int, List[int]] = 1,
    levels:int=1,
    force_sequential:bool=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a simulation to estimate parameters and calculate errors.
    
    Parameters:
    - datahf: 2D numpy array containing data.
    - mean_prior: 1D numpy array for the mean of the prior.
    - cov_prior: 2D numpy array for the covariance of the prior.
    - Yhf: 1D numpy array of observed values.
    - sigma_noise: List of noise levels.
    - n_data: List of number of data points.
    - parameters: 1D numpy array of parameters.
    - sigma: 1D numpy array of standard deviations for the likelihood.
    - rwmh_scaling: 1D numpy array of scaling factors for the RWMH algorithm.
    - rwmh_cov: 2D numpy array for the RWMH covariance.
    - rwmh_adaptive: Boolean indicating if RWMH is adaptive.
    - iterations: Integer for the number of iterations.
    - burnin: Integer for the burn-in period.
    - n_chains: Integer for the number of chains.
    - final_model: The model object with the param_inverse method.
    - subsampling_rate (Union[int, List[int]]): Rate or rates of subsampling the posterior. Default is 1.
    - algo: String indicating the algorithm to use.
    - levels: number of levels to implement a ML MCMC approach. Valid only for Multifidelity networks to exploit different accuracy discretization levels
    - force_sequential: if True impose a sequential approach to the MCMC algorithm

    Returns:
    - best_estimate: The best parameter estimate.
    - best_error: The error corresponding to the best estimate.
    """
    
    # Initialize error and estimate arrays
    error_shape = (len(sigma_noise), len(n_data), len(sigma), len(rwmh_scaling))
    error = []
    estimates = []
    param_results = []

    # Iterate over all combinations of parameters using itertools.product
    for (i, noise), (k, n), (t, s), (j, r) in product(enumerate(sigma_noise), enumerate(n_data), enumerate(sigma), enumerate(rwmh_scaling)):
        t_eval = np.linspace(np.min(datahf[:,0]), np.max(datahf[:,0]), n).reshape(-1, 1)  # Generate evaluation times
        nearest_x, y_obs = process_data(datahf, parameters, t_eval, Yhf)  # Get nearest x values and observations
        cov_likelihood = calculate_cov_likelihood(s, t_eval)  # Compute the covariance for the likelihood

        # Perform parameter estimation and calculate error
        est, err, param_res = final_model.param_inverse(
            mean_prior, t_eval, max_par=max(datahf[:,1]),cov_prior=cov_prior, rmwh_scaling=r, levels=levels,
            cov_noise=noise, cov_likelihood=cov_likelihood, y_obs=y_obs, 
            x_real=parameters, number_chains=n_chains, N=iterations, 
            burn_in=burnin, diagnostic=True, rwmh_cov=rwmh_cov, 
            rwmh_adaptive=rwmh_adaptive, subsampling_rate=subsampling_rate, algo=algo, force_sequential=force_sequential
        )
        estimates.append(est)
        error.append(err)
        param_results.append(param_res)
    # Identify the index of the minimum error
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, "
          f"rwmh_scaling={rwmh_scaling[smallest_index[3]]}")

    return best_estimate, best_error, param_results

# def single_simulation(
#     noise: float, n: int, s: float, r: float, datahf: np.ndarray, parameters: np.ndarray, 
#     Yhf: np.ndarray, mean_prior: np.ndarray, cov_prior: np.ndarray, rwmh_scaling: np.ndarray, 
#     rwmh_cov: np.ndarray, rwmh_adaptive: bool, iterations: int, burnin: int, n_chains: int, 
#     final_model, algo: str, levels: int, force_sequential: bool
# ) -> Tuple[np.ndarray, float, np.ndarray]:
#     """
#     Execute a single simulation for a given set of hyperparameters.

#     Parameters:
#     - noise: Noise level for the covariance.
#     - n: Number of data points.
#     - s: Standard deviation for the likelihood.
#     - r: Scaling factor for the RWMH algorithm.
#     - Other parameters are shared with the main function.

#     Returns:
#     - est: Estimated parameters.
#     - err: Error corresponding to the estimate.
#     - param_res: Results of the parameter estimation.
#     """
#     t_eval = np.linspace(0., 5., n).reshape(-1, 1)
#     _, y_obs = process_data(datahf, parameters, t_eval, Yhf)
#     cov_likelihood = calculate_cov_likelihood(s, t_eval)

#     est, err, param_res = final_model.param_inverse(
#         mean_prior, t_eval, max_par=max(datahf[:, 1]), cov_prior=cov_prior, rmwh_scaling=r, 
#         levels=levels, cov_noise=noise, cov_likelihood=cov_likelihood, y_obs=y_obs, 
#         x_real=parameters, number_chains=n_chains, N=iterations, 
#         burn_in=burnin, diagnostic=True, rwmh_cov=rwmh_cov, 
#         rwmh_adaptive=rwmh_adaptive, algo=algo, force_sequential=force_sequential
#     )
#     return est, err, param_res


# def run_simulation(
#     datahf: np.ndarray, 
#     mean_prior: np.ndarray, 
#     cov_prior: np.ndarray, 
#     Yhf: np.ndarray, 
#     sigma_noise: List[float], 
#     n_data: List[int],
#     parameters: np.ndarray, 
#     sigma: np.ndarray, 
#     rwmh_scaling: np.ndarray, 
#     rwmh_cov: np.ndarray, 
#     rwmh_adaptive: bool, 
#     iterations: int, 
#     burnin: int, 
#     n_chains: int, 
#     final_model, 
#     algo: str, 
#     levels: int = 1,
#     force_sequential: bool = False,
#     max_workers: int = 2  # Limit the number of external processes
# ) -> Tuple[np.ndarray, np.ndarray, list]:
#     """
#     Run a simulation to estimate parameters and calculate errors.

#     Parameters:
#     - datahf: 2D numpy array containing data.
#     - mean_prior: 1D numpy array for the mean of the prior.
#     - cov_prior: 2D numpy array for the covariance of the prior.
#     - Yhf: 1D numpy array of observed values.
#     - sigma_noise: List of noise levels.
#     - n_data: List of number of data points.
#     - parameters: 1D numpy array of parameters.
#     - sigma: 1D numpy array of standard deviations for the likelihood.
#     - rwmh_scaling: 1D numpy array of scaling factors for the RWMH algorithm.
#     - rwmh_cov: 2D numpy array for the RWMH covariance.
#     - rwmh_adaptive: Boolean indicating if RWMH is adaptive.
#     - iterations: Integer for the number of iterations.
#     - burnin: Integer for the burn-in period.
#     - n_chains: Integer for the number of chains.
#     - final_model: The model object with the param_inverse method.
#     - algo: String indicating the algorithm to use.
#     - levels: Number of levels for ML MCMC in multifidelity networks.
#     - force_sequential: If True, force sequential MCMC execution.
#     - max_workers: Maximum number of processes to use for parallelization.

#     Returns:
#     - best_estimate: The best parameter estimate.
#     - best_error: The error corresponding to the best estimate.
#     - param_results: A list of all parameter estimation results.
#     """
    
#     # Define the combinations of hyperparameters to explore
#     combinations = list(product(sigma_noise, n_data, sigma, rwmh_scaling))
    
#     # Parallel execution with a limited number of workers
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         results = list(executor.map(
#             lambda combo: single_simulation(*combo, datahf, parameters, Yhf, 
#                                             mean_prior, cov_prior, rwmh_scaling, 
#                                             rwmh_cov, rwmh_adaptive, iterations, 
#                                             burnin, n_chains, final_model, algo, 
#                                             levels, force_sequential), 
#             combinations
#         ))

#     # Unpacking results
#     estimates, error, param_results = zip(*results)

#     # Identify the index of the minimum error
#     smallest_index = np.argmin(error)
#     best_estimate = estimates[smallest_index]
#     best_error = error[smallest_index]

#     # Extract the best hyperparameter combination
#     best_params = combinations[smallest_index]
#     print(f"The best estimate is given by: sigma_noise={best_params[0]}, "
#           f"number of data={best_params[1]}, sigma={best_params[2]}, "
#           f"rwmh_scaling={best_params[3]}")

#     return best_estimate, best_error, param_results

def run_simulation_cuqi(
    data: dict,
    mean_prior: np.ndarray,
    x_real: np.ndarray,
    N: int,
    burn_in: int,
    cov_prior: np.ndarray,
    sd_noise: list,
    adapt: bool,
    proposal_sd: list,
    number_chains: int,
    algo: str,
    x_data: np.ndarray,
    n_data: list,
    final_model,
    parallel: bool
) -> tuple:
    """
    Run a CUQI simulation to estimate parameters and compute error.

    Args:
        data (dict): Dictionary containing high-fidelity data (keys: "xhf" and "Yhf").
        mean_prior (np.ndarray): Prior mean vector.
        x_real (np.ndarray): Real x values.
        N (int): Number of samples.
        burn_in (int): Number of burn-in samples.
        cov_prior (np.ndarray): Prior covariance matrix.
        sd_noise (list): List of noise standard deviations to evaluate.
        adapt (bool): Whether to use adaptation in the algorithm.
        proposal_sd (list): List of proposal standard deviations to evaluate.
        number_chains (int): Number of MCMC chains.
        algo (str): Algorithm to use for MCMC.
        x_data (np.ndarray): Initial evaluation times.
        n_data (list): List of data sizes to evaluate.
        final_model: Final model object with inverse_cuqi method.
        parallel (bool): Whether to run MCMC chains in parallel.

    Returns:
        tuple: Best estimate and best error found during the simulation.
    """

    # Initialize estimates and error arrays
    estimates = np.zeros((len(sd_noise), len(n_data), len(proposal_sd)))
    error = np.zeros((len(sd_noise), len(n_data), len(proposal_sd)))
    param_results = []

    # Iterate over noise levels
    for i, noise in enumerate(sd_noise):
        # Iterate over number of data points
        for k, n in enumerate(n_data):
            # Generate evaluation times
            x_data = np.linspace(0., 5., n).reshape(-1, 1)                          # CORREGGI 
            nearest_x, y_obs = process_data(data["xhf"], x_real, x_data, data["Yhf"])
            
            # Iterate over proposal standard deviations
            for t, s in enumerate(proposal_sd):

                # Perform parameter estimation and calculate error
                estimates[i, k, t], error[i, k, t], par = final_model.inverse_cuqi(
                    mean_prior=mean_prior,
                    x_real=x_real,
                    max_par=max(data["xhf"][:,1]),
                    y_obs=y_obs,
                    N=N,
                    burn_in=burn_in,
                    cov_prior=cov_prior,
                    sd_noise=noise,  
                    adapt=adapt,
                    scale=s,
                    proposal_sd=s,
                    number_chains=number_chains,
                    algo=algo,
                    x_data=x_data,
                    parallel=parallel
                )
                
                param_results.append(par)
    # Find the smallest error and corresponding indices
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sd_noise={sd_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, proposal_standard_deviation={proposal_sd[smallest_index[2]]}")

    return best_estimate, best_error, param_results



def custom_activation(x: tf.Tensor) -> tf.Tensor:
    """
    Custom activation function combining linear and non-linear transformations.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Transformed tensor.
    """
    return x + K.square(K.sin(x))
    
    
# Custom Loss Function
def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom loss function that ignores certain values in y_pred.

    Args:
        y_true (tf.Tensor): True values.
        y_pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: Computed loss.
    """
    # Identify indices where y_pred is not equal to -10
    goodind = tf.not_equal(y_pred, -10.0)
    
    # Mask y_pred and y_true based on the identified indices
    y_pred_loss = tf.boolean_mask(y_pred, goodind)
    y_pred_true = tf.boolean_mask(y_true, goodind)
    
    # Compute mean squared error loss
    return K.mean(K.square(y_pred_loss - y_pred_true))



# Get Optimizer
def getOpti(name: str, lr: float) -> tf.keras.optimizers.Optimizer:
    """
    Returns the optimizer based on the given name.

    Args:
        name (str): Name of the optimizer.
        lr (float): Learning rate for the optimizer.

    Returns:
        tf.keras.optimizers.Optimizer: The selected optimizer.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    optimizers = {
        'Adam': Adam(learning_rate=lr, amsgrad=True),
        'Nadam': Nadam(learning_rate=lr),
        'Adamax': Adamax(learning_rate=lr),
        'standardadam': 'adam'
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer name: {name}")
    
    return optimizers[name]

def getModel(params: dict, num_inputs: int, name: str, num_outputs: int) -> Model:
    """
    Returns a compiled Keras model based on the specified architecture.

    Parameters:
    - params: Dictionary containing parameters such as 'nodes', 'l2weight', 'kernel_init', 'opt', 'lr', and 'alpha'.
    - num_inputs: Integer representing the number of input features.
    - name: String specifying the model architecture ('HF', 'LF', 'Single', 'Hflin', 'Hfper', 'GP', or 'Inter').
    - num_outputs: Integer representing the number of output nodes.

    Returns:
    - model: A compiled Keras model.
    """
    inputs = Input(shape=(num_inputs,))
    
    if name == 'HF':
        # High-Fidelity model with a single hidden layer
        hidden1 = Dense(int(params['nodes']), activation='tanh', 
                        kernel_regularizer=l2(params['l2weight']), 
                        kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs, activation='linear', name='HF')(hidden1)

    elif name == 'LF':
        # Low-Fidelity model with 4 hidden layers
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden3)
        output = Dense(num_outputs, activation='linear', name='LF')(hidden4)
        
    elif name == 'Single':
        # Single model with 4 hidden layers and L2 regularization
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], 
                        kernel_regularizer=l2(params['l2weight']))(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], 
                        kernel_regularizer=l2(params['l2weight']))(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], 
                        kernel_regularizer=l2(params['l2weight']))(hidden2)
        output = Dense(num_outputs, activation='linear', name='Single')(hidden2)

    elif name == 'Hflin':
        # High-Fidelity linear model with L2 regularization
        hiddenlin = Dense(64, activation='linear', kernel_regularizer=l2(params['l2weight']), 
                          kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs, activation='linear', name='HFlin')(hiddenlin)

    elif name == 'Hfper':
        # High-Fidelity model with a custom activation function
        hiddenlin = Dense(64, activation=custom_activation, kernel_regularizer=l2(params['l2weight']), 
                          kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs, activation='linear', name='HFper')(hiddenlin)

    elif name == 'GP':
        # Gaussian Process-like model with multiple outputs (HF and LF)
        hidden1 = Dense(int(params['nodes']), activation='tanh', 
                        kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                        kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(int(params['nodes']), activation='tanh', 
                        kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                        kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(int(params['nodes']), activation='tanh', 
                        kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                        kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(int(params['nodes']), activation='tanh', 
                        kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                        kernel_initializer=params['kernel_init'])(hidden3)
        GPlayer = Dense(2, activation='linear', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                        kernel_initializer=params['kernel_init'])(hidden4)
        outputLF = Dense(num_outputs, activation='linear', name='LF')(GPlayer)
        outputHF = Dense(num_outputs, activation='linear', name='HF')(GPlayer)
        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1 - params['alpha']], optimizer=opti)
        return model

    elif name == 'Inter':
        # Intermediate model with a merge layer and dual outputs (HF and LF)
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden1)
        outputLF = Dense(num_outputs, activation='linear', name='LF')(hidden2)
        outputadd = Dense(int(params['nodes']), activation='tanh', 
                          kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                          kernel_initializer=params['kernel_init'])(hidden2)
        merge = concatenate([outputLF, outputadd])
        hidden3 = Dense(int(params['nodes']), activation='tanh', 
                        kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                        kernel_initializer=params['kernel_init'])(merge)
        hidden4 = Dense(int(params['nodes']), activation='tanh', 
                        kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                        kernel_initializer=params['kernel_init'])(hidden3)
        outputHF = Dense(4, activation='linear', name='HF')(hidden4)
        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1 - params['alpha']], optimizer=opti)
        return model

    # For other models without multiple outputs
    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params['opt'], params['lr'])
    model.compile(loss='mse', optimizer=opti, metrics=['mse'])
    
    return model

