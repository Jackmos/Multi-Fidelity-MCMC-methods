

# import numpy as np
# import os
# import ray
# import warnings
# import matplotlib.pyplot as plt
# from tensorflow.keras import backend as K
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Input, concatenate
# from tensorflow.keras.optimizers import Adam, Nadam, Adamax
# import tensorflow as tf
# import arviz as az
# from typing import Optional, Any, Dict, Tuple, List
# import logging
# import tinyDA as tda
# from cuqi.distribution import JointDistribution
# from cuqi.sampler import MH, NUTS, pCN
# import time
# # Configure Python logging to suppress detailed Keras messages
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# from itertools import product
# import contextlib




from itertools import product
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, Nadam, Adamax
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from typing import Dict, Any, List, Tuple
import tensorflow as tf
import numpy as np
import time
import logging
# Configure Python logging to suppress detailed Keras messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from itertools import product


def compute_time(function):
    def wrapper(*args, **kwargs):
        init = time.time()
        res = function(*args, **kwargs)
        end = time.time()
        timespam = end - init
        print(f"The function {function.__name__} took {timespam} seconds.")
        return res
    return wrapper

######################## VEDI SE ELIMINNARE

# Enable XLA JIT compilation
tf.config.optimizer.set_jit(True)
#######################################################àà



def process_data(
    datahf: np.ndarray, 
    parameters: np.ndarray, 
    domain_eval: np.ndarray, 
    Yhf: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the dataset elements nearest to specific evaluation points in the input domain.

    Parameters:
    - datahf: 2D numpy array where each row represents an observation. 
              Column 0 contains input values, and column 1 contains parameter values.
    - parameters: 1D numpy array of parameter values for which we want to find the nearest input points in datahf.
    - domain_eval: 1D numpy array of input domain points at which we want to evaluate the model.
    - Yhf: 1D numpy array containing observed y values corresponding to rows in datahf.

    Returns:
    - nearest_input: 1D numpy array of input points from datahf that are closest to each value in domain_eval.
    - y_obs: 1D numpy array of y values from Yhf corresponding to the nearest input points.
    """
    # Identify indices in datahf corresponding to the first parameter value
    indices = np.where(datahf[:, 1] == parameters[0])[0]
    if len(indices) == 0:
        raise ValueError(f"No observations related to parameter: {parameters[0]}")
    
    # Extract the relevant input values from datahf
    datahf_inputs = datahf[indices, 0].reshape(-1, 1)
    
    # Reshape domain_eval for broadcasting
    domain_eval_reshaped = domain_eval.reshape(1, -1)
    
    # Calculate the absolute differences between datahf_inputs and domain_eval points
    input_differences = np.abs(datahf_inputs - domain_eval_reshaped)
    
    # Identify the indices of the minimum differences
    closest_indices = np.argmin(input_differences, axis=0)
    
    # Retrieve the nearest input points and their corresponding y values
    nearest_input = datahf[indices[closest_indices], 0]
    y_obs = Yhf[indices[closest_indices]]
    
    return nearest_input, y_obs


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
    levels: int = 1, 
    force_sequential: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Run a simulation to estimate parameters and calculate errors using a Monte Carlo method.

    Parameters:
    - datahf: 2D numpy array of observational data, where each row corresponds to an input-parameter pair.
    - mean_prior: 1D numpy array representing the mean of the prior distribution.
    - cov_prior: 2D numpy array representing the covariance matrix of the prior distribution.
    - Yhf: 1D numpy array of observed y values.
    - sigma_noise: List of noise levels to be used in the simulation.
    - n_data: List representing the number of data points to sample for evaluation.
    - parameters: 1D numpy array of true parameter values.
    - sigma: 1D numpy array of standard deviations used in the likelihood function.
    - rwmh_scaling: 1D numpy array of scaling factors for the Random Walk Metropolis-Hastings (RWMH) algorithm.
    - rwmh_cov: 2D numpy array representing the covariance matrix used in RWMH.
    - rwmh_adaptive: Boolean indicating whether the RWMH algorithm is adaptive.
    - iterations: Integer representing the number of iterations for the Monte Carlo simulation.
    - burnin: Integer representing the number of burn-in iterations to discard.
    - n_chains: Integer representing the number of chains to run in the simulation.
    - final_model: A model object with a method `param_inverse` for parameter estimation.
    - algo: String specifying the algorithm to use for the Monte Carlo simulation.
    - levels: Integer specifying the number of levels for a Multilevel approach (default is 1).
    - force_sequential: Boolean indicating whether to enforce a sequential MCMC approach (default is False).

    Returns:
    - best_estimate: The best parameter estimate from the simulation.
    - best_error: The smallest error associated with the best estimate.
    - param_results: List of dictionaries containing results from the MCMC algorithm for each parameter set.
    """
    
    # Initialize arrays to store errors and estimates
    error_shape = (len(sigma_noise), len(n_data), len(sigma), len(rwmh_scaling))
    errors = np.zeros(error_shape)
    estimates = np.zeros(error_shape)
    param_results = []

    # Iterate over all combinations of noise, data size, sigma, and RWMH scaling
    for (i, noise), (k, n), (t, sigma_val), (j, scaling_factor) in product(
        enumerate(sigma_noise), enumerate(n_data), enumerate(sigma), enumerate(rwmh_scaling)
    ):
        # Generate evaluation points in the input domain
        domain_eval = np.linspace(0., 5., n).reshape(-1, 1)
        
        # Find the closest input points in datahf and their corresponding y values
        nearest_input, y_obs = process_data(datahf, parameters, domain_eval, Yhf)
        
        # Compute the covariance matrix for the likelihood function
        cov_likelihood = calculate_cov_likelihood(sigma_val, domain_eval)

        # Estimate parameters using the model's param_inverse method and compute error
        estimates[i, k, t, j], errors[i, k, t, j], param_res = final_model.param_inverse(
            mean_prior, 
            domain_eval, 
            max_par=max(datahf[:,1]),
            cov_prior=cov_prior, 
            cov_noise=noise,
            cov_likelihood=cov_likelihood,
            y_obs=y_obs,
            x_real=parameters,
            number_chains=n_chains,
            N=iterations,
            burn_in=burnin,
            levels=levels, 
            diagnostic=True, 
            rwmh_cov=rwmh_cov, 
            rmwh_scaling=scaling_factor,
            rwmh_adaptive=rwmh_adaptive, 
            algo=algo, 
            force_sequential=force_sequential
        )
        param_results.append(param_res)
    
    # Identify the index of the minimum error and retrieve corresponding estimates
    smallest_index = np.unravel_index(np.argmin(errors), errors.shape)
    best_estimate = estimates[smallest_index]
    best_error = errors[smallest_index]
    
    # Output the best parameters for debugging or review
    print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, "
          f"rwmh_scaling={rwmh_scaling[smallest_index[3]]}")

    return best_estimate, best_error, param_results



def run_simulation_cuqi(
    data: dict,
    mean_prior: np.ndarray,
    parameters: np.ndarray,
    iterations: int,
    burn_in: int,
    cov_prior: np.ndarray,
    sd_noise: list,
    adapt: bool,
    proposal_sd: list,
    number_chains: int,
    algo: str,
    n_data: list,
    scale: list,
    fwd_model,
    parallel: bool
) -> tuple:
    """
    Run a CUQI simulation to estimate parameters and compute error.
    
    Parameters:
    - data: Dictionary containing high-fidelity data (keys: "xhf" and "Yhf").
    - mean_prior: 1D numpy array for the mean of the prior.
    - parameters: 1D numpy array of parameter values.
    - iterations: Integer for the number of iterations (samples).
    - burn_in: Integer for the number of burn-in samples.
    - cov_prior: 2D numpy array for the covariance of the prior.
    - sd_noise: List of noise standard deviations to evaluate.
    - adapt: Boolean indicating whether to use adaptation in the algorithm.
    - proposal_sd: List of proposal standard deviations to evaluate.
    - number_chains: Integer for the number of MCMC chains.
    - algo: String indicating the algorithm to use for MCMC.
    - n_data: List of integers representing different data sizes to evaluate.
    - scale: List of scaling factors to evaluate.
    - fwd_model: Forward model object with the inverse_cuqi method.
    - parallel: Boolean indicating whether to run MCMC chains in parallel.

    Returns:
    - best_estimate: The best parameter estimate.
    - best_error: The error corresponding to the best estimate.
    - params_result: List of parameter results from the MCMC algorithm.
    """

    # Initialize estimates and error arrays
    estimates = np.zeros((len(sd_noise), len(n_data), len(scale), len(proposal_sd)))
    error = np.zeros((len(sd_noise), len(n_data), len(scale), len(proposal_sd)))
    params_result = []

    # Iterate over all combinations of noise levels, data sizes, scales, and proposal standard deviations
    for i, noise in enumerate(sd_noise):
        for k, n in enumerate(n_data):
            # Generate evaluation points in the input domain
            x_data = np.linspace(0., 5., n).reshape(-1, 1)
            nearest_x, y_obs = process_data(data["xhf"], parameters, x_data, data["Yhf"])

            for t, s in enumerate(proposal_sd):
                for r, scale_factor in enumerate(scale):
                    # Perform parameter estimation and calculate error
                    estimates[i, k, r, t], error[i, k, r, t], par = fwd_model.inverse_cuqi(
                        mean_prior=mean_prior,
                        x_real=parameters,
                        max_par=max(data["xhf"][:,1]),
                        y_obs=y_obs,
                        N=iterations,
                        burn_in=burn_in,
                        cov_prior=cov_prior,
                        sd_noise=noise,
                        adapt=adapt,
                        scale=scale_factor,
                        proposal_sd=s,
                        number_chains=number_chains,
                        algo=algo,
                        x_data=x_data,
                        parallel=parallel
                    )
                    params_result.append(par)
    
    # Find the smallest error and corresponding indices
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sd_noise={sd_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, scale={scale[smallest_index[2]]}, "
          f"proposal_sd={proposal_sd[smallest_index[3]]}")

    return best_estimate, best_error, params_result




# Custom Activation Function
def custom_activation(x: tf.Tensor) -> tf.Tensor:
    """
    Apply a custom activation function combining linear and non-linear transformations.
    
    Parameters:
    - x: TensorFlow tensor, input tensor to the activation function.
    
    Returns:
    - TensorFlow tensor, transformed tensor after applying the custom activation.
    """
    return x + K.square(K.sin(x))

# Custom Loss Function
def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute a custom loss function, ignoring certain values in y_pred.
    
    Parameters:
    - y_true: TensorFlow tensor, true values.
    - y_pred: TensorFlow tensor, predicted values.
    
    Returns:
    - TensorFlow tensor, computed loss based on the mean squared error, 
      ignoring y_pred values equal to -10.
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
    Retrieve an optimizer based on the given name.
    
    Parameters:
    - name: String, name of the optimizer to retrieve.
    - lr: Float, learning rate to be used with the optimizer.
    
    Returns:
    - tf.keras.optimizers.Optimizer: The specified optimizer instance.
    
    Raises:
    - ValueError: If the provided optimizer name is unknown.
    """
    optimizers = {
        'Adam': Adam(learning_rate=lr, amsgrad=True),
        'Nadam': Nadam(learning_rate=lr),
        'Adamax': Adamax(learning_rate=lr),
        'standardadam': 'adam'  # Placeholder for a potential standard Adam optimizer
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer name: {name}")
    
    return optimizers[name]


def getModel(params: Dict[str, Any], num_inputs: int, name: str, num_outputs: int) -> tf.keras.models.Model:
    """
    Create and return a compiled Keras model based on the specified parameters and model type.
    
    Parameters:
    - params: Dictionary containing model configuration parameters, such as 
              'nodes', 'l2weight', 'kernel_init', 'alpha', and 'opt'.
    - num_inputs: Integer specifying the number of input features.
    - name: String indicating the model type ('HF', 'LF', 'Single', 'Hflin', 'Hfper', 'GP', 'Inter').
    - num_outputs: Integer specifying the number of output features.
    
    Returns:
    - A compiled Keras model tailored to the specified model type.
    """

    inputs = Input(shape=(num_inputs,))

    if name == 'HF':
        hidden1 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2(params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HF')(hidden1)

    elif name == 'LF':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(hidden2)
        hidden4 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(hidden3)
        output = Dense(num_outputs, activation='linear', name='LF')(hidden4)

    elif name == 'Single':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']),
                        name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']),
                        name=f'{name}_hidden2')(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']),
                        name=f'{name}_hidden3')(hidden2)
        hidden4 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']),
                        name=f'{name}_hidden4')(hidden3)
        output = Dense(num_outputs, activation='linear', name='Single')(hidden4)

    elif name == 'Hflin':
        hiddenlin = Dense(64, activation='linear', kernel_regularizer=l2(params['l2weight']),
                          kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HFlin')(hiddenlin)

    elif name == 'Hfper':
        hiddenlin = Dense(64, activation=custom_activation, kernel_regularizer=l2(params['l2weight']),
                          kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HFper')(hiddenlin)

    elif name == 'GP':
        hidden1 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
        hidden3 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(hidden2)
        hidden4 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(hidden3)
        GPlayer = Dense(2, activation='linear', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_GP')(hidden4)
        outputLF = Dense(1, activation='linear', name='LF')(GPlayer)
        outputHF = Dense(1, activation='linear', name='HF')(GPlayer)
        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1 - params['alpha']], optimizer=opti)
        return model

    elif name == 'Inter':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
        outputLF = Dense(1, activation='linear', name='LF')(hidden2)
        outputadd = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                          kernel_initializer=params['kernel_init'])(hidden2)
        merge = concatenate([outputLF, outputadd])
        hidden3 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(merge)
        hidden4 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(hidden3)
        outputHF = Dense(1, activation='linear', name='HF')(hidden4)
        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1 - params['alpha']], optimizer=opti)
        return model

    # For other cases, create a standard model
    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params['opt'], params['lr'])
    model.compile(loss='mse', optimizer=opti, metrics=['mse'])
    return model
