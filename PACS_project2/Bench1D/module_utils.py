

import numpy as np
import os
import ray
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam, Nadam, Adamax
import tensorflow as tf
import arviz as az
from typing import Optional, Any, Dict, Tuple, List
import logging
import tinyDA as tda
from cuqi.distribution import JointDistribution
from cuqi.sampler import MH, NUTS, pCN
import time
# Configure Python logging to suppress detailed Keras messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from itertools import product
import contextlib





######################## VEDI SE ELIMINNARE
def compute_time(function):
    def wrapper(*args, **kwargs):
        init = time.time()
        res = function(*args, **kwargs)
        end = time.time()
        timespam = end - init
        print(f"The function {function.__name__} took {timespam} seconds.")
        return res
    return wrapper


# Enable XLA JIT compilation
tf.config.optimizer.set_jit(True)
#######################################################àà






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
    algo: str
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
    - algo: String indicating the algorithm to use.
    
    Returns:
    - best_estimate: The best parameter estimate.
    - best_error: The error corresponding to the best estimate.
    """
    
    # Initialize error and estimate arrays
    error_shape = (len(sigma_noise), len(n_data), len(sigma), len(rwmh_scaling))
    error = np.zeros(error_shape)
    estimates = np.zeros(error_shape)
    
    # Iterate over all combinations of parameters using itertools.product
    for (i, noise), (k, n), (t, s), (j, r) in product(enumerate(sigma_noise), enumerate(n_data), enumerate(sigma), enumerate(rwmh_scaling)):
        t_eval = np.linspace(0., 5., n).reshape(-1, 1)  # Generate evaluation times
        nearest_x, y_obs = process_data(datahf, parameters, t_eval, Yhf)  # Get nearest x values and observations
        cov_likelihood = calculate_cov_likelihood(s, t_eval)  # Compute the covariance for the likelihood

        # Perform parameter estimation and calculate error
        estimates[i, k, t, j], error[i, k, t, j] = final_model.param_inverse(
            mean_prior, t_eval, max_par=max(datahf[:,1]),cov_prior=cov_prior, rmwh_scaling=r, 
            cov_noise=noise, cov_likelihood=cov_likelihood, y_obs=y_obs, 
            x_real=parameters, number_chains=n_chains, N=iterations, 
            burn_in=burnin, diagnostic=True, rwmh_cov=rwmh_cov, 
            rwmh_adaptive=rwmh_adaptive, algo=algo
        )
    
    # Identify the index of the minimum error
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, "
          f"rwmh_scaling={rwmh_scaling[smallest_index[3]]}")

    return best_estimate, best_error



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

    # Iterate over noise levels
    for i, noise in enumerate(sd_noise):
        # Iterate over number of data points
        for k, n in enumerate(n_data):
            # Generate evaluation times
            x_data = np.linspace(0., 5., n).reshape(-1, 1)
            nearest_x, y_obs = process_data(data["xhf"], x_real, x_data, data["Yhf"])
            
            # Iterate over proposal standard deviations
            for t, s in enumerate(proposal_sd):

                # Perform parameter estimation and calculate error
                estimates[i, k, t], error[i, k, t] = final_model.inverse_cuqi(
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
                
    # Find the smallest error and corresponding indices
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sd_noise={sd_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, proposal_standard_deviation={proposal_sd[smallest_index[2]]}")

    return best_estimate, best_error




# Custom Activation Function
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
        #'RMSprop': RMSprop(learning_rate=lr),
        'standardadam': 'adam'
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer name: {name}")
    
    return optimizers[name]


def getModel(params: Dict[str, Any], num_inputs: int, name: str, num_outputs: int) -> tf.keras.models.Model:
    """
    Returns a compiled Keras model based on the given parameters and model type.

    Args:
        params (Dict[str, Any]): Dictionary of parameters for the model.
        num_inputs (int): Number of input features.
        name (str): Type of model to create ('HF', 'LF', 'Single', 'Hflin', 'Hfper', 'GP', 'Inter').
        num_outputs (int): Number of output features.

    Returns:
        tf.keras.models.Model: Compiled Keras model.
    """
    inputs = Input(shape=(num_inputs,))

    if name == 'HF':
        hidden1 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HF')(hidden1)

    elif name == 'LF':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(hidden2)
        hidden4 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(hidden3)
        output = Dense(num_outputs, activation='linear', name='LF')(hidden4)

    elif name == 'Single':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']), name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']), name=f'{name}_hidden2')(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']), name=f'{name}_hidden3')(hidden2)
        hidden4 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']), name=f'{name}_hidden4')(hidden3)
        output = Dense(num_outputs, activation='linear', name='Single')(hidden4)

    elif name == 'Hflin':
        hiddenlin = Dense(64, activation='linear', kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HFlin')(hiddenlin)

    elif name == 'Hfper':
        hiddenlin = Dense(64, activation=custom_activation, kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HFper')(hiddenlin)

    elif name == 'GP':
        hidden1 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
        hidden3 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(hidden2)
        hidden4 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(hidden3)
        GPlayer = Dense(2, activation='linear', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_GP')(hidden4)
        outputLF = Dense(1, activation='linear', name='LF')(GPlayer)
        outputHF = Dense(1, activation='linear', name='HF')(GPlayer)
        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1-params['alpha']], optimizer=opti)
        return model

    elif name == 'Inter':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
        outputLF = Dense(1, activation='linear', name='LF')(hidden2)
        outputadd = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'])(hidden2)
        merge = concatenate([outputLF, outputadd])
        hidden3 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(merge)
        hidden4 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(hidden3)
        outputHF = Dense(1, activation='linear', name='HF')(hidden4)
        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1-params['alpha']], optimizer=opti)
        return model

    # For other cases, create a standard model
    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params['opt'], params['lr'])
    model.compile(loss='mse', optimizer=opti, metrics=['mse'])
    return model

# # MCMC Sampling Function

# def MCMC(
#     my_posterior: Any, 
#     N: int, 
#     burnin: int, 
#     n: int = 1, 
#     diagnostic: bool = True, 
#     rwmh_cov: Optional[np.ndarray] = None, 
#     rmwh_scaling: float = 0.1, 
#     period: int = 100, 
#     t0: int = 0, 
#     rwmh_adaptive: bool = False, 
#     algo: str = "MH", 
#     dim: int = 0
# ) -> np.ndarray:
#     """
#     Perform MCMC sampling using the specified algorithm.

#     Args:
#         my_posterior (Any): The posterior distribution to sample from.
#         N (int): Number of MCMC iterations.
#         burnin (int): Number of burn-in iterations.
#         n (int, optional): Number of chains to run. Defaults to 1.
#         diagnostic (bool, optional): Whether to print and plot diagnostics. Defaults to True.
#         rwmh_cov (Optional[np.ndarray], optional): Covariance matrix for the proposal distribution. Defaults to None.
#         rmwh_scaling (float, optional): Scaling factor for the proposal distribution. Defaults to 0.1.
#         period (int, optional): Adaptation period for certain algorithms. Defaults to 100.
#         t0 (int, optional): Tuning parameter for adaptive algorithms. Defaults to 0.
#         rwmh_adaptive (bool, optional): Whether to use adaptive algorithms. Defaults to False.
#         algo (str, optional): Algorithm to use for MCMC sampling. Defaults to "MH".
#         dim (int, optional): Dimensionality of the problem, used for certain algorithms. Defaults to 0.

#     Returns:
#         np.ndarray: Estimated values from the MCMC sampling.
#     """
#     # Get the Maximum A Posteriori (MAP) estimate
#     MAP = tda.get_MAP(my_posterior) if dim != 1 else None

#     # Select the MCMC algorithm
#     if algo == "MH":
#         my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive)
#     elif algo == "AM":
#         my_proposal = tda.AdaptiveMetropolis(C0=rwmh_cov, adaptive=rwmh_adaptive, period=period, t0=t0)
#     elif algo == "CN":
#         my_proposal = tda.CrankNicolson(scaling=rmwh_scaling, adaptive=rwmh_adaptive, period=period)
#     elif algo == "DREAMZ":
#         my_proposal = tda.DREAMZ(M0=10*dim, adaptive=rwmh_adaptive, period=period)
#     else:
#         raise ValueError(f"Unknown algorithm {algo}")

#     # Sample using the selected proposal and posterior
#     my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
#     idata = tda.to_inference_data(my_chains, burnin=burnin)
#     estimates = np.array(az.summary(idata)['mean'])

#     # Print and plot diagnostics if enabled
#     print(f"Estimated values are {estimates}")
#     if diagnostic:
#         print(az.summary(idata))
#         az.plot_trace(idata)
#         print("Autocorrelation...")
#         az.plot_autocorr(idata)

#     return estimates


# # def MCMC_cuqi(
# #     y: Any, 
# #     x: Any, 
# #     observation: np.ndarray, 
# #     N: int, 
# #     burn_in: int, 
# #     n: int = 1, 
# #     diagnostic: bool = True, 
# #     algo: str = "MH", 
# #     adapt: bool = False, 
# #     scale: float = 0.3,
# #     parallel: bool = False
# # ) -> np.ndarray:
# #     """
# #     Perform MCMC sampling using CUQI library.

# #     Args:
# #         y (Any): Dependent variable.
# #         x (Any): Independent variable.
# #         observation (np.ndarray): Observed data.
# #         N (int): Number of samples to draw.
# #         burn_in (int): Number of burn-in samples to discard.
# #         n (int, optional): Number of chains. Defaults to 1.
# #         diagnostic (bool, optional): Whether to plot diagnostic plots. Defaults to True.
# #         algo (str, optional): Sampling algorithm to use. Defaults to "MH".
# #         adapt (bool, optional): Whether to use adaptive sampling. Defaults to False.
# #         scale (float, optional): Scaling factor for MH algorithm. Defaults to 0.3.

# #     Returns:
# #         np.ndarray: Array of estimated parameter means.
# #     """
# #     dim = observation.shape[0]
# #     x_init = np.random.rand(dim)
# #     estimates = np.empty((1, 0))   # Dimension of parameters to estimate
# #     chains = np.empty((0, 1, N - burn_in)) 
# #     post = np.empty((1, 0))  
# #     posterior = JointDistribution(y, x)(y=observation)
    
# #     if parallel:
# #         # Parallel execution using Ray
# #         logging.getLogger().setLevel(logging.WARNING)
# #         tf.get_logger().setLevel('ERROR')
# #         context=ray.init(num_cpus=4,logging_level=logging.WARNING)
# #         print(context.dashboard_url)

# #         futures = [chain_creation_parallel.remote(N, burn_in, diagnostic, algo, adapt, scale,posterior, x_init) for _ in range(n)]
# #         results = ray.get(futures)
# #         ray.shutdown()
# #     else:
# #         # Sequential execution
# #         results = [chain_creation(N, burn_in, diagnostic, algo, adapt, scale,posterior, x_init) for _ in range(n)]


# #     for i in range(len(results)):
# #         estimates = np.column_stack((estimates, results[i][0]))
# #         chains = np.concatenate((chains, results[i][1]), axis=0)
# #         post = np.concatenate((post, results[i][2]), axis=1)


# #     # Plot trace plots for each parameter
# #     for l in range(chains.shape[1]):
# #         plt.figure(figsize=(10, 4))
# #         for i in range(chains.shape[0]):
# #             plt.plot(chains[i, l, :])
# #         plt.xlabel('Sample')
# #         plt.ylabel('Value')
# #         plt.title(f'Trace Plot for variable {l}')
# #         plt.legend([f'Chain {i+1}' for i in range(chains.shape[0])])
# #         plt.show()

# #     # Diagnostic plots
# #     if diagnostic:
 
# #         num_bins = 20
# #         plt.figure()
# #         for num in range(post.shape[0]):
# #             bin_edges = np.linspace(np.min(post[num, :]), np.max(post[num, :]), num_bins + 1)
# #             hist, _ = np.histogram(post[num, :], bins=bin_edges)
# #             hist = hist / post.shape[1]
# #             bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# #             plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', label=f'Var {num + 1}')
# #             plt.xlabel('Value')
# #             plt.ylabel('Probability')
# #             plt.title('Distribution')
# #             plt.legend()
# #             plt.show()

# #         autocov = az.autocov(chains[:, 0, :])
# #         ess = az.ess(chains[:, 0, :])
# #         print(f"ESS values = {ess}")
# #         plt.figure()
# #         plt.plot(autocov[0, :])
# #         plt.title('Autocovariance first chain')
# #         plt.xlabel('Lag')
# #         plt.ylabel('Autocovariance')
# #         plt.legend()
# #         plt.show()
# #     return estimates

# # @ray.remote
# # def chain_creation_parallel(N, 
# #     burn_in, 
# #     diagnostic, 
# #     algo, 
# #     adapt, 
# #     scale,
# #  posterior, x_init):
# #     return chain_creation(N, 
# #     burn_in, 
# #     diagnostic, 
# #     algo, 
# #     adapt, 
# #     scale,
# #  posterior, x_init)
    
    
# # @compute_time    
# # def chain_creation(N, 
# #     burn_in, 
# #     diagnostic, 
# #     algo, 
# #     adapt, 
# #     scale,
# #  posterior, x_init): 
    
# #     # Select the appropriate sampler
# #     if algo == "NUTS":
# #         sampler = NUTS(posterior, x0=x_init)
# #     elif algo == "MH":
# #         sampler = MH(posterior, scale=scale) if not adapt else MH(posterior)
# #     elif algo == "pCN":
# #         sampler = pCN(posterior, x0=x_init)
# #     else:
# #         raise ValueError(f"Unknown algorithm {algo}")

# #     # Perform sampling
# #     samples = sampler.sample_adapt(N - burn_in, burn_in) if adapt else sampler.sample(N - burn_in, burn_in)

# #     # Compute estimates and store chains
# #     estimates = samples.mean()[:, np.newaxis]# np.column_stack((estimates, samples.mean()[:, np.newaxis]))
# #     chains = np.expand_dims(samples.samples, axis=0)# np.concatenate((chains, np.expand_dims(samples.samples, axis=0)), axis=0)
# #     post =samples.samples# np.concatenate((post, samples.samples), axis=1)

# #     print(f"Mean values: {estimates.mean(axis=1)}")
    
# #     return (estimates, chains,post)


# def MCMC_cuqi(
#     y: Any, 
#     x: Any, 
#     observation: np.ndarray, 
#     N: int, 
#     burn_in: int, 
#     n: int = 1, 
#     diagnostic: bool = True, 
#     algo: str = "MH", 
#     adapt: bool = False, 
#     scale: float = 0.3,
#     parallel: bool = False
# ) -> np.ndarray:
#     """
#     Perform MCMC sampling using CUQI library.

#     Args:
#         y (Any): Dependent variable.
#         x (Any): Independent variable.
#         observation (np.ndarray): Observed data.
#         N (int): Number of samples to draw.
#         burn_in (int): Number of burn-in samples to discard.
#         n (int, optional): Number of chains. Defaults to 1.
#         diagnostic (bool, optional): Whether to plot diagnostic plots. Defaults to True.
#         algo (str, optional): Sampling algorithm to use. Defaults to "MH".
#         adapt (bool, optional): Whether to use adaptive sampling. Defaults to False.
#         scale (float, optional): Scaling factor for MH algorithm. Defaults to 0.3.
#         parallel (bool, optional): Whether to run chains in parallel. Defaults to False.

#     Returns:
#         np.ndarray: Array of estimated parameter means.
#     """
#     dim = observation.shape[0]
#     x_init = np.random.rand(dim)
#     estimates = np.empty((1, 0))
#     chains = np.empty((0, 1, N - burn_in))
#     post = np.empty((1, 0))
#     posterior = JointDistribution(y, x)(y=observation)

#     if parallel:
#         # Parallel execution using Ray
#         # logging.getLogger().setLevel(logging.WARNING)
#         # tf.get_logger().setLevel('ERROR')
#         # # ray.init(num_cpus=4, logging_level=logging.WARNING)
        
#         # ray.init(logging_level=logging.WARNING)
#         logging.getLogger('tensorflow').setLevel(logging.ERROR)
#         tf.get_logger().setLevel('ERROR')    
#         ray.init(ignore_reinit_error=True, logging_level=logging.WARNING, log_to_driver=False)


#         futures = [chain_creation_parallel.remote(N, burn_in, diagnostic, algo, adapt, scale, posterior, x_init) for _ in range(n)]
#         results = ray.get(futures)
#         ray.shutdown()
#     else:
#         # Sequential execution
#         results = [chain_creation(N, burn_in, diagnostic, algo, adapt, scale, posterior, x_init) for _ in range(n)]

#     for result in results:
#         estimates = np.column_stack((estimates, result[0]))
#         chains = np.concatenate((chains, result[1]), axis=0)
#         post = np.concatenate((post, result[2]), axis=1)

#     # Plot trace plots for each parameter
#     for l in range(chains.shape[1]):
#         plt.figure(figsize=(10, 4))
#         for i in range(chains.shape[0]):
#             plt.plot(chains[i, l, :])
#         plt.xlabel('Sample')
#         plt.ylabel('Value')
#         plt.title(f'Trace Plot for variable {l}')
#         plt.legend([f'Chain {i+1}' for i in range(chains.shape[0])])
#         plt.show()

#     # Diagnostic plots
#     if diagnostic:
#         num_bins = 20
#         plt.figure()
#         for num in range(post.shape[0]):
#             bin_edges = np.linspace(np.min(post[num, :]), np.max(post[num, :]), num_bins + 1)
#             hist, _ = np.histogram(post[num, :], bins=bin_edges)
#             hist = hist / post.shape[1]
#             bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#             plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', label=f'Var {num + 1}')
#             plt.xlabel('Value')
#             plt.ylabel('Probability')
#             plt.title('Distribution')
#             plt.legend()
#             plt.show()

#         autocov = az.autocov(chains[:, 0, :])
#         ess = az.ess(chains[:, 0, :])
#         print(f"ESS values = {ess}")
#         plt.figure()
#         plt.plot(autocov[0, :])
#         plt.title('Autocovariance first chain')
#         plt.xlabel('Lag')
#         plt.ylabel('Autocovariance')
#         plt.legend()
#         plt.show()

#     return estimates

# @ray.remote
# def chain_creation_parallel(
#     N: int, 
#     burn_in: int, 
#     diagnostic: bool, 
#     algo: str, 
#     adapt: bool, 
#     scale: float,
#     posterior: Any, 
#     x_init: np.ndarray
# ) -> tuple:
#     """
#     Wrapper function to parallelize chain creation using Ray.

#     Args:
#         N (int): Number of samples.
#         burn_in (int): Number of burn-in samples.
#         diagnostic (bool): Whether to plot diagnostics.
#         algo (str): MCMC algorithm to use.
#         adapt (bool): Whether to use adaptive sampling.
#         scale (float): Scaling factor for MH algorithm.
#         posterior (Any): Posterior distribution to sample from.
#         x_init (np.ndarray): Initial parameter values.

#     Returns:
#         tuple: Estimates, chains, and posterior samples.
#     """

#     return chain_creation(N, burn_in, diagnostic, algo, adapt, scale, posterior, x_init)


# def chain_creation(
#     N: int, 
#     burn_in: int, 
#     diagnostic: bool, 
#     algo: str, 
#     adapt: bool, 
#     scale: float,
#     posterior: Any, 
#     x_init: np.ndarray
# ) -> tuple:
#     """
#     Create a chain using specified MCMC algorithm.

#     Args:
#         N (int): Number of samples.
#         burn_in (int): Number of burn-in samples.
#         diagnostic (bool): Whether to plot diagnostics.
#         algo (str): MCMC algorithm to use.
#         adapt (bool): Whether to use adaptive sampling.
#         scale (float): Scaling factor for MH algorithm.
#         posterior (Any): Posterior distribution to sample from.
#         x_init (np.ndarray): Initial parameter values.

#     Returns:
#         tuple: Estimates, chains, and posterior samples.
#     """
#     # Select the appropriate sampler
#     if algo == "NUTS":
#         sampler = NUTS(posterior, x0=x_init)
#     elif algo == "MH":
#         sampler = MH(posterior, scale=scale) if not adapt else MH(posterior)
#     elif algo == "pCN":
#         sampler = pCN(posterior, x0=x_init)
#     else:
#         raise ValueError(f"Unknown algorithm {algo}")

#     # Perform sampling
#     samples = sampler.sample_adapt(N - burn_in, burn_in) if adapt else sampler.sample(N - burn_in, burn_in)

#     # Compute estimates and store chains
#     estimates = samples.mean()[:, np.newaxis]
#     chains = np.expand_dims(samples.samples, axis=0)
#     post = samples.samples

#     print(f"Mean values: {estimates.mean(axis=1)}")

#     return (estimates, chains, post)

# # Plot Histogram
# def plot_hist(estimates: np.ndarray, real_x: np.ndarray, output1: np.ndarray, output2: np.ndarray) -> None:
#     """
#     Plot histogram comparing estimated and real values.

#     Args:
#         estimates (np.ndarray): Estimated values from the model.
#         real_x (np.ndarray): True values to compare against.
#         output1 (np.ndarray): First set of output values for comparison.
#         output2 (np.ndarray): Second set of output values for comparison.

#     Returns:
#         None
#     """
#     # Calculate the absolute differences
#     diff_output = np.abs(output1 - output2)
#     diff_value = np.abs(real_x - estimates)
#     print(f"The difference between estimated values {diff_value}\n")

#     # Stack real and estimated values for plotting
#     values = np.vstack((real_x, estimates))
#     categories = np.arange(1, values.shape[1] + 1)
#     bar_width = 0.35
#     bar_positions = [categories - bar_width / 2 + i * bar_width for i in range(values.shape[0])]

#     # Create the plot
#     plt.figure()
#     for i in range(values.shape[0]):
#         plt.bar(bar_positions[i], values[i, :], width=bar_width)

#     # Customize the plot
#     plt.ylabel('Value')
#     plt.title('Comparison of Real and Estimated Values')
#     plt.xticks(categories)
#     plt.legend(["Real value", "Estimate"])
#     plt.show()