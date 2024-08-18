import sys
import os
import numpy as np
from numpy import newaxis as _
import copy
import pickle
import time
from pprint import pprint
from numba import njit, jit

from keras.models import load_model
from typing import Callable, Tuple, Any, List, Optional, Union, Dict
from contextlib import contextmanager
from module_utils import *
from abc import ABC, abstractmethod
from functools import wraps
from enum import Enum
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import concurrent.futures
from keras.optimizers import Adam, Nadam, Adamax, RMSprop
from scipy.stats import multivariate_normal
from Helpers import *
from BIP_functions import *
from cuqi.distribution import Uniform, Gaussian,JointDistribution
from cuqi.sampler import MH
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Continuous2D, Discrete
import re
import tinyDA as tda
import optuna
from keras.src.callbacks.tensorboard import TensorBoard
import dask
from dask.distributed import Client
import multiprocessing
import joblib
from joblib import parallel_backend
import gc

############################
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure compatibility mode for deprecated functions
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
########################################

# GPU Optimization: Enable memory growth if using TensorFlow with GPUs.
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)



def check_observation_shape(y_obs: np.ndarray) -> Tuple[Optional[int], Optional[Any]]:
    """
    Check the dimensionality of the observation array and set geometry accordingly.

    Parameters:
    - y_obs: np.ndarray, the observation data

    Returns:
    - dim_obs: int or None, the dimensionality of the observation data (1D or 2D)
    - range_geometry: object or None, the corresponding geometry object for the observation data.
    """
    if len(y_obs.shape) == 1 or y_obs.shape[1] == 1:
        dim_obs = 1
        range_geometry = Continuous1D(y_obs.shape[0])
    elif y_obs.shape[1] == 2:
        dim_obs = 2
        range_geometry = Continuous2D(y_obs.shape[0])
    else:
        warnings.warn("Impossible for Cuqipy to manage a problem with 3 or more equations", UserWarning)
        return None, None
    return dim_obs, range_geometry


def initialize_model(forward_fn: Any, algo: str, range_geometry: Any, m: int) -> Any:
    """
    Initialize the CuqiModel based on the selected algorithm.

    Parameters:
    - forward_fn: function, the forward prediction function
    - algo: str, the selected MCMC algorithm ("MH" or "NUTS")
    - range_geometry: object, geometry describing the range of the model
    - m: int, the number of parameters to estimate

    Returns:
    - CuqiModel, the initialized CuqiModel object for the specified algorithm.
    """
    if algo == "NUTS":
        fun = Function(forward_fn)
        return CuqiModel(forward=forward_fn, jacobian=fun.compute_jacobian, 
                         range_geometry=range_geometry, domain_geometry=Discrete(m))
    else:
        return CuqiModel(forward=forward_fn, range_geometry=range_geometry, 
                         domain_geometry=Discrete(m))

@njit
def concatenate_inputs(inputs: np.ndarray, x_final: np.ndarray) -> np.ndarray:
    """
    Concatenate inputs and final processed data depending on dimensionality.

    Parameters:
    - inputs: np.ndarray, the initial input data (n, 1) or other shapes
    - x_final: np.ndarray, the final processed data to be concatenated with inputs
    
    Returns:
    - np.ndarray, concatenated input data after processing
    """
    # Handle different dimensionalities of the processed `x_final` data.
    if x_final.ndim == 2:
        # 2D Case: self.inputs (n, 1) and x_final (n, dim-1)
        concatenated_input = np.concatenate((inputs, x_final), axis=1)

    elif x_final.ndim == 3:
        # 3D Case: self.inputs (n, 1) and x_final (1, n, dim-1)
        inputs_expanded = np.expand_dims(inputs, axis=0)  
        concatenated_input = np.concatenate((inputs_expanded, x_final), axis=-1)

    else:
        # Raise an error if `x_final` has an unsupported number of dimensions.
        raise ValueError("Unsupported number of dimensions for x_final")
    
    return concatenated_input

@njit
def relative_error(estimates: np.ndarray, x_real: np.ndarray) -> np.ndarray:
    """
    Compute the relative error between estimated values and true values.

    Parameters:
    - estimates: np.ndarray, the estimated parameter values
    - x_real: np.ndarray, the true parameter values for comparison
    
    Returns:
    - np.ndarray, the relative error between estimates and true values.
    """
    return np.abs(estimates - x_real) / np.abs(x_real + 1e-10)

@njit
def calculate_metrics(output_test: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the Mean Squared Error (MSE) and R^2 score between the test data and predictions.

    Parameters:
    - output_test: np.ndarray, true output data
    - pred: np.ndarray, predicted output data
    
    Returns:
    - test_mse: float, Mean Squared Error between test and predicted outputs
    - r2: float, R^2 score indicating the proportion of variance explained by the model
    """
    test_mse = np.mean(np.square(output_test - pred))
    r2 = 1 - np.sum(np.square(output_test - pred)) / np.sum(np.square(output_test - np.mean(output_test)))
    return test_mse, r2

@jit
def apply_noise(y_obs: np.ndarray, cov_noise: float) -> np.ndarray:
    """
    Add noise to the observation data based on a specified covariance.

    Parameters:
    - y_obs: np.ndarray, the original observation data
    - cov_noise: float, the standard deviation of the noise to be added
    
    Returns:
    - np.ndarray, the perturbed observation data with added noise.
    """
    return y_obs + np.random.normal(loc=0.0, scale=cov_noise, size=y_obs.shape)


def setup_prior(mean_prior: np.ndarray, cov_prior: Optional[np.ndarray]) -> multivariate_normal:
    """
    Set up the prior distribution for Bayesian inference.

    Parameters:
    - mean_prior: np.ndarray, the mean vector for the prior distribution
    - cov_prior: Optional[np.ndarray], covariance matrix for the prior. If None, a default is used.
    
    Returns:
    - multivariate_normal, a multivariate normal distribution representing the prior.
    """
    if cov_prior is None:
        cov_prior = np.eye(len(mean_prior)) * 0.2  # Scale covariance
    return multivariate_normal(mean_prior, cov_prior)

def setup_likelihood(y_obs: np.ndarray, cov_likelihood: Optional[np.ndarray], cov_noise: float, dim: int) -> Any:
    """
    Set up the likelihood function for Bayesian inference.

    Parameters:
    - y_obs: np.ndarray, observed data
    - cov_likelihood: Optional[np.ndarray], covariance matrix for the likelihood. If None, a default is used.
    - cov_noise: float, standard deviation of the observation noise
    - dim: int, dimensionality of the observation data
    
    Returns:
    - tda.GaussianLogLike, the Gaussian likelihood function for the Bayesian model.
    """
    if cov_likelihood is None:
        cov_likelihood = cov_noise ** 2 * np.eye(dim)
    return tda.GaussianLogLike(y_obs, cov_likelihood)

def simulate_observations(x_real: np.ndarray, cov_noise: float, model_wrapper: Any) -> np.ndarray:
    """
    Simulate noisy observations using a given model wrapper.

    Parameters:
    - x_real: np.ndarray, the true parameter values
    - cov_noise: float, standard deviation of the noise to add to observations
    - model_wrapper: Any, a function or object that generates predictions based on the true parameters
    
    Returns:
    - np.ndarray, simulated observation data with added noise.
    """
    return model_wrapper(x_real) + np.random.normal(loc=0.0, scale=cov_noise, size=x_real.shape)


# Define the types of networks as an enumeration for type safety and clarity.
class NetworkType(Enum):
    LF = "LF"
    MF = "MF"
    HF = "HF"
    HFLIN = "Hflin"
    HFPER = "Hfper"
    SINGLE="Single"
    INTER = "Inter"
    LSTM = "LSTM"
    LSTM_SUPPORT="LSTM_support"
    LSTM_SUPPORT2="LSTM_support2"
    STEP = "step"


# Factory class to build different types of networks based on the provided type.
class NetworkFactory:

    @staticmethod
    def build_network(network_type: str,
                      names: List[str] = [],
                      params: Optional[dict] = None,
                      data_train: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                      output_train: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                      N: int = 1000,
                      n: int = 10,
                      train: bool = True,
                      do_HPO: bool = False,
                      verbose: bool = False,
                      device: str = '/CPU:0',
                      profiler: TensorBoard = None) -> 'INetwork':
        """
        Build and return a network of the specified type.

        Parameters:
        - network_type (str): Type of the network to be built.
        - names (List[str]): List of names used for MultiFidelity networks.
        - params (Optional[dict]): Dictionary of parameters for the network.
        - data_train (Optional[Union[np.ndarray, List[np.ndarray]]]): Training data.
        - output_train (Optional[Union[np.ndarray, List[np.ndarray]]]): Training output data.
        - N (int): Number of samples.
        - n (int): Some integer parameter.
        - train (bool): Flag indicating whether to train the network.
        - do_HPO (bool): Flag indicating whether to perform hyperparameter optimization.
        - verbose (bool): Flag indicating whether to print verbose output.
        - device (str): denotes GPU or CPU 
        - profiler (TensorBoard): allows to use the profiler to study the keras network performance

        Returns:
        - INetwork: The created network object.

        Raises:
        - ValueError: If an invalid network type is provided.
        """

        try:
            # Convert the network_type string to an enum for safer comparison and handling.
            # NetworkType is expected to be an Enum class that holds various types of networks.
            network_type_enum = NetworkType[network_type.upper()]
        except KeyError:
            # If the network type is not a recognized enum value, check if it matches an n-step pattern (e.g., "5STEP").
            match = re.match(r'(\d+)STEP', network_type, re.IGNORECASE)
            if match:
                # Extract the number from the pattern (e.g., 5 in "5STEP") and build a MultiFidelity network.
                n_step = int(match.group(1))
                return MultiFidelity(names, params, data_train, output_train, N, n, train,do_HPO, verbose, device=device, profiler=profiler)
            else:
                # If network type is neither a valid enum nor a recognizable pattern, raise an error.
                raise ValueError(f"Invalid network type: {network_type}")
        
        # Check if the network type is a low-fidelity, mid-fidelity, high-fidelity, or other similar neural network.
        if network_type_enum in {NetworkType.LF, NetworkType.MF, NetworkType.HF, NetworkType.HFLIN,NetworkType.SINGLE, NetworkType.HFPER}:
            # Build a neural network with the provided parameters and configurations.
            return Neural_Network(network_type, params, data_train, output_train, N, n, train, do_HPO, verbose, device=device, profiler=profiler)
        
        # Special case for step-based networks (could be related to MultiFidelity).
        elif network_type_enum == NetworkType["STEP"]:
            # Ensure data_train and output_train are both lists of numpy arrays for MultiFidelity networks.
            if not (isinstance(data_train, list) and isinstance(output_train, list)):
                raise ValueError("For MultiFidelity network, data_train and output_train must be lists of numpy arrays.")
            
            # Create a MultiFidelity network.
            return MultiFidelity(names, params, data_train, output_train, N, n,train, do_HPO, verbose, device=device, profiler=profiler)
        
        # Case for intermediate networks, such as transition networks between fidelity levels.
        elif network_type_enum == NetworkType.INTER:
            # Build and return an Intermediate network.
            return Intermediate(name=network_type, params=params, data_train=data_train, output_train=output_train, N=N, n=n, train=train, do_HPO=do_HPO, device=device)
        
        # Cases for LSTM-based networks, including regular LSTM and support LSTM types.
        elif network_type_enum == NetworkType.LSTM or network_type_enum == NetworkType.LSTM_SUPPORT  or network_type_enum == NetworkType.LSTM_SUPPORT2:
            # Build and return an LSTM network.
            return LSTM_network(name=network_type, params=params, data_train=data_train, output_train=output_train, N=N, train=train, do_HPO=do_HPO, verbose=verbose, device=device)

        # Raise an error if none of the valid cases match.
        raise ValueError(f"Invalid network type: {network_type}")




# Abstract base class for network-related operations.
class INetwork(ABC):
    """
    Abstract base class for defining a network interface with essential methods.
    This class uses the Abstract Base Class (ABC) mechanism to enforce the implementation
    of specific methods in any subclass.
    """

    def __init__(self):
        self.inputs = None
        # `transformations` is a list of transformations to be applied to the input data.
        self.transformations = []
        self._data_train = None
        self._output_train = None
        self._n=None
        self._N=None
        self._params=None

    def variable_input(self, input_discr: Any) -> None:
        """
        Sets the input variable when solving the inverse problem.
        
        Args:
            input_discr (Any): The input discriminator.
        """
        self.inputs = input_discr


    @property
    def data_train(self) -> np.ndarray:
        """
        Get the input training data.

        Returns:
        - np.ndarray: The current input training data.
        """
        return self._data_train

    @data_train.setter
    def data_train(self, data_train: np.ndarray) -> None:
        """
        Set the input training data for the network.

        Parameters:
        - data_train (np.ndarray): Input training data.
        """
        self._data_train = data_train

    @property
    def output_train(self) -> np.ndarray:
        """
        Get the expected output corresponding to the training data.

        Returns:
        - np.ndarray: The current output training data.
        """
        return self._output_train

    @output_train.setter
    def output_train(self, output_train: np.ndarray) -> None:
        """
        Set the expected output corresponding to the training data.

        Parameters:
        - output_train (np.ndarray): Expected output for training data.
        """
        self._output_train = output_train

    @property
    def params(self):
        """Returns the hyperparameters of the model."""
        return self._params

    @params.setter
    def params(self, value):
        """Sets the hyperparameters and prints them."""
        self._params = value
        self.print_params()

    def print_params(self):
        """Prints the model's hyperparameters in a readable format."""
        print("Model Hyperparameters:")
        for key, val in self._params.items():
            print(f"{key}: {val}")

    @property
    def N(self):
        """Returns the number of data points."""
        return self._N

    @N.setter
    def N(self, value):
        """Sets the number of data points."""
        if not isinstance(value,int) or value<0:
            raise ValueError('Number of epochs must be a positive integer!')
        self._N = value

    @property
    def n(self):
        """Returns the dimensionality of the inputs."""
        return self._n

    @n.setter
    def n(self, value):
        """Sets the dimensionality of the inputs."""
        if not isinstance(value,int) or value<0:
            raise ValueError('Batch size must be a positive integer!')

        self._n = value


    def _input_wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:
        """
        Prepare input data for prediction.

        Parameters:
        - x_test (np.ndarray): Test data for prediction.
        - multi_input (bool): Flag indicating if there are multiple inputs.

        Returns:
        - np.ndarray: Prepared data for prediction or an empty array if inputs are not set.
        """

        # Ensure that the input `x_test` is a 2D array.
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)

        # Transpose `x_test` to match the expected shape for further operations.
        x_test = x_test.T

        # Apply any transformations stored in `self.transformations`.
        if self.transformations:
            x_final = reduce(lambda acc, transf: np.hstack([acc, transf(acc)]), self.transformations, x_test)
        else:
            x_final = x_test

        # If the `inputs` attribute has been set tile the `x_final` array to match the number of samples in `self.inputs`.
        if self.inputs is not None:
            x_final = np.tile(x_final, (self.inputs.shape[0], 1))          
        else:
            # If `self.inputs` is not set, issue a warning and return an empty array.
            warning_message = "Inputs are not set."
            warnings.warn(warning_message, UserWarning)
            return np.array([])

        return x_final
    
    def _wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:
        """
        Wrapper for making predictions with processed test inputs.
        
        Args:
            x_test (np.ndarray): Test input data.
            multi_input (bool): Flag to indicate if multiple inputs are used.
        
        Returns:
            np.ndarray: Predicted output data.
        """

        # Process the `x_test` input data using `_input_wrapper_prediction`.
        x_final = self._input_wrapper_prediction(x_test, multi_input)

        concatenated_input = concatenate_inputs(self.inputs, x_final)

        # Use the subclass's `prediction` method to predict
 
        return self.prediction(concatenated_input).flatten()



    @compute_time
    def param_inverse(self, mean_prior: np.ndarray, 
                    x_data: np.ndarray, 
                    max_par: float, 
                    cov_prior: Optional[np.ndarray] = None, 
                    cov_noise: float = 0.1, 
                    cov_likelihood: Optional[np.ndarray] = None,
                    y_obs: Optional[np.ndarray] = None, 
                    x_real: Optional[np.ndarray] = None, 
                    number_chains: int = 1, 
                    N: int = 1000, 
                    burn_in: int = 500, 
                    levels: int = 1, 
                    diagnostic: bool = True, 
                    rwmh_cov: Optional[np.ndarray] = None, 
                    rmwh_scaling: float = 0.1, 
                    rwmh_adaptive: bool = True,
                    subsampling_rate: Union[int, List[int]] = 1,
                    algo: str = "MH",
                    force_sequential: bool = False,
                    transformation: List[Any] = []) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Perform parameter inversion using MCMC sampling.

        Parameters:
        - mean_prior (np.ndarray): Mean of the prior distribution.
        - x_data (np.ndarray): Input data for the model.
        - max_par (float): Maximum parameter value for estimation, used for plotting.
        - cov_prior (Optional[np.ndarray]): Covariance of the prior distribution.
        - cov_noise (float): Covariance of the noise in the observations.
        - cov_likelihood (Optional[np.ndarray]): Covariance matrix for the likelihood function.
        - y_obs (Optional[np.ndarray]): Observed data to match against the model.
        - x_real (Optional[np.ndarray]): True parameter values for error computation.
        - number_chains (int): Number of MCMC chains to run.
        - N (int): Total number of MCMC iterations.
        - burn_in (int): Number of burn-in iterations.
        - levels (int): Number of model levels (for multi-level modeling).
        - diagnostic (bool): Flag indicating whether to generate diagnostic plots.
        - rwmh_cov (Optional[np.ndarray]): Covariance matrix for the RWMH proposal distribution.
        - rmwh_scaling (float): Scaling factor for the RWMH proposal.
        - rwmh_adaptive (bool): Flag for enabling adaptive RWMH proposals.
        - subsampling_rate (Union[int, List[int]]): Rate or rates of subsampling the posterior. Default is 1.
        - algo (str): Algorithm to use for MCMC sampling ('MH', 'AM', 'CN', 'DREAMZ').
        - force_sequential (bool): Flag to enforce sequential processing of the MCMC algorithm.
        - transformation (List[Any]): List of transformations to apply to the input data.

        Returns:
        - Tuple[np.ndarray, np.ndarray, List[dict]]:
            - estimates (np.ndarray): Estimated parameters after MCMC sampling.
            - error (np.ndarray): Relative error of the estimates compared to `x_real`.
            - param_results (List[dict]): List of dictionaries containing details of the MCMC sampling results.
        """

        self.transformations = transformation
        self.inputs = x_data

        # Determine the dimensionality of the data.
        dim = (x_real.shape[0] if x_real is not None else (y_obs.shape[0] if y_obs is not None else None))
        if dim is None:
            warnings.warn("No observation nor data given", UserWarning)
            return np.array([]), np.array([])

        # Check if the number of iterations is sufficient (i.e., greater than the burn-in period).
        if N <= burn_in:
            warnings.warn("Number of steps insufficient, smaller or equal to burn-in", UserWarning)

        # Set up the prior distribution
        my_prior = setup_prior(mean_prior, cov_prior)

        # If `y_obs` is not provided, simulate it using predictions with added Gaussian noise.
        if y_obs is None:
            y_obs = simulate_observations(x_real, cov_noise, self._wrapper_prediction)
        else:
            y_obs = apply_noise(y_obs, cov_noise).flatten()

        # Multi-level model handling
        if levels > 1:
            if not hasattr(self, 'model_list'):
                warnings.warn("Single-level case considered", UserWarning)
            elif levels > len(self.model_list):
                raise ValueError("Number of levels exceeds available models")

            # Setup multi-level model hierarchy
            my_loglike = [setup_likelihood(y_obs, cov_likelihood, cov_noise, dim) for _ in range(levels)]
            my_posterior = [tda.Posterior(my_prior, my_loglike[i], self.model_list[i]._wrapper_prediction) for i in range(levels)]
        else:
            # Single level setup
            my_loglike = setup_likelihood(y_obs, cov_likelihood, cov_noise, dim)
            my_posterior = [tda.Posterior(my_prior, my_loglike, self._wrapper_prediction)]


        # Default to the identity matrix for the RWMH proposal covariance if none is provided.
        if rwmh_cov is None:
            rwmh_cov = np.eye(len(x_real))

        # Perform MCMC sampling, using the specified algorithm and settings.
        estimates, param_results = MCMC(
            my_posterior=my_posterior, 
            N=N, 
            burnin=burn_in, 
            n=number_chains, 
            diagnostic=diagnostic, 
            rwmh_cov=rwmh_cov, 
            rmwh_scaling=rmwh_scaling, 
            rwmh_adaptive=rwmh_adaptive, 
            algo=algo, 
            subsampling_rate=subsampling_rate,
            force_sequential=force_sequential,
            dim=dim
        )

        # Generate histograms for visualization of estimates.
        if diagnostic:
            self.plot_diagnostics(estimates, x_real, max_par)
        
        # Compute the relative error between estimates and x_real
        error = relative_error(estimates, x_real)
        
        return estimates, error, param_results
    

    @compute_time
    def inverse_cuqi(self,mean_prior: np.ndarray,                                    
                    x_data: np.ndarray,
                    max_par:float,
                    x_real: Optional[np.ndarray] = None, 
                    y_obs: Optional[np.ndarray] = None, 
                    N: int = 1000, 
                    burn_in: int = 500, 
                    cov_prior: float = 0.5, 
                    sd_noise: float = 0.1,
                    adapt: bool = False, 
                    scale: float = 0.3, 
                    proposal_sd: float = 0.3, 
                    x_init: Optional[Union[int, float, np.ndarray]] = None, #####
                    diagnostic: bool = True, 
                    number_chains: int = 1, 
                    algo: str = "MH", 
                    transformation: List = [], 
                    parallel: bool=False) -> Union[np.ndarray, float]:
        """
        Solves an inverse problem using a Bayesian framework with MCMC sampling.
        
        This function takes in prior knowledge, observation data, and other MCMC parameters to 
        estimate unknown parameters in an inverse problem setting. The chosen MCMC algorithm 
        (e.g., Metropolis-Hastings (MH) or No-U-Turn Sampler (NUTS)) is used to draw samples 
        from the posterior distribution, and the function calculates and returns the mean 
        parameter estimates along with the relative error.

        Parameters:
        - mean_prior: np.ndarray, mean of the prior distribution
        - x_data: np.ndarray, input data (independent variables)
        - max_par: float, maximum allowed parameter value for diagnostics
        - x_real: np.ndarray, the true values of the parameters for error computation
        - y_obs: np.ndarray, observed data (dependent variables)
        - N: int, number of MCMC iterations
        - burn_in: int, burn-in period for MCMC
        - cov_prior: float, covariance of the prior distribution
        - sd_noise: float, standard deviation of the noise to be added to observations
        - adapt: bool, whether to use adaptive scaling in the MCMC proposal
        - scale: float, scale factor for the proposal distribution
        - proposal_sd: float, standard deviation of the proposal distribution
        - x_init: np.ndarray or scalar, initial parameter values for MCMC (optional)
        - diagnostic: bool, whether to plot diagnostic results (default is True)
        - number_chains: int, number of MCMC chains (default is 1)
        - algo: str, MCMC algorithm to use, either "MH" for Metropolis-Hastings or "NUTS" for No-U-Turn Sampler
        - transformation: List, transformations applied to the parameters
        - parallel: bool, whether to run the chains in parallel
        
        Returns:
        - estimates: np.ndarray, the estimated parameter values
        - error: float, the relative error of the estimates compared to true values
        - parameters: np.ndarray, the full MCMC chain of parameter samples
        """
        # Set inputs and transformations
        self.inputs = x_data
        self.transformations = transformation

        # Check if the number of steps is greater than burn-in period
        if N <= burn_in:
            warning_message = "Number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning)
            return

        if y_obs is not None:
            dim_obs, range_geometry = check_observation_shape(y_obs)
            if dim_obs is None:
                return
        else:
            warnings.warn("No observation nor data given", UserWarning)
            return

        # Initialize model and Gaussian objects based on the chosen algorithm
        m = x_real.shape[0]
        A = initialize_model(self._wrapper_prediction, algo, range_geometry, m)


        # Initialize Gaussian distributions for the prior and the observation noise
        x = Gaussian(mean=mean_prior, cov=cov_prior)
        y = Gaussian(A(x), sqrtcov=proposal_sd)

        # Perturb observations
        y_obs = apply_noise(y_obs, sd_noise) # Add noise to provided observations

        # Run MCMC to get estimates
        estimates,parameters = MCMC_cuqi(y, x, y_obs, N, m, burn_in, number_chains, diagnostic=diagnostic, algo=algo, adapt=adapt, scale=scale, parallel=parallel)
        estimates = np.mean(estimates, axis=1)

        # Calculate and print error
        error =  relative_error(estimates, x_real)
        print(f"Error wrt true parameters: {error}")

        # Plot diagnostics if required
        if diagnostic:
            self.plot_diagnostics(estimates, x_real, max_par)

        return estimates, error,parameters


    def _plot_diagnostics(self,estimates, x_real, max_par):
        """
        Plot diagnostics to compare the estimates with the true values.
        """
        plot_hist(estimates, x_real, self._wrapper_prediction(estimates), self._wrapper_prediction(x_real), max_par)



    def performance(self, data_test: np.ndarray, output_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluates the performance of the model on test data.

        Args:
            data_test (np.ndarray): Test data.
            output_test (np.ndarray): Expected output data.

        Returns:
            Tuple[float, float]: Test Mean Squared Error (MSE) and R^2 score.
        """
        # Predict using the model
        pred = self.prediction(data_test)

        # Ensure the output shape matches the prediction shape
        if len(output_test.shape) < len(pred.shape):
            output_test = output_test[:, np.newaxis]
        
        # Calculate Mean Squared Error (MSE)
        test_mse, r2 = calculate_metrics(output_test, pred)
        print(f"Test MSE: {test_mse}")
        print(f"R^2: {r2}")
        
        return test_mse, r2


    @staticmethod
    def summary(self) -> None:
        """
        Prints the summary of the model architecture.
        """
        self.model.summary()

    @staticmethod
    def save(self, file_path: str) -> None: 
        """ 
        Save the trained Neural Network model to a file. 
        Args: 
            file_path (str): The path where the model will be saved. 
        """ 
        self.model.save(file_path) 
        print(f"Model saved to {file_path}") 

    def load(self, file_path: str) -> None: 
        """ 
        Load a trained Neural Network model from a file. 
        Args: 
            file_path (str): The path from where the model will be loaded. 
        """ 
        self.model = load_model(file_path, custom_objects={'FourierLayer': FourierLayer, 'custom_activation':custom_activation}) 
        self.input_shape=self.model.inputs[0][-1]
        self._output_shape=self.model.outputs[0][-1]

        print(f"Model loaded from {file_path}")

    @abstractmethod
    def prediction(self) -> None:
        """
        Abstract method to be implemented for making predictions using the network.
        Subclasses must provide the implementation for this method.
        """
        pass


    @abstractmethod
    def HPO(self) -> None:
        """
        Abstract method to be implemented for hyperparameter optimization.
        Subclasses must provide the implementation for this method.
        """
        pass

    @abstractmethod
    def training(self) -> None:
        """
        Abstract method to be implemented for training the network.
        Subclasses must provide the implementation for this method.
        """
        pass


class Neural_Network(INetwork):
    
    def __init__(
        self, 
        name: str, 
        params: Optional[Dict[str, Any]] = None, 
        data_train: Optional[np.ndarray] = None, 
        output_train: Optional[np.ndarray] = None, 
        N: int = 1000, 
        n: int = 10, 
        train: bool = True, 
        do_HPO: bool = False, 
        transformations: Optional[list] = None, 
        verbose: bool = False,
        device: str = None, 
        profiler: Optional[tf.keras.callbacks.TensorBoard] = None
    ):
        """
        Initializes the Neural_Network instance.
        
        Args:
            name (str): Name of the network.
            params (Optional[dict]): Hyperparameters of the network.
            data_train (Optional[np.ndarray]): Training data.
            output_train (Optional[np.ndarray]): Training outputs.
            N (int): Number of epochs for training.
            n (int): Batch size for training.
            train (bool): Flag to indicate if training should be performed.
            do_HPO (bool): Flag to indicate if hyperparameter optimization is to be performed.
            transformations (Optional[list]): List of transformations to apply to the data.
            verbose (bool): Flag to indicate verbosity of the output.
            device (str): Device for computation (e.g., '/CPU:0' or '/GPU:0').
            profiler (Optional[TensorBoard]): TensorBoard profiler for monitoring training.
        """
        # Clear any previous TensorFlow/Keras sessions to avoid clutter from old models.
        K.clear_session()

        # Initialize instance variables
        self.name = name
        self.type = "NN"
        self._params = params
        self._N = N
        self._n = n
        self.verbose = verbose
        self.hist = None
        self._data_train = data_train
        self._output_train = output_train
        self.transformations = transformations if transformations is not None else []
        self.inputs = None
        self.level = 0  # attribute used for MLDA in Bayesian inverse problems
        self.device = device

        # Set input and output shapes based on training data dimensions
        self.input_shape = self._get_shape(self._data_train)
        self.output_shape = self._get_shape(self._output_train)

        # Perform hyperparameter optimization if required or if no parameters are provided
        if do_HPO:
            if output_train is None or data_train is None:
                warning_message = "Not enough data given for HPO!"
                warnings.warn(warning_message, UserWarning)
            self._params = self.HPO(data_train, output_train)
            print("New parameters identified during HPO:")
            pprint(self._params)

        # Initialize the model with the given or optimized parameters
        self.model = getModel(self._params, self.input_shape, self.name, self.output_shape)

        # Train the model if required and if no HPO was performed
        if train and output_train is not None:
            self.hist = self.training(
                data_train, 
                output_train, 
                epoch=self._N, 
                batch=self._n, 
                device=device, 
                callbacks=profiler
            )
            # Plot training loss after training is complete
            self.plot_training_loss()


    # def __del__(self):
    #     """
    #     Destructor for the Neural_Network class.
    #     Ensures that TensorFlow sessions are cleared and memory is freed.
    #     """
    #     # Clear any TensorFlow sessions to free up GPU memory
    #     K.clear_session()

    #     # Run garbage collection to free up any remaining memory
    #     gc.collect()

    #     if self.verbose:
    #         print(f"{self.name} instance has been destroyed and resources have been freed.")


    def _get_shape(self, data: Optional[np.ndarray]) -> int:
        """
        Gets the shape of the data.

        Args:
            data (Optional[np.ndarray]): The data to get the shape of.

        Returns:
            int: The number of features in the data (second dimension).
        """
        return data.shape[1] if data is not None and len(data.shape) > 1 else 1

    def plot_training_loss(self) -> None:
        """
        Plots the training loss over the epochs.
        """
        if self.hist is not None and 'loss' in self.hist.history:
            plt.plot(self.hist.history['loss'][100:], label='Training Loss')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()
        else:
            warnings.warn("No training history or 'loss' key found.", UserWarning)


    @compute_time
    def training(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        epoch: int, 
        batch: int, 
        callbacks: Optional[tf.keras.callbacks.TensorBoard] = None
    ) -> Any:
        """
        Trains the model on the given data.
        
        Args:
            x (np.ndarray): Training data.
            y (np.ndarray): Training outputs.
            epoch (int): Number of epochs for training.
            batch (int): Batch size for training.
            callbacks (Optional[tf.keras.callbacks.TensorBoard]): TensorBoard profiler for monitoring training.

        Returns:
            Any: The training history.
        """

        # Enable mixed precision training for memory optimization
        if device.startswith('/GPU'):
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')

        self.hist = self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=0, callbacks=[callbacks] if callbacks else None)
        tf.keras.backend.clear_session()    
        
        # gc.collect()  # Explicit garbage collection after training to free memory

        return self.hist


    def prediction(self, x_test: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the test data.

        Args:
            x_test (np.ndarray): Test data.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.verbose:
            return self.model.predict(x_test)
        else:
            with Suppressor():  # Suppress output if verbosity is off
                return self.model.predict(x_test)

    def HPO(self, data_train: np.ndarray, output_train: np.ndarray) -> Dict[str, Any]:
        """
        Performs hyperparameter optimization using Bayesian optimization.

        Args:
            data_train (np.ndarray): Training data.
            output_train (np.ndarray): Training outputs.

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """

        def objective(trial):
            K.clear_session()
            tf.compat.v1.reset_default_graph()  # Ensure a clean graph for each trial
            
            # Suggest hyperparameters
            params = {
                "nodes": trial.suggest_int("nodes", 4, 64, log=True),
                "l2weight": trial.suggest_float("l2weight", 1e-4, 1e-1, log=True),
                "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                "kernel_init": trial.suggest_categorical("kernel_init", ["uniform", "glorot_uniform"]),
                "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),
            }

            # Training within HPO with device control
            with tf.device(self.device if self.device else '/GPU:0'):  # Default to GPU if not specified and an appropriate GPU is present                # Perform k-fold cross-validation to evaluate the model
                loss = kCrossVal_parallel(self._N, data_train, output_train, 
                                          params, self.name, self.input_shape, 
                                          self.output_shape
                                          )    
            # Implement early stopping within the HPO loop
            trial.report(loss, step=trial.number)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return loss

        # Set logging level to avoid TensorFlow warnings
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        tf.get_logger().setLevel('ERROR')

        # Create an Optuna study for optimization
        current_directory = os.getcwd()
        storage_path = os.path.join(current_directory, 'optuna_study.db')
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),storage=f'sqlite:///{storage_path}')

        # Optimize the objective function
        study.optimize(objective, n_trials=15, n_jobs=-1) # Bayesian optimization

        # Return the best hyperparameters found
        return study.best_params
        

    def _set_level(self, x_data: np.ndarray, prev_steps: List, level: int = 1) -> None:            
        """
        Sets the level of the model and stores the input data and previous steps. Usefull in a multifidelity scenario

        Args:
            x_data (np.ndarray): The input data for the model.
            prev_steps (List): A list of models that have been used in previous steps.
            level (int): The current level of the model (default is 1).
        """
        self.level = level
        self.inputs = x_data 
        self.prev_steps = prev_steps   # Stores the model list up to this point

    def _input_wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:     
        """
        Wraps the input for prediction, potentially combining it with previous model predictions.

        Args:
            x_test (np.ndarray): The test data for prediction.
            multi_input (bool): Flag indicating if the model expects multiple inputs.

        Returns:
            np.ndarray: The final input array for prediction.
        """
        x_final = super()._input_wrapper_prediction(x_test, multi_input)

        if self.level > 1:
            for l in range(0, self.level - 1):
                x_final = np.hstack((x_final, self.prev_steps[l].prediction(concatenate_inputs(self.inputs, x_final))))
        
        return x_final
    
class MultiFidelity(INetwork):
    """
    MultiFidelity network that allows for the sequential training and prediction of models
    at different levels of fidelity.
    """

    def __init__(
        self, 
        names: List[str], 
        params: Optional[List[dict]] = None, 
        data_train: Optional[List[np.ndarray]] = None, 
        output_train: Optional[List[np.ndarray]] = None, 
        N: Optional[List[int]] = None, 
        n: Optional[List[int]] = None, 
        train: bool = True,
        do_HPO: bool = False, 
        verbose: bool = False,
        device: str = None, 
        profiler: Optional[TensorBoard] = None
    ):
        """
        Initialize MultiFidelity network.

        Args:
            names (List[str]): Names of the networks.
            params (Optional[List[dict]]): Parameters for each network.
            data_train (Optional[List[np.ndarray]]): Training data for the networks.
            output_train (Optional[List[np.ndarray]]): Training outputs for each network.
            N (Optional[List[int]]): Number of epochs for each network.
            n (Optional[List[int]]): Batch sizes for each network.
            train (bool): Whether to train the network
            do_HPO (bool): Whether to perform hyperparameter optimization.
            verbose (bool): Whether to print verbose output.
            device (str): Device to use for training ('/CPU:0' or '/GPU:0').
            profiler (Optional[TensorBoard]): TensorBoard profiler for monitoring training.
        """
        # Initialize attributes
        self.names = names
        self._Ns = N if N is not None else [1000] * len(names)  # Set default epochs if not provided
        self._ns = n if n is not None else [10] * len(names)  # Set default batch sizes if not provided
        self._data_train = data_train
        self._output_train = output_train
        self.steps = int((len(names) - 1) / (len(data_train) - 1)) + 1  # Calculate number of steps
        self.model_list = []  # To store models for each fidelity level
        self.transformations = []  # Store transformations if any (currently not used)
        self.input_shape = 1
        self.output_shape = 1

        # Validate the provided training data and output data
        if not data_train or not output_train or len(data_train) != len(output_train):
            raise ValueError('The training and output data are either incoherent or insufficient.')

        # Set input and output shapes based on the first dataset
        if data_train and len(data_train[0].shape) > 1:
            self.input_shape = data_train[0].shape[1]
        if output_train and len(output_train[0].shape) > 1:
            self.output_shape = output_train[0].shape[1]

        # Clear the previous TensorFlow session
        K.clear_session()

        # Ensure `params` list is the same length as `names`, filling with `None` if necessary
        if params is None:
            params = [None] * len(names)
        elif len(params) < len(names):
            params += [None] * (len(names) - len(params))

        self._params=params
        # Initialize training data for the first model
        data_train_support = data_train[0]

        # Iterate over the network names to create and train each model
        for index, name in enumerate(names):
            # Build and train the network for the current fidelity level
            model = NetworkFactory.build_network(
                name,
                params=params[index],
                data_train=data_train_support,     # Use current training data
                output_train=output_train[index],  # Use corresponding output data
                N=self._Ns[index],
                n=self._ns[index],
                train=train,
                do_HPO=do_HPO,
                verbose=verbose,
                device=device,
                profiler=profiler
            )
            self.model_list.append(model)

            # Update training data for the next model, if any
            if (index + 1) < len(data_train) and train:
                data_train_support = data_train[index + 1]
                
                # Incorporate predictions from previous models into the training data
                for l in range(index + 1):
                    data_train_support = np.c_[data_train_support, self.model_list[l].prediction(data_train_support)]

    def plot_training_loss(self) -> None:
        """
        Plot training loss for all models in the MultiFidelity network.
        """
        for model in self.model_list:
            model.plot_training_loss()


    @property
    def params(self) -> List[dict]:
        """Get the hyperparameters of each network."""
        return [model.params for model in self.model_list]

    @params.setter
    def params(self, value: List[dict]) -> None:
        """Sets the hyperparameters, ensuring they are provided as a list."""
        if not isinstance(value, list):
            raise TypeError("Params must be a list.")
        
        if len(value) != len(self.model_list):
            raise ValueError("Number of parameter sets must be equal to the number of networks.")

        for i in range(len(self.model_list)):
            self.model_list[i].params = value[i]

    @property
    def N(self) -> List[int]:
        """Get the number of epochs for each network."""
        return [model.N for model in self.model_list]

    @N.setter
    def N(self, value: List[int]) -> None:
        """Sets the number of epochs, ensuring it's provided as a list."""
        if not isinstance(value, list):
            raise ValueError("N must be a list.")
        
        if len(value) != len(self.model_list):
            raise ValueError("Number of epochs must be equal to the number of networks.")

        for i in range(len(self.model_list)):
            self.model_list[i].Ns = value[i]

    @property
    def n(self) -> List[int]:
        """Get the batch sizes for each network."""
        return [model.n for model in self.model_list]

    @n.setter
    def n(self, value: List[int]) -> None:
        """Sets the batch sizes, ensuring they're provided as a list."""
        if not isinstance(value, list):
            raise ValueError("n must be a list.")
        
        if len(value) != len(self.model_list):
            raise ValueError("Number of batch sizes must be equal to the number of networks.")

        for i in range(len(self.model_list)):
            self.model_list[i].ns = value[i]

    def performance(self, data_test: np.ndarray, output_test: np.ndarray, position: Optional[int] = None) -> Tuple[float, float]:
        """
        Evaluate the performance of the model.

        Args:
            data_test (np.ndarray): Test data.
            output_test (np.ndarray): Expected output data.
            position (Optional[int]): Position of the model in the model list to evaluate.

        Returns:
            Tuple[float, float]: Test Mean Squared Error (MSE) and R^2 score.
        """
        # If no specific model is specified, evaluate the last model
        if position is None:
            position = len(self.model_list)
        elif not isinstance(position, int) or position > len(self.model_list):
            raise ValueError('The required model does not exist.')

        # Copy test data to avoid modifying the original input
        data = copy.copy(data_test)

        # Generate predictions sequentially for each preceding model up to the specified position
        for i in range(position - 1):
            pred = self.model_list[i].prediction(data)
            data = np.c_[data, pred.reshape(-1, 1)] if len(pred.shape) <= 1 else np.c_[data, pred]


        # Final prediction for the specified model
        pred = self.model_list[position - 1].prediction(data)

        # Adjust output shape if necessary
        output_test = output_test[:, np.newaxis] if len(output_test.shape) < len(pred.shape) else output_test


        test_mse, r2 = calculate_metrics(output_test, pred)
        print(f"Test MSE: {test_mse}")
        print(f"R^2: {r2}")

        return test_mse, r2
    
    def summary(self) -> None:
        """Print the summary of all models in the MultiFidelity network."""
        for Model in self.model_list:
            Model.summary()
        # free memory after summary
        gc.collect()

    def prediction(self, data_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the multi-fidelity model.

        Args:
            data_test (np.ndarray): Test data.

        Returns:
            np.ndarray: Predictions.
        """
        self.outputs = data_test

        # Reshape outputs if necessary
        if len(self.outputs.shape) == 1:
            self.outputs = self.outputs.reshape(-1, 1)

        # Sequentially generate predictions from each model and append to the output
        for index in range(len(self.names)):
            self.outputs = np.c_[self.outputs, self.model_list[index].prediction(self.outputs)]

        return self.outputs[:, -self.output_shape:]  # Return only the final model's prediction

    @staticmethod
    def save(self, file_paths: List[str]) -> None: 
        """ 
        Save the trained Neural networks model to a file. 
        Args: 
            file_paths (List[str]): The path where the model will be saved. 
        """ 

        for i,Model in enumerate(self.model_list):
            Model.save(file_paths[i])


    def load(self, file_paths: List[str]) -> None: 
        """ 
        Load a trained Neural networks model from a file. 
        Args: 
            file_paths (List[str]): The path from where the model will be loaded. 
        """
        self.model_list=[]

        for i in range(len(file_paths)):

            self.model_list.append(load_model(file_paths[i], custom_objects={'FourierLayer': FourierLayer, 'custom_activation':custom_activation}) )

        self.input_shape=self.model_list[0].inputs[0][-1]
        self.output_shape=self.model_list[-1].outputs[0][-1]

    def training(self,device: str = None, profiler: Optional[TensorBoard] = None):
        """
        Train all models in the MultiFidelity network.
        
        Args:
            device (str): Device to run the training on (e.g., '/CPU:0', '/GPU:0').
            profiler (Optional[TensorBoard]): Profiler for monitoring training performance.
        """
        if self._params==None:
            raise ValueError("params list is empty")
        
        if self._data_train==None or self._output_train==None:
            raise ValueError("Not enough data given")
        
        data_train_support = self._data_train[0]
                
        # Iterate over the network names to create and train each model
        for index, name in enumerate(self.names):
            # Build and train the network for the current fidelity level
            model = NetworkFactory.build_network(
                name,
                params=self._params[index],
                data_train=data_train_support,           # Use current training data
                output_train=self._output_train[index],  # Use corresponding output data
                N=self._Ns[index],
                n=self._ns[index],
                train=True,
                do_HPO=False,
                verbose=False,
                device=device,
                profiler=profiler
            )
            self.model_list.append(model)

            # Update training data for the next model, if any
            if (index + 1) < len(self._data_train):
                data_train_support = self._data_train[index + 1]
                
                # Incorporate predictions from previous models into the training data
                for l in range(index + 1):
                    data_train_support = np.c_[data_train_support, self.model_list[l].prediction(data_train_support)]
    
    def HPO(self,device: str = None, profiler: Optional[TensorBoard] = None):
        """
        Perform Hyperparameter Optimization (HPO) for all models in the MultiFidelity network.
        
        Args:
            device (str): Device to run the HPO on (e.g., '/CPU:0', '/GPU:0').
            profiler (Optional[TensorBoard]): Profiler for monitoring HPO performance.
        """
        if self._params==None:
            raise ValueError("params list is empty")
        
        if self._data_train==None or self._output_train==None:
            raise ValueError("Not enough data given")
    
        data_train_support = self._data_train[0]

        for index, name in enumerate(self.names):
            # Build and train the network for the current fidelity level
            model = NetworkFactory.build_network(
                name,
                params=self._params[index],
                data_train=data_train_support,  # Use current training data
                output_train=self._output_train[index],  # Use corresponding output data
                N=self._Ns[index],
                n=self._ns[index],
                do_HPO=True,
                verbose=False,
                device=device,
                profiler=profiler
            )
            self.model_list.append(model)

            # Update training data for the next model, if any
            if (index + 1) < len(self._data_train):
                data_train_support = self._data_train[index + 1]
                
                # Incorporate predictions from previous models into the training data
                for l in range(index + 1):
                    data_train_support = np.c_[data_train_support, self.model_list[l].prediction(data_train_support)]


    def get_output(self) -> np.ndarray:
        """
        Get the final output from the outputs.

        Returns:
            np.ndarray: The last output.
        """
        return self.model_list[-1].prediction(self._data_train[-1])


class LSTM_network(INetwork):

    def __init__(self, 
                 name: str = 'LSTM',
                 params: Optional[dict] = None, 
                 data_train: Optional[np.ndarray] = None, 
                 output_train: Optional[np.ndarray] = None, 
                 N: int = 1000, 
                 dim_input: int = 0,
                 dim_output: int = 0,
                 train: bool = True, 
                 do_HPO: bool = False, 
                 transformations: List[Any] = [], 
                 verbose: bool = False,
                 device: str = None):
        """
        Initialize LSTM_network instance.

        Args:
            name (str): Name of the model.
            params (Optional[dict]): Hyperparameters for the network.
            data_train (Optional[np.ndarray]): Training data.
            output_train (Optional[np.ndarray]): Training outputs.
            N (int): Number of epochs for training.
            dim_input (int): Dimension of input data.
            dim_output (int): Dimension of output data.
            train (bool): Flag to indicate if training should be performed.
            do_HPO (bool): Flag to indicate if hyperparameter optimization is to be performed.
            transformations (List[Any]): List of transformations to apply to the data.
            verbose (bool): Flag to indicate verbosity of the output.
            device (str): Device for computation (e.g., '/CPU:0' or '/GPU:0').
        """
        self.name = name
        self._params = params
        self._N = N
        self.verbose = verbose
        self.hist = None
        self._data_train = data_train
        self._output_train = output_train
        self.transformations = transformations
        self.input_shape = dim_input
        self.output_shape = dim_output
        self.inputs = None
        self.device = device

        # Determine input and output shapes
        if data_train is not None:
            self.input_shape = data_train.shape[-1] if len(data_train.shape) > 1 else self.input_shape
        if output_train is not None:
            self.output_shape = output_train.shape[-1] if len(output_train.shape) > 1 else self.output_shape

        # If training data is missing, disable training
        if data_train is None or output_train is None:
            train = False

        # Perform hyperparameter optimization if required       
        if do_HPO:
            if output_train is None or data_train is None:
                warnings.warn("Not enough data given!", UserWarning)
            self._params = self.HPO(data_train, output_train)
            print("New parameters identified during HPO:")
            pprint(self._params)

        # Warn if params are missing and training is required
        if self._params is None and train:
            warnings.warn("Params field is empty!", UserWarning)
        
        # Initialize the model
        self.model = getModel(self._params, self.input_shape, self.name, self.output_shape)

        # Train the model if training is enabled
        if train and output_train is not None:
            self.hist = self.training(
                seq_length=int(self._params['sequence_length']),
                seq_freq=int(self._params['sequence_freq']),
                epoch=self._N
            ) 
            self.plot_training_loss()
        else: 
            print(f"Class instance {self.name} created, load a Keras model")

    def set_parameters(self, params: Optional[dict]) -> None:
        """
        Set the hyperparameters for the LSTM network.

        Args:
            params (Optional[dict]): Hyperparameters to be set.
        """
        self._params = params

    @compute_time
    def training(self, seq_length: int, seq_freq: int, epoch: int):
        """
        Train the LSTM model.

        Args:
            seq_length (int): Sequence length for training.
            seq_freq (int): Sequence frequency for training.
            epoch (int): Number of epochs for training.
            device (str): Device for computation.

        Returns:
            tf.keras.callbacks.History: Training history.
        """
        self.sequence_length = seq_length
        self.sequence_freq = seq_freq
        self.input_train_seq, self.output_train_seq = self._sliding_windows(
            self._data_train, self._output_train, self.sequence_length, self.sequence_freq)

        # Early stopping callback
        callback = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=self._params['patience'], restore_best_weights=True)

        # Enable mixed precision training for memory optimization
        if device.startswith('/GPU'):
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')

        # Set random seed for reproducibility
        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism()

        # Train the model, with device control
        with tf.device(self.device if self.device else '/GPU:0'):  # Default to GPU if not specified and an appropriate GPU is present
            self.hist = self.model.fit(
                self.input_train_seq, self.output_train_seq, epochs=epoch, 
                verbose=self.verbose, callbacks=[callback]
            )
        tf.keras.backend.clear_session()
        return self.hist

    def plot_training_loss(self) -> None:
        """
        Plots the training loss over epochs.
        """
        if self.hist is not None and 'loss' in self.hist.history:
            plt.plot(self.hist.history['loss'][100:], label='Training Loss')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()

    def prediction(self, x_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the LSTM model.

        Args:
            x_test (np.ndarray): Test data.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.verbose:
            y_pred = self.model.predict(x_test)
        else:
            with Suppressor():
                y_pred = self.model.predict(x_test)
        return y_pred

    @staticmethod
    @njit
    def _sliding_windows(data_input: np.ndarray, data_output: np.ndarray, seq_length: int, freq: int = 1) -> Tuple[np.ndarray,np.ndarray]:
        """
        Generates sliding windows for the given data and labels.

        Args:
            data_input (np.ndarray): Input data.
            data_output (np.ndarray): Output data.
            seq_length (int): Length of each sequence.
            freq (int): Frequency of each sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Input sequences and corresponding output sequences.
        """
        x, y = [], []

        for i in range(data_input.shape[0]):
            for j in range(0, data_input.shape[1] - seq_length, freq):
                _x = data_input[i, j:(j + seq_length), :]
                _y = data_output[i, j:(j + seq_length), :]
                x.append(_x)
                y.append(_y)

        return np.array(x), np.array(y)

    def HPO(self, data_train: np.ndarray, output_train: np.ndarray) -> Dict[str, Any]:
        """
        Performs hyperparameter optimization using Bayesian optimization.

        Args:
            data_train (np.ndarray): Training data.
            output_train (np.ndarray): Training outputs.
            device (str): Device for computation.

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """
        def objective(trial):
            K.clear_session()
            params = {
                "nodes": trial.suggest_int("nodes", 4, 64, log=True),
                "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),
                "sequence_length": trial.suggest_int("sequence_length", 10, 100),
                "sequence_freq": trial.suggest_int("sequence_freq", 2, 10),
                "patience": trial.suggest_int("patience", 50, 100),
                "lay": trial.suggest_int("lay", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.05, 0.5, log=True)
            }
            
            # Training within HPO with device control
            with tf.device(self.device if self.device else '/GPU:0'):  # Default to GPU if not specified and an appropriate GPU is present
                loss = kCrossVal_parallel(
                    Nepo=self._N, x=data_train, y=output_train, 
                    params=params, name=self.name, input_shape=self.input_shape, 
                    output_shape=self.output_shape, p=5, n_jobs=-1
                )

            # Implement early stopping within the HPO loop
            trial.report(loss, step=trial.number)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return loss

        # Optimize using Optuna
        # Create an Optuna study for optimization
        current_directory = os.getcwd()
        storage_path = os.path.join(current_directory, 'optuna_study.db')
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),storage=f'sqlite:///{storage_path}')
        study.optimize(objective, n_trials=10, n_jobs=-1) # Bayesian optimization
        best_params = study.best_params
        return best_params

    def param_inverse(  self, 
                        mean_prior: np.ndarray, 
                        x_data: np.ndarray,
                        max_par:float, 
                        cov_prior: Optional[np.ndarray] = None, 
                        cov_noise: float = 0.1, 
                        cov_likelihood: Optional[np.ndarray] = None, 
                        y_obs: Optional[np.ndarray] = None, 
                        x_real: Optional[np.ndarray] = None, 
                        number_chains: int = 1, 
                        N: int = 1000, 
                        burn_in: int = 500, 
                        levels: int = 1, 
                        diagnostic: bool = True, 
                        rwmh_cov: Optional[np.ndarray] = None, 
                        rmwh_scaling: float = 0.1, 
                        rwmh_adaptive: bool = True, 
                        algo: str = "MH", 
                        transformation: List[Any] = [], 
                        forward_low_fidelity: Optional[Callable] = None, 
                        force_sequential: bool = False, 
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform parameter inversion using MCMC sampling, potentially utilizing a low fidelity forward model.

        Parameters:
        - mean_prior (np.ndarray): Mean of the prior distribution.
        - x_data_support (np.ndarray): Input data for the basis reduction.
        - x_data (np.ndarray): Input data.
        - cov_prior (Optional[np.ndarray]): Covariance of the prior distribution (optional).
        - cov_noise (float): Noise covariance.
        - cov_likelihood (Optional[np.ndarray]): Covariance of the likelihood (optional).
        - y_obs (Optional[np.ndarray]): Observed data (optional).
        - x_real (Optional[np.ndarray]): Real parameters (optional).
        - number_chains (int): Number of MCMC chains.
        - N (int): Number of MCMC iterations.
        - burn_in (int): Number of burn-in iterations.
        - levels (int): Number of model levels.
        - diagnostic (bool): Flag to enable diagnostic plots.
        - rwmh_cov (Optional[np.ndarray]): Covariance matrix for RWMH proposal (optional).
        - rmwh_scaling (float): Scaling factor for RWMH.
        - rwmh_adaptive (bool): Flag for adaptive RWMH.
        - algo (str): MCMC algorithm to use ("MH", "AM", "CN", "DREAMZ").
        - transformation (List[Any]): List of transformations to apply.
        - forward_low_fidelity (Optional[Callable]): Low fidelity forward model function (optional).
        - force_sequential (bool): True to avoid parallelization  
        - *args: Additional positional arguments used depending on the class with the data of the LSTM model

        Returns:
        - Tuple[np.ndarray, np.ndarray]: MCMC estimates of the parameters and relative error of the estimates.
        """

        # Check if forward_low_fidelity is provided and is a callable function
        if forward_low_fidelity is None or not callable(forward_low_fidelity):
            raise KeyError("Provide forward_low_fidelity related to the class of the problem. It must be a callable function")

        self.input_support = next(iter(kwargs.values()))
        # Store the forward_low_fidelity function for later use
        self._forward_low_fidelity = forward_low_fidelity
        # Call the parent class's param_inverse method with the provided parameters
        return super().param_inverse(
            mean_prior=mean_prior, 
            x_data=x_data, 
            max_par=max_par,
            cov_prior=cov_prior, 
            cov_noise=cov_noise, 
            cov_likelihood=cov_likelihood, 
            y_obs=y_obs, 
            x_real=x_real, 
            number_chains=number_chains, 
            N=N, 
            burn_in=burn_in, 
            levels=levels, 
            diagnostic=diagnostic, 
            rwmh_cov=rwmh_cov, 
            rmwh_scaling=rmwh_scaling, 
            rwmh_adaptive=rwmh_adaptive, 
            algo=algo, 
            transformation=transformation,
            force_sequential = force_sequential
        )

    def _input_wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:
        """
        Wrap the input for prediction by applying low-fidelity forward modeling.

        Args:
            x_test (np.ndarray): Test data.
            multi_input (bool): Flag indicating if multiple inputs are used.

        Returns:
            np.ndarray: Transformed prediction input.
        """
        # Call the parent class's _input_wrapper_prediction method
        x_final = super()._input_wrapper_prediction(x_test, multi_input)

        # Apply forward low-fidelity modeling to the wrapped input
        prediction_input = self._forward_low_fidelity(x_final, self.inputs, self.input_support)

        return prediction_input#self.prediction(prediction_input).flatten()
        
    
class Intermediate(INetwork):

    def __init__(self, 
                 name: str = "Inter", 
                 params: Optional[dict] = None, 
                 data_train: Optional[np.ndarray] = None, 
                 output_train: Optional[np.ndarray] = None, 
                 N: int = 1000, 
                 n: int = 10, 
                 dim_input:int=0,
                 dim_output:int=0,
                 train: bool = True, 
                 do_HPO: bool = False, 
                 transformations: List[Any] = [], 
                 verbose: bool = False,
                 device:str ='/CPU:0'):
        """
        Initialize Intermediate network.

        Args:
            name (str): Name of the network.
            params (Optional[dict]): Parameters for the network.
            data_train (Optional[np.ndarray]): Training data for the network.
            output_train (Optional[np.ndarray]): Training outputs for the network.
            N (int): Number of epochs for training.
            n (int): Batch size for training.
            train (bool): Whether to train the model.
            do_HPO (bool): Whether to perform hyperparameter optimization.
            transformations (Optional[list]): List of transformations to apply to the data.
            verbose (bool): Whether to print verbose output.
        """
        self.name = name
        self._params = params
        self._N = N
        self._n = n
        self.verbose = verbose
        self.hist = None
        self._data_train = data_train
        self._output_train = output_train
        self.input_shape = 1
        self.output_shape = 1
        self.inputs = None
        self.transformations = transformations

        # Validate and concatenate data
        if data_train is None or output_train is None or len(data_train) != 2 or len(output_train) != 2:
            raise ValueError('The data are incoherent or insufficient')
                
        data_train = np.concatenate((data_train[1], data_train[0]), axis=0)
        output_train = np.concatenate((output_train[1], output_train[0]), axis=0)

        # Set input and output shapes
        self.input_shape = self._get_shape(data_train)
        self.output_shape = self._get_shape(output_train)

        if do_HPO or params is None:
            if output_train is None or data_train is None:
                warnings.warn("Not enough data given!", UserWarning)
            self._params = self.HPO(data_train, output_train,device=device)

        # Create the model
        self.model = getModel(self._params, self.input_shape, self.name, self.output_shape)

        if train:
            self.hist = self.training(data_train, output_train, epoch=self._N, batch=self._n,device=device) 
            self.plot_training_loss()

    def _get_shape(self, data: Optional[np.ndarray]) -> int:
        """
        Gets the shape of the data.

        Args:
            data (Optional[np.ndarray]): The data to get the shape of.

        Returns:
            int: The shape of the data.
        """
        return data.shape[1] if data is not None and len(data.shape) > 1 else 1

    def plot_training_loss(self) -> None:
        """
        Plot the training loss.
        """
        if self.hist is not None and 'loss' in self.hist.history:
            plt.plot(self.hist.history['loss'][100:], label='Training Loss')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()
        else:
            warnings.warn("No training history or 'loss' key found.", UserWarning)

    @compute_time
    def training(self, x: np.ndarray, y: np.ndarray, epoch: int, batch: int, device: str = '/CPU:0', callbacks: TensorBoard=None) -> Any:
        """
        Trains the model on the given data.
        
        Args:
            x (np.ndarray): Training data.
            y (np.ndarray): Training outputs.
            epoch (int): Number of epochs for training.
            batch (int): Batch size for training.

        Returns:
            Any: The training history.
        """
        if callbacks is not None:
            with tf.device(device):

                self.hist = self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=0, callbacks=[callbacks])
        else:
            with tf.device(device):

                self.hist = self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=0)
        
        return self.hist
    
    def prediction(self, x_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model.

        Args:
            x_test (np.ndarray): Test data.

        Returns:
            np.ndarray: Predictions.
        """
        if self.verbose:
            return self.model.predict(x_test)
        else:
            with Suppressor():
                return self.model.predict(x_test)

  

    def HPO(self, data_train: np.ndarray, output_train: np.ndarray, device: str = '/CPU:0') -> Dict[str, Any]:
        """
        Performs hyperparameter optimization using Bayesian optimization.

        Args:
            data_train (np.ndarray): Training data.
            output_train (np.ndarray): Training outputs.

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """
        print("HPO name ", self.name )
        def objective(trial):
            K.clear_session()
            tf.compat.v1.reset_default_graph()  # Ensure clean graph for each trial
            
            params = {
                "nodes": trial.suggest_int("nodes", 4, 64, log=True),
                "l2weight": trial.suggest_float("l2weight", 1e-4, 1e-1, log=True),
                "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                "kernel_init": trial.suggest_categorical("kernel_init", ["uniform", "glorot_uniform"]),
                "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),
            }

            with tf.device(device):
                #loss = kCrossVal(self.n, self.N, data_train, output_train, params, self.name, self.input_shape, self.output_shape)
                loss = kCrossVal_parallel(self._N, data_train, output_train, params, self.name, self.input_shape, self.output_shape)            
            return loss

        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        tf.get_logger().setLevel('ERROR')
        
        
        # Create an Optuna study for optimization
        current_directory = os.getcwd()
        storage_path = os.path.join(current_directory, 'optuna_study.db')
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),storage=f'sqlite:///{storage_path}')

        # Use n_jobs=2 since it is stable on your system
        study.optimize(objective, n_trials=4, n_jobs=-1)
        
        best_params = study.best_params
        return best_params
    

    def _input_wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:    # check
        x_final = super()._input_wrapper_prediction(x_test, multi_input)

        if self.level >1 :

            for l in range(0,self.level-1):
                x_final = np.concatenate((x_final, self.prev_steps[l]._wrapper_prediction(x_final, multi_input)), axis=1)
        
        return x_final

