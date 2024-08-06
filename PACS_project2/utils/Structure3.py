import sys
import os
import numpy as np
from numpy import newaxis as _
import copy

import time
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

from cuqi.distribution import Uniform, Gaussian,JointDistribution
from cuqi.sampler import MH
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Discrete
import re

import tinyDA as tda
import optuna
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




# Function to load context functions from a specified folder
def load_context_functions(context_folder: str) -> bool:
    """
    Adds the specified context folder to the system path.

    Parameters:
    - context_folder: Path to the folder containing context functions.

    Returns:
    - True if the folder is successfully added to the system path.
    - False if there is an exception.
    """
    try:
        sys.path.append(context_folder)
        return True
    except Exception as e:
        print(f"Error loading context folder {context_folder}: {e}")
        return False

def compute_time(func: Callable) -> Callable:
    """
    Decorator to measure and print the execution time of a function.

    Parameters:
    - func: The function to be measured.

    Returns:
    - The wrapper function that prints the execution time.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The function {func.__name__} took {elapsed_time:.6f} seconds.")
        return result
    return wrapper

@contextmanager
def Suppressor() -> None:
    """
    Context manager to suppress stdout output using contextlib.
    Redirects stdout to os.devnull during its context.
    
    Usage:
    with suppress_output():
        # Code that produces output
    """
    original_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


# Class to handle a function and compute its Jacobian matrix
class Function:
    def __init__(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Initialize the class with a given function.

        Parameters:
        - func: The function for which we want to compute the Jacobian.
        """
        self.func = func

    def compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix for the given point x.

        Parameters:
        - x: The point at which to compute the Jacobian.

        Returns:
        - The Jacobian matrix.
        """
        x = np.atleast_1d(x).astype(float)  # Ensure x is a 1D numpy array of floats
        n = len(x)  # Number of variables in x
        f_x = np.atleast_1d(self.func(x))  # Function value at x, ensured to be a numpy array
        m = len(f_x)  # Number of functions (length of output of func)

        jacobian = np.zeros((m, n))  # Initialize the Jacobian matrix with zeros
        h = 1e-8  # Small perturbation for numerical differentiation
        for i in range(n):
            x_pos = x.copy()
            x_neg = x.copy()

            # Perturb the i-th element positively and negatively
            x_pos[i] += h
            x_neg[i] -= h

            f_pos = np.atleast_1d(self.func(x_pos))  # Function value at x + h
            f_neg = np.atleast_1d(self.func(x_neg))  # Function value at x - h

            # Compute partial derivative using central difference
            jacobian[:, i] = (f_pos - f_neg) / (2 * h)

        return jacobian



def sliding_windows(data_input: np.ndarray, data_output: np.ndarray, seq_length: int, freq: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sliding windows of sequences from the input and output data.

    Parameters:
    - data_input (np.ndarray): Input data array of shape (num_samples, num_timesteps, num_features).
    - data_output (np.ndarray): Output data array of shape (num_samples, num_timesteps, num_output_features).
    - seq_length (int): Length of each sequence window.
    - freq (int): Frequency of sliding. Defaults to 1.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Arrays of input and output windows.
    """
    # Extract the dimensions of the input data
    num_samples, num_timesteps, num_features = data_input.shape
    _, _, num_output_features = data_output.shape

    # Check if the sequence length is valid
    if seq_length > num_timesteps:
        raise ValueError("Sequence length cannot be greater than the number of timesteps in the data.")

    # Calculate the number of windows per sample
    windows_per_sample = (num_timesteps - seq_length) // freq + 1

    # Calculate the total number of windows
    total_windows = num_samples * windows_per_sample

    # Preallocate the arrays for input and output windows
    input_windows = np.empty((total_windows, seq_length, num_features), dtype=data_input.dtype)
    output_windows = np.empty((total_windows, seq_length, num_output_features), dtype=data_output.dtype)

    window_index = 0
    for sample_idx in range(num_samples):
        for time_idx in range(0, num_timesteps - seq_length + 1, freq):
            input_windows[window_index] = data_input[sample_idx, time_idx:time_idx + seq_length]
            output_windows[window_index] = data_output[sample_idx, time_idx:time_idx + seq_length]
            window_index += 1

    return input_windows, output_windows

def add_noise(noise_std_data: np.ndarray, 
              noise_std_output: np.ndarray, 
              data: np.ndarray, 
              output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian noise to data and output arrays and concatenate the results.

    Parameters:
    - noise_std_data (np.ndarray): Standard deviations for the noise to be added to the data.
    - noise_std_output (np.ndarray): Standard deviations for the noise to be added to the output.
    - data (np.ndarray): The input data array.
    - output (np.ndarray): The output data array.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Tuple containing the concatenated noisy data and output arrays.
    """
    # Initialize the flags with the original data and output arrays
    output_flag = output
    data_flag = data
    
    # Loop over the standard deviations and add noise
    for std_data, std_output in zip(noise_std_data, noise_std_output):
        # Generate Gaussian noise for data and output
        noise_data = np.random.normal(0, std_data, data.shape)
        noise_output = np.random.normal(0, std_output, output.shape)
        
        # Add noise to the original data and output
        noisy_data = data + noise_data
        noisy_output = output + noise_output
        
        # Concatenate the noisy data and output to the flags
        data_flag = np.concatenate((data_flag, noisy_data), axis=0)
        output_flag = np.concatenate((output_flag, noisy_output), axis=0)
    
    return output_flag, data_flag




# Define the types of networks as an enumeration for type safety and clarity.
class NetworkType(Enum):
    LF = "LF"
    MF = "MF"
    HF = "HF"
    HFLIN = "Hflin"
    HFPER = "Hfper"
    INTER = "Inter"
    LSTM = "LSTM"
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
                      verbose: bool = False) -> 'INetwork':
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

        Returns:
        - INetwork: The created network object.

        Raises:
        - ValueError: If an invalid network type is provided.
        """
        try:
            # Use Enum for safer and clearer type checking.
            network_type_enum = NetworkType[network_type.upper()]
        except KeyError:
            # Check if it's an n-step network type using regex
            match = re.match(r'(\d+)STEP', network_type, re.IGNORECASE)
            if match:
                n_step = int(match.group(1))
                return MultiFidelity(names, params, data_train, output_train, N, n, do_HPO, verbose)
            else:
                raise ValueError(f"Invalid network type: {network_type}")
        # Map the enum to the respective network class constructors.
        if network_type_enum in {NetworkType.LF, NetworkType.MF, NetworkType.HF, NetworkType.HFLIN, NetworkType.HFPER}:
            return Neural_Network(network_type, params, data_train, output_train, N, n, train, do_HPO, verbose)
        
        elif  network_type_enum ==  NetworkType["STEP"]: #network_type_enum == NetworkType.step:
            # Ensure data_train and output_train are lists for MultiFidelity networks.
            if not (isinstance(data_train, list) and isinstance(output_train, list)):
                raise ValueError("For MultiFidelity network, data_train and output_train must be lists of numpy arrays.")
            return MultiFidelity(names, params, data_train, output_train, N, n, do_HPO, verbose)
        
        elif network_type_enum == NetworkType.INTER:
            return Intermediate(params, data_train, output_train, N, n, train, do_HPO, verbose)
        
        elif network_type_enum == NetworkType.LSTM:
            return LSTM_network(params, data_train, output_train, N, train, do_HPO, verbose)

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
        self.transformations = []

    def variable_input(self, input_discr: Any) -> None:
        """
        Sets the input variable when solving the inverse problem.
        
        Args:
            input_discr (Any): The input discriminator.
        """
        self.inputs = input_discr

    def _input_wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:
        """
        Prepare input data for prediction.

        Parameters:
        - x_test (np.ndarray): Test data for prediction.
        - multi_input (bool): Flag indicating if there are multiple inputs.

        Returns:
        - np.ndarray: Prepared data for prediction or an empty array if inputs are not set.
        """
        # Ensure x_test is 2D.
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)

        # Transpose x_test for consistent shape.
        x_test = x_test.T

        # Apply transformations if any.
        if self.transformations:
            x_final = reduce(lambda acc, transf: np.hstack([acc, transf(acc)]), self.transformations, x_test)
        else:
            x_final = x_test

        # Handle inputs and prepare final data for prediction.
        if self.inputs is not None:
            x_final = np.tile(x_final, (self.inputs.shape[0], 1))          # [0]?
        else:
            warning_message = "Inputs are not set."
            warnings.warn(warning_message, UserWarning)
            return np.array([])

        return x_final

    def wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:
        """
        Wrapper for making predictions with processed test inputs.
        
        Args:
            x_test (np.ndarray): Test input data.
            multi_input (bool): Flag to indicate if multiple inputs are used.
        
        Returns:
            np.ndarray: Predicted output data.
        """
        x_final = self._input_wrapper_prediction(x_test, multi_input)
        return self.prediction(np.concatenate((self.inputs, x_final), axis=1)).flatten()      

    @compute_time
    def param_inverse(self, mean_prior: np.ndarray, x_data: np.ndarray, cov_prior: Optional[np.ndarray] = None, 
                      cov_noise: float = 0.1, cov_likelihood: Optional[np.ndarray] = None, y_obs: Optional[np.ndarray] = None, 
                      x_real: Optional[np.ndarray] = None, number_chains: int = 1, N: int = 1000, burn_in: int = 500, 
                      levels: int = 1, diagnostic: bool = True, rwmh_cov: Optional[np.ndarray] = None, rmwh_scaling: float = 0.1, 
                      rwmh_adaptive: bool = True, algo: str = "MH", transformation: List[Any] = []) -> Tuple[np.ndarray, np.ndarray]:

        """
        Perform parameter inversion using MCMC sampling.

        Parameters:
        - mean_prior: Mean of the prior distribution.
        - x_data: Input data.
        - cov_prior: Covariance of the prior distribution (optional).
        - cov_noise: Noise covariance.
        - cov_likelihood: Covariance of the likelihood (optional).
        - y_obs: Observed data (optional).
        - x_real: Real parameters (optional).
        - number_chains: Number of MCMC chains.
        - N: Number of MCMC iterations.
        - burn_in: Number of burn-in iterations.
        - levels: Number of model levels.
        - diagnostic: Flag to enable diagnostic plots.
        - rwmh_cov: Covariance matrix for RWMH proposal (optional).
        - rmwh_scaling: Scaling factor for RWMH.
        - rwmh_adaptive: Flag for adaptive RWMH.
        - algo: MCMC algorithm to use ("MH", "AM", "CN", "DREAMZ").
        - transformation: List of transformations to apply.

        Returns:
        - estimates: MCMC estimates of the parameters.
        - error: Relative error of the estimates.
        """
        self.transformations = transformation
        self.inputs = x_data

        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim = y_obs.shape[0]
        else:
            warnings.warn("No observation nor data given", UserWarning)
            return np.array([]), np.array([])

        if N <= burn_in:
            warnings.warn("Number of steps insufficient, smaller or equal to burn-in", UserWarning)

        if cov_prior is None:
            cov_prior = mean_prior * 0.2

        if cov_likelihood is None:
            cov_likelihood = cov_noise**2 * np.eye(x_real.shape[0])

        my_prior = multivariate_normal(mean_prior, cov_prior)

        if y_obs is None:
            y_obs = self.wrapper_prediction(x_real) + np.random.normal(loc=0., scale=cov_noise, size=x_real.shape)
        else:
            y_obs += np.random.normal(loc=0., scale=cov_noise, size=y_obs.shape)
            y_obs = y_obs.flatten()

        if levels > 1:
            if levels > len(self.model_list):
                warnings.warn("Number of levels is exceeding the number of models", UserWarning)
            else:
                my_loglike = [tda.GaussianLogLike(y_obs, cov_likelihood) for _ in range(levels)]
                my_posterior = [tda.Posterior(my_prior, my_loglike[i], self.model_list[i].wrapper_prediction) for i in range(levels)]
        else:
            my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
            my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)

        if rwmh_cov is None:
            rwmh_cov = np.eye(len(x_real))

        estimates = MCMC(my_posterior, N, burn_in, number_chains, diagnostic, rwmh_cov, rmwh_scaling, rwmh_adaptive, algo, dim)

        if diagnostic:
            plot_hist(estimates, x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        error = np.abs(estimates - x_real) / np.abs(x_real + 1e-10)
        return estimates, error    

    @compute_time
    def inverse_cuqi(self,mean_prior: np.ndarray, 
                    x_data: np.ndarray,
                    x_real: Optional[np.ndarray] = None, 
                    y_obs: Optional[np.ndarray] = None, 
                    N: int = 1000, 
                    burn_in: int = 500, 
                    cov_prior: float = 0.5, 
                    sd_noise: float = 0.1,
                    adapt: bool = False, 
                    scale: float = 0.3, 
                    proposal_sd: float = 0.3, 
                    x_init: Optional[Union[int, float, np.ndarray]] = None, 
                    diagnostic: bool = True, 
                    number_chains: int = 1, 
                    algo: str = "MH", 
                    transformation: List = [], parallel: bool=False) -> Union[np.ndarray, float]:
        """
        Perform Bayesian inference using the CUQI framework.

        Parameters:
        - mean_prior: np.ndarray : Prior mean
        - x_data: np.ndarray : Input data
        - x_real: Optional[np.ndarray] : Real data (optional)
        - y_obs: Optional[np.ndarray] : Observed data (optional)
        - N: int : Number of iterations (default: 1000)
        - burn_in: int : Number of burn-in steps (default: 500)
        - cov_prior: float : Covariance of the prior (default: 0.5)
        - sd_noise: float : Standard deviation of noise (default: 0.1)
        - adapt: bool : Whether to adapt the proposal distribution (default: False)
        - scale: float : Scaling factor for the proposal distribution (default: 0.3)
        - proposal_sd: float : Standard deviation of the proposal distribution (default: 0.3)
        - x_init: Optional[Union[int, float, np.ndarray]] : Initial guess (default: None)
        - diagnostic: bool : Whether to plot diagnostic information (default: True)
        - number_chains: int : Number of MCMC chains (default: 1)
        - algo: str : Algorithm to use ("MH" or "NUTS", default: "MH")
        - transformation: List : List of transformations (default: [])

        Returns:
        - estimates: np.ndarray : Estimated parameters
        - error: float : Error with respect to true parameters
        """

        # Set inputs and transformations
        self.inputs = x_data
        self.transformations = transformation

        # Check if observations are provided
        if y_obs is not None:
            dim = y_obs.shape[1]
        else:
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)
            return

        # Check if the number of steps is greater than burn-in period
        if N <= burn_in:
            warning_message = "Number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning)
            return

        # Initialize x_init if not provided
        if x_init is None:
            x_init = np.random.rand(dim)
        elif isinstance(x_init, (int, float)):
            x_init = x_init * np.ones(dim)
        
        m = x_real.shape[0]
        
        # Select algorithm and initialize CuqiModel and Gaussian objects
        if algo == "NUTS":
            fun = Function(self.wrapper_prediction)
            A = CuqiModel(forward=self.wrapper_prediction, jacobian=fun.compute_jacobian, range_geometry=Continuous1D(dim), domain_geometry=Continuous1D(dim))
            x = Gaussian(mean=mean_prior, cov=cov_prior)
        else:
            # A = CuqiModel(forward=self.wrapper_prediction, range_geometry=Continuous1D(dim), domain_geometry=Continuous1D(m))
            # x = Gaussian(mean=mean_prior, cov=cov_prior)
            A = CuqiModel(forward=self.wrapper_prediction, range_geometry=Discrete(dim), domain_geometry=Discrete(m))
            x = Gaussian(mean=mean_prior, cov=cov_prior)

        y = Gaussian(mean=A(x), cov=proposal_sd)

        # Generate or perturb observations
        if y_obs is None:
            y_obs = y(x=x_real).sample()
        else:
            y_obs = y_obs + np.random.normal(loc=0., scale=sd_noise, size=y_obs.shape)

        # Run MCMC to get estimates
        estimates = MCMC_cuqi(y, x, y_obs, N, burn_in, number_chains, diagnostic=diagnostic, algo=algo, adapt=adapt, scale=scale, parallel=parallel)
        estimates = np.mean(estimates, axis=1)

        # Calculate and print error
        error = np.linalg.norm(estimates - x_real)
        print(f"Error wrt true parameters: {error}")

        # Plot diagnostics if required
        if diagnostic:
            plot_hist(estimates, x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))

        return estimates, error


    @abstractmethod
    def prediction(self) -> None:
        """
        Abstract method to be implemented for making predictions using the network.
        Subclasses must provide the implementation for this method.
        """
        pass

    @abstractmethod
    def performance(self) -> None:
        """
        Abstract method to be implemented for evaluating the network's performance.
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

    @staticmethod
    @abstractmethod
    def save() -> None:
        """
        Abstract static method to be implemented for saving the network's state or model.
        Subclasses must provide the implementation for this method.
        """
        pass


class Neural_Network(INetwork):
    
    def __init__(
        self, 
        name: str, 
        params: Optional[dict] = None, 
        data_train: Optional[np.ndarray] = None, 
        output_train: Optional[np.ndarray] = None, 
        N: int = 1000, 
        n: int = 10, 
        train: bool = True, 
        do_HPO: bool = False, 
        transformations: Optional[list] = None, 
        verbose: bool = False
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
        """
        K.clear_session()
        self.name = name
        self.params = params
        self.N = N
        self.n = n
        self.verbose = verbose
        self.hist = None
        self.data_train = data_train
        self.output_train = output_train
        self.transformations = transformations if transformations is not None else []
        self.inputs = None

        # Set input and output shapes based on training data dimensions
        self.input_shape = self._get_shape(data_train)
        self.output_shape = self._get_shape(output_train)
        # Perform hyperparameter optimization if required or if no parameters are provided
        if do_HPO or params is None:
            if output_train is None or data_train is None:
                warning_message = "Not enough data given!"
                warnings.warn(warning_message, UserWarning)
            self.params = self.HPO(data_train, output_train)

        # Initialize the model
        self.model = getModel(self.params, self.input_shape, self.name, self.output_shape)

        # Train the model if required
        if train:
            self.hist = self.training(data_train, output_train, epoch=self.N, batch=self.n) 
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
        Plots the training loss.
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
    def training(self, x: np.ndarray, y: np.ndarray, epoch: int, batch: int) -> Any:
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
        self.hist = self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=0)
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
            with Suppressor():
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
            params = {
                "nodes": trial.suggest_int("nodes", 4, 64, log=True),
                "l2weight": trial.suggest_float("l2weight", 1e-4, 1e-1, log=True),  # Adjusted to use suggest_float
                "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),  # Adjusted to use suggest_float
                "kernel_init": trial.suggest_categorical("kernel_init", ["uniform", "glorot_uniform"]),
                "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),

            }
            loss = kCrossVal(self.n, self.N, data_train, output_train, params, self.name, self.input_shape, self.output_shape)
            return loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5, n_jobs=-1)  # Parallelize trials
        best_params = study.best_params
        return best_params

    def objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Objective function to minimize during hyperparameter optimization.

        Args:
            params (Dict[str, Any]): Hyperparameters to evaluate.

        Returns:
            Dict[str, Any]: The loss value and parameters.
        """
        K.clear_session()
        loss = kCrossVal(self.n, self.N, self.data_train, self.output_train, params, self.name, self.input_shape, self.output_shape)
        return {"loss": loss, "params": params, "status": STATUS_OK}


    def performance(self, data_test: np.ndarray, output_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the performance of the model on test data.

        Args:
            data_test (np.ndarray): Test data.
            output_test (np.ndarray): Expected output data.

        Returns:
            Tuple[float, float]: Test Mean Squared Error (MSE) and R^2 score.
        """
        pred = self.prediction(data_test)

        if len(output_test.shape) < len(pred.shape):
            output_test = output_test[:, _]
        
        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2 = 1 - np.sum(np.square(output_test - pred)) / np.sum(
            np.square(output_test - np.mean(output_test))
        )
        print(f"R^2: {r2}")
        
        return test_mse, r2   


    def save(self, path: str, identifier: str="_") -> None:
        """
        Save the NN model and the class instance.

        Args:
            path (str): Directory path to save the model and instance.
            identifier (str): Identifier for the saved files.
        """
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)

        # Save the Keras model separately
        model_path = os.path.join(path, f'NN_model_{identifier}.h5')
        self.model.save(model_path)

        # Save the class instance excluding the Keras model
        temp_model = self.model
        self.model = None
        with open(os.path.join(path, f'class_instance_{identifier}.pkl'), 'wb') as f:
            pickle.dump(self, f)

        # Restore the model attribute
        self.model = temp_model

    @classmethod
    def load(cls, path: str, identifier: str) -> ' Neural_Network':
        """
        Load the NN model and the class instance.

        Args:
            path (str): Directory path from which to load the model and instance.
            identifier (str): Identifier for the saved files.

        Returns:
             Neural_Network: The loaded  Neural_Network instance.
        """
        with open(os.path.join(path, f'class_instance_{identifier}.pkl'), 'rb') as f:
            instance = pickle.load(f)

        model_path = os.path.join(path, f'NN_model_{identifier}.h5')
        custom_objects = {
            'mse': MeanSquaredError()  # Add any custom objects required by the model
        }
        instance.model = load_model(model_path, custom_objects=custom_objects)

        return instance

class MultiFidelity(INetwork):
        
    def __init__(self, 
                 names: List[str], 
                 params: Optional[List[dict]] = None, 
                 data_train: Optional[List[np.ndarray]] = None, 
                 output_train: Optional[List[np.ndarray]] = None, 
                 N: Optional[List[int]] = None, 
                 n: Optional[List[int]] = None, 
                 do_HPO: bool = False, 
                 verbose: bool = False):
        """
        Initialize MultiFidelity network.

        Args:
            names (List[str]): Names of the networks.
            params (Optional[List[dict]]): Parameters for each network.
            data_train (Optional[List[np.ndarray]]): Training data for each network.
            output_train (Optional[List[np.ndarray]]): Training outputs for each network.
            N (Optional[List[int]]): Number of epochs for each network.
            n (Optional[List[int]]): Batch sizes for each network.
            do_HPO (bool): Whether to perform hyperparameter optimization.
            verbose (bool): Whether to print verbose output.
        """
        self.names = names
        self.Ns = N if N is not None else [1000] * len(names) 
        self.ns = n if n is not None else [10] * len(names)  
        self.data_train = data_train
        self.output_train=output_train
        self.steps = int((len(names) - 1) / (len(data_train) - 1)) + 1
        self.model_list = []
        self.transformations = []
        self.input_shape = 1
        self.output_shape = 1

        if not data_train or not output_train or len(data_train) != len(output_train):
            raise ValueError('The data are incoherent or insufficient')

        if data_train and len(data_train[0].shape) > 1:
            self.input_shape = data_train[0].shape[1]

        if output_train and len(output_train[0].shape) > 1:
            self.output_shape = output_train[0].shape[1]

        K.clear_session()
        if params is None:
            params = [None] * len(names)
        elif len(params) < len(names):
            params += [None] * (len(names) - len(params))

        count = 1
        for index, name in enumerate(names):
            model = NetworkFactory.build_network(
                name,
                params=params[index],
                data_train=data_train[count - 1],
                output_train=output_train[count - 1],
                N=self.Ns[index],
                n=self.ns[count - 1],
                train=True,
                do_HPO=do_HPO,
                verbose=verbose
            )
            self.model_list.append(model)
            
            if (index + 1) == (self.steps - 1) * (count - 1) + 1:
                count += 1

            for i in range(count - 1, len(data_train)):
                data_train[i] = np.c_[data_train[i], model.prediction(data_train[i])]


    def plot_training_loss(self) -> None:
        """
        Plot training loss for all models in the MultiFidelity network.
        """
        for model in self.model_list:
            model.plot_training_loss()


    
    # FUNZIONE PER TRAINING
    # FUNZIONE PER HPO

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
        if position is None:
            position = len(self.model_list)
        elif not isinstance(position, int) or position > len(self.model_list):
            raise ValueError('The required NN is not existent')

        data = copy.copy(data_test)

        for i in range(position - 1):
            pred=self.model_list[i].prediction(data)
            if len(pred.shape)<=1:
                data = np.c_[data, pred.reshape(-1, 1)]
            else:
                data = np.c_[data, pred]


        pred = self.model_list[position - 1].prediction(data)

        if len(output_test.shape) < len(pred.shape):
            output_test = output_test[:, np.newaxis]

        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2 = 1 - np.sum(np.square(output_test - pred)) / np.sum(np.square(output_test - np.mean(output_test)))
        print(f"R^2: {r2}")

        return test_mse, r2

    def prediction(self, data_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the multi-fidelity model.

        Args:
            data_test (np.ndarray): Test data.

        Returns:
            np.ndarray: Predictions.
        """
        self.outputs = data_test

        if len(self.outputs.shape) == 1:
            self.outputs = self.outputs.reshape(-1, 1)

        for index in range(len(self.names)):
            self.outputs = np.c_[self.outputs, self.model_list[index].prediction(self.outputs)]

        return self.outputs[:, -self.output_shape:]

    def save(self, path: str, identifier: str = "_") -> None:
        """
        Save all models in the model_list to the specified path.

        Args:
            path (str): Directory to save the models.
            identifier (str): Identifier to append to the model names.
        """
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)

        # Save each model in the model_list with a unique identifier
        for index, model in enumerate(self.model_list):
            model_name = identifier + self.names[index]
            model.save(os.path.join(path, f'model_{model_name}.h5'))

    def training(self):
        pass
    
    def HPO(self):
        pass
    
    @classmethod
    def load(cls, path: str, identifier: str) -> 'MultiFidelity':
        """
        Load all models into the model_list from the specified path.

        Args:
            path (str): Directory from which to load the models.
            identifier (str): Identifier used in the model names.

        Returns:
            MultiFidelity: An instance of the MultiFidelity class with loaded models.
        """
        # Create an instance of MultiFidelity without training data
        instance = cls(names=[], params=[], data_train=[], output_train=[], N=[], n=[])

        # Load each model from the specified path
        for name in instance.names:
            model_path = os.path.join(path, f'model_{identifier + name}.h5')
            model = Neural_Network.load(model_path, identifier + name)
            instance.model_list.append(model)

        return instance


    def get_output(self) -> np.ndarray:
        """
        Get the final output from the outputs.

        Returns:
            np.ndarray: The last output.
        """
        return self.model_list[-1].prediction(self.data_train[-1])


class LSTM_network(INetwork):

    def __init__(self, 
                 params: Optional[dict] = None, 
                 data_train: Optional[np.ndarray] = None, 
                 output_train: Optional[np.ndarray] = None, 
                 N: int = 1000, 
                 train: bool = True, 
                 do_HPO: bool = False, 
                 transformations: List[Any] = [], 
                 verbose: bool = False):
        """
        Initialize LSTM_network instance.

        Args:
            params (Optional[dict]): Hyperparameters for the network.
            data_train (Optional[np.ndarray]): Training data.
            output_train (Optional[np.ndarray]): Training outputs.
            N (int): Number of epochs for training.
            train (bool): Flag to indicate if training should be performed.
            do_HPO (bool): Flag to indicate if hyperparameter optimization is to be performed.
            transformations (List[Any]): List of transformations to apply to the data.
            verbose (bool): Flag to indicate verbosity of the output.
        """
        self.params = params
        self.name = "LSTM" 
        self.N = N
        self.verbose = verbose
        self.hist = None
        self.data_train = data_train
        self.output_train = output_train
        self.transformations = transformations

        self.input_shape = 1
        self.output_shape = 1
        self.inputs = None

        # Determine input and output shapes
        if data_train is not None and len(data_train.shape) > 1:
            self.input_shape = data_train.shape[-1]
        if output_train is not None and len(output_train.shape) > 1:
            self.output_shape = output_train.shape[-1]

        # Perform hyperparameter optimization if required
        if do_HPO or params is None:
            if output_train is None or data_train is None:
                warnings.warn("Not enough data given!", UserWarning)
            self.params = self.HPO(data_train, output_train)

        # Initialize the model
        self.model = getModel(self.params,self.input_shape,self.name,self.output_shape)  # dim_input = n_POD + 2, dim_output = n_POD

        # Train the model if required
        if train: 
            self.hist = self.training(int(params['sequence_length']),int(params['sequence_freq']),epoch=self.N) 
            self.plot_training_loss()
        else:
            name = './models/MF_POD_model'
            self.model = tf.keras.models.load_model(name) 


    @compute_time
    def training(self,seq_length,seq_freq,epoch):
        """
        Train the LSTM model.

        Args:
            seq_length (int): Sequence length for training.
            seq_freq (int): Sequence frequency for training.
            epoch (int): Number of epochs for training.

        Returns:
            tf.keras.callbacks.History: Training history.
        """
        self.sequence_length = seq_length
        self.sequence_freq = seq_freq
        self.input_train_seq, self.output_train_seq = self._sliding_windows(self.data_train, self.output_train, self.sequence_length, self.sequence_freq)

        callback = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=self.params['patience'], restore_best_weights=True)
        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism() # for reproducibility

        self.hist = self.model.fit(self.input_train_seq, self.output_train_seq, epochs=epoch, verbose = self.verbose, callbacks=[callback])

        return self.hist

    def plot_training_loss(self) -> None:
        """
        Plots the training loss.
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
        
    def performance(self, data_test: np.ndarray, output_test: np.ndarray, position: Optional[int] = None) -> Tuple[float, float]:
        """
        Evaluate the performance of the LSTM model.

        Args:
            data_test (np.ndarray): Test data.
            output_test (np.ndarray): Test outputs.
            position (Optional[int]): Position of the model in the model list.

        Returns:
            Tuple[float, float]: Mean Squared Error (MSE) and R-squared (R^2) values.
        """
        data = copy.copy(data_test)
        if position is None:
            position = len(self.model_list)
        elif not isinstance(position, int) or position > len(self.model_list):
            raise ValueError('The required NN does not exist.')

        # Iterate through the model list and make predictions
        for i in range(position - 1):
            data = np.concatenate((data, self.model_list[i].wrapper_prediction(data).reshape(-1, 1)), axis=1)

        # Predict using the specified model
        pred = self.model_list[position - 1].prediction(data)

        # Ensure the output shape matches the prediction shape
        if len(output_test.shape) < len(pred.shape):
            output_test = output_test[:, np.newaxis]
            
        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2 = 1 - np.sum(np.square(output_test - pred)) / np.sum(np.square(output_test - np.mean(output_test)))
        print(f"R^2: {r2}")

        return test_mse, r2

    def save(self, path: str, identifier: str) -> None:
        """
        Save the LSTM model and the class instance.

        Args:
            path (str): Directory path to save the model and instance.
            identifier (str): Identifier for the saved files.
        """
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)

        # Save the Keras model separately
        model_path = os.path.join(path, f'lstm_model_{identifier}.h5')
        self.model.save(model_path)

        # Save the class instance excluding the Keras model
        temp_model = self.model
        self.model = None
        with open(os.path.join(path, f'class_instance_{identifier}.pkl'), 'wb') as f:
            pickle.dump(self, f)

        # Restore the model attribute
        self.model = temp_model

    @classmethod
    def load(cls, path: str, identifier: str) -> 'LSTM_network':
        """
        Load the LSTM model and the class instance.

        Args:
            path (str): Directory path from which to load the model and instance.
            identifier (str): Identifier for the saved files.

        Returns:
            LSTM_network: The loaded LSTM_network instance.
        """
        with open(os.path.join(path, f'class_instance_{identifier}.pkl'), 'rb') as f:
            instance = pickle.load(f)

        model_path = os.path.join(path, f'lstm_model_{identifier}.h5')
        custom_objects = {
            'mse': MeanSquaredError()  # Add any custom objects required by the model
        }
        instance.model = load_model(model_path, custom_objects=custom_objects)

        return instance


    def _sliding_windows(self, data_input, data_output, seq_length, freq=1):
        """
        Generates sliding windows for the given data and labels.

        Args:
            data (np.ndarray): Input data.
            labels (np.ndarray): Output labels.
            seq_length (int): Length of each sequence.
            seq_freq (int): Frequency of each sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Input sequences and corresponding output sequences.
        """
        x = []
        y = []

        for i in range(data_input.shape[0]):
            for j in range(0, data_input.shape[1] - seq_length, freq):
                _x = data_input[i, j:(j + seq_length), :]
                _y = data_output[i, j:(j + seq_length), :]
                x.append(_x)
                y.append(_y)

        return np.array(x), np.array(y)

    def param_inverse(self, mean_prior: np.ndarray, x_data: np.ndarray, x_data2: np.ndarray, cov_prior: Optional[np.ndarray] = None, 
                      cov_noise: float = 0.1, cov_likelihood: Optional[np.ndarray] = None, y_obs: Optional[np.ndarray] = None, 
                      x_real: Optional[np.ndarray] = None, number_chains: int = 1, N: int = 1000, burn_in: int = 500, 
                      levels: int = 1, diagnostic: bool = True, rwmh_cov: Optional[np.ndarray] = None, rmwh_scaling: float = 0.1, 
                      rwmh_adaptive: bool = True, algo: str = "MH", transformation: List[Any] = [], _forward_low_fidelity: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform parameter inversion using MCMC sampling, potentially utilizing a low fidelity forward model.

        Parameters:
        - mean_prior: Mean of the prior distribution.
        - x_data: Input data.
        - cov_prior: Covariance of the prior distribution (optional).
        - cov_noise: Noise covariance.
        - cov_likelihood: Covariance of the likelihood (optional).
        - y_obs: Observed data (optional).
        - x_real: Real parameters (optional).
        - number_chains: Number of MCMC chains.
        - N: Number of MCMC iterations.
        - burn_in: Number of burn-in iterations.
        - levels: Number of model levels.
        - diagnostic: Flag to enable diagnostic plots.
        - rwmh_cov: Covariance matrix for RWMH proposal (optional).
        - rmwh_scaling: Scaling factor for RWMH.
        - rwmh_adaptive: Flag for adaptive RWMH.
        - algo: MCMC algorithm to use ("MH", "AM", "CN", "DREAMZ").
        - transformation: List of transformations to apply.
        - _forward_low_fidelity: Low fidelity forward model function (optional).

        Returns:
        - estimates: MCMC estimates of the parameters.
        - error: Relative error of the estimates.
        """

        # Check if _forward_low_fidelity is provided and is a callable function
        if _forward_low_fidelity is None or not callable(_forward_low_fidelity):
            raise KeyError("Provide _forward_low_fidelity, which must be a callable function")
        
        self.input2=x_data2
        # Store the _forward_low_fidelity function for later use
        self._forward_low_fidelity = _forward_low_fidelity
        # Call the parent class's param_inverse method with the provided parameters
        return super().param_inverse(
            mean_prior=mean_prior, 
            x_data=x_data, 
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
            transformation=transformation
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
        prediction_input = self._forward_low_fidelity(x_final)

        return self.prediction(prediction_input).flatten()
    
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
            params = {
                "nodes": trial.suggest_int("nodes", 4, 64, log=True),
                "l2weight": trial.suggest_float("l2weight", 1e-4, 1e-1, log=True),
                "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                "kernel_init": trial.suggest_categorical("kernel_init", ["uniform", "glorot_uniform"]),
                "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),
                "sequence_length": trial.suggest_int("sequence_length", 10, 100),
                "sequence_freq": trial.suggest_int("sequence_freq", 1, 10),
                "patience": trial.suggest_int("patience", 3, 10),
            }
            loss = kCrossVal_parallel(N=self.n, Nepo=self.N, x=data_train, y=output_train, 
                                      params=params, name=self.name, input_shape=self.input_shape, 
                                      output_shape=self.output_shape, p=5, n_jobs=-1)
            return loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20, n_jobs=-1)
        best_params = study.best_params
        return best_params

class Intermediate(INetwork):
    def __init__(self, 
                 params: Optional[dict] = None, 
                 data_train: Optional[List[np.ndarray]] = None, 
                 output_train: Optional[List[np.ndarray]] = None, 
                 N: int = 1000, 
                 n: int = 10, 
                 train: bool = True, 
                 do_HPO: bool = False, 
                 verbose: bool = False):
        """
        Initialize Intermediate network.

        Args:
            params (Optional[dict]): Parameters for the network.
            data_train (Optional[List[np.ndarray]]): Training data for the network.
            output_train (Optional[List[np.ndarray]]): Training outputs for the network.
            N (int): Number of epochs for training.
            n (int): Batch size for training.
            train (bool): Whether to train the model.
            do_HPO (bool): Whether to perform hyperparameter optimization.
            verbose (bool): Whether to print verbose output.
        """
        self.params = params
        self.name = "Inter" 
        self.N = N
        self.n = n
        self.verbose = verbose
        self.hist = None
        self.data_train = data_train
        self.output_train = output_train
        self.transformations = []
        
        self.input_shape = 1
        self.output_shape = 1

        if len(data_train) != 2 or len(output_train) != 2:
            raise ValueError('The data are incoherent or insufficient')
                
        # Concatenate data for training
        data_train = np.concatenate((data_train[1], data_train[0]), axis=0)
        output_train = np.concatenate((output_train[1], output_train[0]), axis=0)
        
        # Determine input and output shapes
        if data_train is not None and len(data_train.shape) > 1:
            self.input_shape = data_train.shape[1]

        if output_train is not None and len(output_train.shape) > 1:
            self.output_shape = output_train.shape[1]

        if do_HPO or params is None:
            if output_train is None or data_train is None:
                warnings.warn("Not enough data given!", UserWarning)
            self.params = self.HPO(data_train, output_train)

        # Create the model
        self.model = getModel(self.params, self.input_shape, self.name, self.output_shape)

        if train:
            self.hist = self.model.fit(data_train, output_train, epochs=self.N, batch_size=self.n, verbose=self.verbose) 
            self.plot_training_loss()

    @compute_time
    def training(self, x: np.ndarray, y: np.ndarray, epoch: int, batch: int) -> Any:
        """
        Train the model on the given data.

        Args:
            x (np.ndarray): Training data.
            y (np.ndarray): Training labels.
            epoch (int): Number of epochs.
            batch (int): Batch size.

        Returns:
            Any: Training history.
        """
        self.hist = self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=self.verbose) 
        return self.hist
    
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

    def prediction(self, x_test: List[np.ndarray]) -> np.ndarray:
        """
        Make predictions using the model.

        Args:
            x_test (List[np.ndarray]): Test data.

        Returns:
            np.ndarray: Predictions.
        """
        if len(x_test) != 2:
            raise ValueError("Not enough data given")
        
        x_test = np.concatenate((x_test[1], x_test[0]), axis=0)

        if self.verbose:
            return self.model.predict(x_test)
        else:
            with Suppressor():
                return self.model.predict(x_test)

    def performance(self, data_test: List[np.ndarray], output_test: List[np.ndarray]) -> Tuple[float, float]:
        """
        Evaluate the performance of the model.

        Args:
            data_test (List[np.ndarray]): Test data.
            output_test (List[np.ndarray]): Test labels.

        Returns:
            Tuple[float, float]: Test MSE and R^2 score.
        """
        data_test = np.concatenate((data_test[1], data_test[0]), axis=0)
        output_test = np.concatenate((output_test[1], output_test[0]), axis=0)
        
        pred = self.prediction(data_test)

        if len(output_test.shape) < len(pred.shape):
            output_test = output_test[:, np.newaxis]

        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2 = 1 - np.sum(np.square(output_test - pred)) / np.sum(np.square(output_test - np.mean(output_test)))
        print(f"R^2: {r2}")
        
        return test_mse, r2

    def objective(self, par: dict) -> Dict[str, Any]:
        """
        Objective function for hyperparameter optimization.

        Args:
            par (dict): Hyperparameters.

        Returns:
            Dict[str, Any]: Result of cross-validation.
        """
        K.clear_session()
        CVres = kCrossVal(self.n, self.N, self.data_train, self.output_train, par, self.name, self.input_shape)
        return {"loss": CVres, "params": par, "status": STATUS_OK} 

    def HPO(self, data_train: np.ndarray, output_train: np.ndarray) -> dict:
        """
        Hyperparameter Optimization (to be implemented).

        Args:
            data_train (np.ndarray): Training data.
            output_train (np.ndarray): Training labels.

        Returns:
            dict: Best hyperparameters.
        """
        pass  # Implement hyperparameter optimization logic here










def transfBestparam(best_params: Dict[str, Any], dic: Dict[str, Any]) -> None:
    """
    Transforms kernel and optimizer parameters from numeric indicators to string values.
    
    Args:
        best_params (Dict[str, Any]): Best parameters obtained from hyperparameter optimization.
        dic (Dict[str, Any]): Dictionary mapping numeric indicators to string values.
    """
    for key in best_params:
        if key in ["kernel_init", "opt"]:
            best_params[key] = dic[key][best_params[key]]

def getOpti(name: str, lr: float) -> Union[str, Adam, Nadam, Adamax, RMSprop]:
    """
    Retrieve the optimizer based on the provided name and learning rate.

    Args:
        name (str): Name of the optimizer.
        lr (float): Learning rate for the optimizer.

    Returns:
        Union[str, Adam, Nadam, Adamax, RMSprop]: The optimizer object or 'adam' string for standard Adam.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    optimizers = {
        'Adam': Adam(learning_rate=lr, amsgrad=True),
        'Nadam': Nadam(learning_rate=lr),
        'Adamax': Adamax(learning_rate=lr),
        'RMSprop': RMSprop(learning_rate=lr),
        'standardadam': 'adam'
    }
    
    if name not in optimizers:
        raise ValueError(f"Optimizer name '{name}' is not recognized. Valid options are: {list(optimizers.keys())}")
    
    return optimizers[name]
  



def kCrossVal(N: int, Nepo: int, x: np.ndarray, y: np.ndarray, params: Dict[str, Any], 
              name: str, input_shape: int, output_shape: int, p: int = 1) -> float:
    """
    Perform k-fold cross-validation on the model.

    Args:
        N (int): Total number of samples.
        Nepo (int): Number of epochs for training.
        x (np.ndarray): Training data.
        y (np.ndarray): Training outputs.
        params (Dict[str, Any]): Hyperparameters for the model.
        name (str): Name of the model.
        input_shape (int): Shape of the input data.
        output_shape (int): Shape of the output data.
        p (int): Number of folds.

    Returns:
        float: Average cross-validation loss.
    """
    model = getModel(params, input_shape, name, output_shape)
    kf = KFold(n_splits=int(N/p), shuffle=True)
    scores = []

    for train_index, test_index in kf.split(x):
        x_train, x_val = x[train_index], x[test_index]
        y_train, y_val = y[train_index], y[test_index]

        model.fit(x_train, y_train, epochs=Nepo, batch_size=len(train_index), verbose=0)  # Optimization: Efficient model training
        predictions = model.predict(x_val)
        score = np.mean(np.square(y_val - predictions[:, 0]))  # Optimization: Efficient calculation of the score
        scores.append(score)

    return np.mean(scores)

def kCrossValSingle(N: int, Nepo: int, x: np.ndarray, y: np.ndarray, params: Dict[str, Any], 
                    name: str, input_shape: int) -> float:
    """
    Perform k-fold cross-validation on the model with a single fold.

    Args:
        N (int): Total number of samples.
        Nepo (int): Number of epochs for training.
        x (np.ndarray): Training data.
        y (np.ndarray): Training outputs.
        params (Dict[str, Any]): Hyperparameters for the model.
        name (str): Name of the model.
        input_shape (int): Shape of the input data.

    Returns:
        float: Average cross-validation loss.
    """
    kf = KFold(n_splits=N, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(x):
        x_train, x_val = x[train_index], x[test_index]
        y_train, y_val = y[train_index], y[test_index]

        model = getModel(params, input_shape, name, y.shape[1])
        model.fit(x_train, y_train, epochs=Nepo, batch_size=len(train_index), verbose=0)  # Optimization: Efficient model training
        predictions = model.predict(x_val)
        score = np.mean(np.square(y_val - predictions))  # Optimization: Efficient calculation of the score
        scores.append(score)

    return np.mean(scores)

def kCrossValGP(Nhf: int, Nlf: int, Nepo: int, xhf: np.ndarray, yhf: np.ndarray, xlf: np.ndarray, 
                ylf: np.ndarray, params: Dict[str, Any], name: str, input_shape: int, p: int = 1) -> float:
    """
    Perform k-fold cross-validation for Gaussian Process models.

    Args:
        Nhf (int): Number of high-fidelity samples.
        Nlf (int): Number of low-fidelity samples.
        Nepo (int): Number of epochs for training.
        xhf (np.ndarray): High-fidelity training data.
        yhf (np.ndarray): High-fidelity training outputs.
        xlf (np.ndarray): Low-fidelity training data.
        ylf (np.ndarray): Low-fidelity training outputs.
        params (Dict[str, Any]): Hyperparameters for the model.
        name (str): Name of the model.
        input_shape (int): Shape of the input data.
        p (int): Number of folds.

    Returns:
        float: Average cross-validation loss.
    """
    Nfolds = int(Nhf / p)
    kf = KFold(n_splits=Nfolds, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(xhf):
        xhf_train, xhf_val = xhf[train_index], xhf[test_index]
        yhf_train, yhf_val = yhf[train_index], yhf[test_index]
        x_train = np.concatenate((xhf_train, xlf))
        yhf_train_aug = np.concatenate((yhf_train, np.full(Nlf, -10)))
        ylf_train_aug = np.concatenate((np.full(len(xhf_train), -10), ylf))

        model = getModel(params, input_shape, name, yhf.shape[1])
        model.fit(x_train, [yhf_train_aug, ylf_train_aug], epochs=params['epochs'] * Nepo, batch_size=len(train_index), verbose=0)  # Optimization: Efficient model training
        predictions = model.predict(xhf_val)
        score = np.mean(np.square(yhf_val - predictions[0][:, 0]))  # Optimization: Efficient calculation of the score
        scores.append(score)

    return np.mean(scores)


def kCrossVal_parallel(N: int, Nepo: int, x: np.ndarray, y: np.ndarray, params: Dict[str, Any], 
                       name: str, input_shape: int, output_shape: int, p: int = 1, n_jobs: int = -1) -> float:
    """
    Perform k-fold cross-validation on the model using parallel processing.

    Args:
        N (int): Total number of samples.
        Nepo (int): Number of epochs for training.
        x (np.ndarray): Training data.
        y (np.ndarray): Training outputs.
        params (Dict[str, Any]): Hyperparameters for the model.
        name (str): Name of the model.
        input_shape (int): Shape of the input data.
        output_shape (int): Shape of the output data.
        p (int): Number of folds.
        n_jobs (int): Number of parallel jobs.

    Returns:
        float: Average cross-validation loss.
    """
    kf = KFold(n_splits=p, shuffle=True)

    def fit_and_score(train_index, test_index):
        model = getModel(params, input_shape, name, output_shape)
        x_train, x_val = x[train_index], x[test_index]
        y_train, y_val = y[train_index], y[test_index]
        model.fit(x_train, y_train, epochs=Nepo, batch_size=len(train_index), verbose=0)  # Optimization: Efficient model training
        predictions = model.predict(x_val)
        return np.mean(np.square(y_val - predictions[:, 0]))  # Optimization: Efficient calculation of the score

    scores = Parallel(n_jobs=n_jobs)(delayed(fit_and_score)(train_index, test_index) for train_index, test_index in kf.split(x))  # Optimization: Parallel processing
    return np.mean(scores)



