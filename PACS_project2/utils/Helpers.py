import contextlib
import numpy as np
import os
import sys
import time
import tinyDA as tda

from cuqi.geometry import Continuous1D, Continuous2D, Discrete
from cuqi.model import Model as CuqiModel
from enum import Enum
from functools import wraps
from joblib import Parallel, delayed
from numba import jit, njit
from numpy import newaxis as _   # easier reading
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam, Adamax, Nadam, RMSprop
from typing import Any, Callable, Dict, Optional, Tuple, Union

from BIP_functions import *
from Helpers import *
from Structure import *
from module_utils import *



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

@contextlib.contextmanager
def Suppressor():
    try:        
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            with contextlib.redirect_stdout(devnull):                
                yield
    finally:        
        pass


# Helpers functions for Structure..py, INetwork class 

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


# Class to handle a function and compute its Jacobian matrix (used for NUTS, cuqipy library)
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

  
# Different types of Cross Validations
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

        # Reshape y_val if necessary and calculate mean squared error
        y_val = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val

        model.fit(x_train, y_train, epochs=Nepo, batch_size=len(train_index), verbose=0)  # Optimization: Efficient model training
        predictions = model.predict(x_val)
        score = np.mean(np.square(y_val - predictions[:, 0]))  # Optimization: Efficient calculation of the score
        scores.append(score)

    return np.mean(scores)


def kCrossValSingle(N: int, Nepo: int, x: np.ndarray, y: np.ndarray, params: Dict[str, Any], 
                    name: str, input_shape: int, output_shape:int) -> float:
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

        # Reshape y_val if necessary and calculate mean squared error
        y_val = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val

        model = getModel(params, input_shape, name, output_shape)
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

        # Reshape y_val if necessary and calculate mean squared error
        yhf_val = yhf_val.reshape(-1, 1) if len(yhf_val.shape) == 1 else yhf_val

        model = getModel(params, input_shape, name, yhf.shape[1])
        model.fit(x_train, [yhf_train_aug, ylf_train_aug], epochs=params['epochs'] * Nepo, batch_size=len(train_index), verbose=0)  # Optimization: Efficient model training
        predictions = model.predict(xhf_val)
        score = np.mean(np.square(yhf_val - predictions[0][:, 0]))  # Optimization: Efficient calculation of the score
        scores.append(score)

    return np.mean(scores)




def kCrossVal_parallel( Nepo: int, x: np.ndarray, y: np.ndarray, params: Dict[str, Any], 
                       name: str, input_shape: int, output_shape: int, p: int = 2, n_jobs: int = -1) -> float:
    """
    Perform k-fold cross-validation on the model using parallel processing.

    Args:
        Nepo (int): Number of epochs for training.
        x (np.ndarray): Training data.
        y (np.ndarray): Training outputs (expected to have 2 columns).
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
        
        # Train the model
        model.fit(x_train, y_train, epochs=Nepo, batch_size=len(train_index), verbose=0)
        
        # Get predictions
        predictions = model.predict(x_val)
        
        # Reshape y_val if necessary and calculate mean squared error
        y_val = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val
        
        # Ensure that predictions and y_val have the same shape
        assert predictions.shape == y_val.shape, f"Mismatch in shapes: predictions {predictions.shape}, y_val {y_val.shape}"
        
        # Calculate mean squared error over all output columns
        return np.mean(np.square(y_val - predictions))

    # Perform k-fold cross-validation in parallel
    scores = Parallel(n_jobs=n_jobs)(delayed(fit_and_score)(train_index, test_index) for train_index, test_index in kf.split(x))
    
    return np.mean(scores)