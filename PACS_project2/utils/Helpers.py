import os
import sys
import time
from functools import wraps
import contextlib 
from typing import Callable, Tuple, Any, Dict, Union
import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tensorflow.keras.optimizers import Adam, Nadam, Adamax, RMSprop
from tensorflow.keras.models import Model
from module_utils import *
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

@contextlib.contextmanager
def Suppressor():
    try:        
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            with contextlib.redirect_stdout(devnull):                
                yield
    finally:        
        pass


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



