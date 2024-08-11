import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, LSTM, Dropout
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
import tensorflow as tf
from typing import Callable, Tuple, Any, List, Optional, Union, Dict

from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter

from itertools import product
#from cuqi.model import Model as CuqiModel


# Define custom types for better readability
ParamsType = Dict[str, Union[int, float, str, Any]]
OutputType = Union[Model, List[Model]]


def custom_activation(x):
    """
    Custom activation function that modifies the input tensor `x`.

    This activation function adds the square of the sine of `x` to the original input `x`.
    The function can introduce non-linearity in the model in a way that might help capture
    more complex patterns in the data.

    Args:
        x: A tensor or variable representing the input to the activation function.

    Returns:
        A tensor of the same shape as `x`, with the activation applied.
    """
    return x + K.square(K.sin(x))

def normalization(x: float, xmax: float, xmin: float) -> float:
    """
    Normalize a value x to the range [0, 1] based on the given minimum and maximum values.

    Parameters:
    - x (float): The value to be normalized.
    - xmax (float): The maximum value of the original range.
    - xmin (float): The minimum value of the original range.

    Returns:
    - float: The normalized value in the range [0, 1].
    """
    return (x - xmin) / (xmax - xmin)

def denormalization(x: float, xmax: float, xmin: float) -> float:
    """
    Denormalize a value x from the range [0, 1] back to the original range [xmin, xmax].

    Parameters:
    - x (float): The normalized value in the range [0, 1].
    - xmax (float): The maximum value of the original range.
    - xmin (float): The minimum value of the original range.

    Returns:
    - float: The denormalized value in the original range [xmin, xmax].
    """
    return x * (xmax - xmin) + xmin

    
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


def getModel(params: ParamsType, num_inputs: int, name: str, num_outputs: int) -> OutputType:
    """
    Create and return a compiled Keras model based on the given architecture name and parameters.

    Args:
        params (Dict): Dictionary containing model parameters like 'nodes', 'dropout', 'l2weight', etc.
        num_inputs (int): Number of input features.
        name (str): Name of the model architecture ('LSTM', 'HF', 'LF', 'Single', 'Hflin', 'Hfper', 'GP', 'Inter').
        num_outputs (int): Number of output neurons.

    Returns:
        model (Model or List[Model]): Compiled Keras model, or list of models for some architectures.
    """

    inputs = None
    output = None

    if name == "LSTM":
        # Input layer for LSTM model
        inputs = Input(shape=(None, num_inputs))
        a = inputs

        # LSTM layers with Dropout
        for i in range(params['lay']):
            a = Dropout(params['dropout'])(a)
            a = LSTM(params['nodes'], return_sequences=True)(a)

        # Dense output layer
        output = Dense(num_outputs, activation='linear')(a)

    elif name == 'HF':
        # High-frequency model with regularization
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs, activation='linear', name='HF')(hidden1)

    elif name == 'LF':
        # Low-frequency model with 4 hidden layers
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden3)
        output = Dense(num_outputs, activation='linear', name='LF')(hidden4)

    elif name == 'Single':
        # Single model architecture with L2 regularization
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']))(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']))(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']))(hidden2)
        output = Dense(num_outputs, activation='linear', name='Single')(hidden3)

    elif name == 'Hflin':
        # High-frequency linear model
        inputs = Input(shape=(num_inputs,))
        hiddenlin = Dense(64, activation='linear', kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs, activation='linear', name='HFlin')(hiddenlin)

    elif name == 'Hfper':
        # High-frequency model with custom activation
        inputs = Input(shape=(num_inputs,))
        hiddenlin = Dense(64, activation=custom_activation, kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs, activation='linear', name='HFper')(hiddenlin)

    elif name == 'GP':
        # Gaussian Process inspired model with dual outputs
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'])(hidden3)
        GPlayer = Dense(2, activation='linear', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'])(hidden4)
        outputLF = Dense(1, activation='linear', name='LF')(GPlayer)
        outputHF = Dense(1, activation='linear', name='HF')(GPlayer)
        output = [outputHF, outputLF]

        # Compile model with custom loss and optimizer
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1-params['alpha']], optimizer=opti)
        return model

    elif name == 'Inter':
        # Intermediate frequency model with dual outputs
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden1)
        outputLF = Dense(1, activation='linear', name='LF')(hidden2)
        outputadd = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'])(hidden2)
        merge = concatenate([outputLF, outputadd])
        hidden3 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'])(merge)
        hidden4 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'])(hidden3)
        outputHF = Dense(1, activation='linear', name='HF')(hidden4)
        output = [outputHF, outputLF]

        # Compile model with custom loss and optimizer
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1-params['alpha']], optimizer=opti)
        return model

    # General model compilation for other architectures
    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params['opt'], params['lr'])
    model.compile(loss='mse', optimizer=opti, metrics=['mse'])

    return model



def process_data(datahf: np.ndarray, 
                 parameters: np.ndarray, 
                 t_eval: np.ndarray, 
                 Yhf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the elements of a dataset nearest to the given parameters.
    
    Parameters:
    - datahf (np.ndarray): 2D array where datahf[:, 1] contains parameter values.
    - parameters (np.ndarray): 1D array of parameter values to find in datahf.
    - t_eval (np.ndarray): 2D array of evaluation times.
    - Yhf (np.ndarray): 1D array of corresponding y values.
    
    Returns:
    - nearest_x (np.ndarray): 1D array of x values closest to each t_eval.
    - y_obs (np.ndarray): 1D array of corresponding y values from Yhf.
    """
    indices = np.where(datahf[:, 1] == parameters[0])[0]
    if len(indices) == 0:
        raise ValueError(f"No observations related to parameter: {parameters[0]}")
    
    datahf_values = datahf[indices, 0].reshape(-1, 1)
    t_eval_values = t_eval.reshape(1, -1)
    differences = np.abs(datahf_values - t_eval_values)
    closest_indices = np.argmin(differences, axis=0)
    nearest_x = datahf[indices[closest_indices], 0]
    y_obs = Yhf[indices[closest_indices]%indices.shape[0]]
    
    return nearest_x, y_obs

def calculate_cov_likelihood(sigma: float, t_eval: np.ndarray) -> np.ndarray:
    """
    Calculate the covariance matrix for the likelihood.
    
    Parameters:
    - sigma (float): Standard deviation for the likelihood.
    - t_eval (np.ndarray): 2D array of evaluation times.
    
    Returns:
    - cov_likelihood (np.ndarray): 2D array representing the covariance matrix.
    """
    return sigma ** 2 * np.eye(t_eval.shape[0])

def run_simulation(datahf_x: np.ndarray, 
                   datahf: np.ndarray, 
                   mean_prior: np.ndarray, 
                   cov_prior: np.ndarray, 
                   Yhf: np.ndarray, 
                   sigma_noise: List[float], 
                   n_data: List[int], 
                   n_data_x:List[int], 
                   parameters: np.ndarray, 
                   sigma: np.ndarray, 
                   rwmh_scaling: np.ndarray, 
                   rwmh_cov: np.ndarray, 
                   rwmh_adaptive: bool, 
                   iterations: int,
                   burnin: int, 
                   n_chains: int, 
                   final_model: Any, 
                   algo: str, 
                   force_sequential:bool=False,
                   forward_low_fidelity: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Run a simulation to estimate parameters and calculate errors.
    
    Parameters:
    - datahf_x (np.ndarray): 1D array containing data.
    - datahf (np.ndarray): 2D array containing data  (t and parameter).
    - mean_prior (np.ndarray): 1D array for the mean of the prior.
    - cov_prior (np.ndarray): 2D array for the covariance of the prior.
    - Yhf (np.ndarray): 1D array of observed values.
    - sigma_noise (List[float]): List of noise levels.
    - n_data (List[int]): List of number of data points along t.
    - n_data_x (List[int]): List of number of data points along x.
    - parameters (np.ndarray): 1D array of parameters.
    - sigma (np.ndarray): 1D array of standard deviations for the likelihood.
    - rwmh_scaling (np.ndarray): 1D array of scaling factors for the RWMH algorithm.
    - rwmh_cov (np.ndarray): 2D array for the RWMH covariance.
    - rwmh_adaptive (bool): Boolean indicating if RWMH is adaptive.
    - iterations (int): Integer for the number of iterations.
    - burnin (int): Integer for the burn-in period.
    - n_chains (int): Integer for the number of chains.
    - final_model (Any): The model object with the param_inverse method.
    - algo (str): String indicating the algorithm to use.
    - force_sequenntial (bool): True to avoid parallelization
    - forward_low_fidelity (Optional[Callable]): Low fidelity forward model function (optional).
    
    Returns:
    - best_estimate (np.ndarray): The best parameter estimate.
    - best_error (np.ndarray): The error corresponding to the best estimate.
    - param_final (List[dict]): list of parameter of MCMC algorithm  
    """
    
    # Initialize error and estimate arrays
    error_shape = (len(sigma_noise), len(n_data), len(n_data_x),len(sigma), len(rwmh_scaling))
    error = np.zeros(error_shape)
    estimates = np.zeros(error_shape)
    param_final=[]    
    # Iterate over all combinations of parameters using itertools.product
    for (i, noise), (k, n),(w,n_x), (t, s), (j, r) in product(enumerate(sigma_noise), enumerate(n_data), enumerate(n_data_x), enumerate(sigma), enumerate(rwmh_scaling)):
        t_eval = np.linspace(np.min(datahf[:,0]), np.max(datahf[:,0]), n).reshape(-1, 1)  # Generate evaluation times
        x_eval=np.linspace(np.min(datahf_x),np.max(datahf_x),n_x).reshape(-1, 1)
        _, y_obs = process_data(datahf, parameters, t_eval, Yhf)  # Get nearest t values and observations
        cov_likelihood = calculate_cov_likelihood(s, t_eval)  # Compute the covariance for the likelihood
        
        # Perform parameter estimation and calculate error 
        
        estimates[i, k, w,t, j], error[i, k,w, t, j], par = final_model.param_inverse(
            mean_prior=mean_prior, 
            x_data=t_eval, 
            cov_prior=cov_prior, 
            rmwh_scaling=r, 
            cov_noise=noise, 
            cov_likelihood=cov_likelihood, 
            y_obs=y_obs, 
            x_real=parameters, 
            number_chains=n_chains, 
            N=iterations, 
            burn_in=burnin, 
            diagnostic=True, 
            rwmh_cov=rwmh_cov, 
            rwmh_adaptive=rwmh_adaptive, 
            algo=algo, 
            force_sequential=force_sequential,
            forward_low_fidelity=forward_low_fidelity, 
            x_eval
        )
        param_final.append(par)
    # Identify the index of the minimum error
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, "
          f"rwmh_scaling={rwmh_scaling[smallest_index[3]]}")

    return best_estimate, best_error, param_final