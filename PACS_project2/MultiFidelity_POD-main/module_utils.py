from bayes_opt import BayesianOptimization
from typing import Callable, Tuple, Any,  List, Optional, Dict, Union
from itertools import product
from numba import njit
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, LSTM, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Nadam, Adamax
from keras.layers import Layer



# Custom activation function that modifies the input with a squared sinusoidal term.
def custom_activation(x: tf.Tensor) -> tf.Tensor:
    """
    Custom activation function that applies a non-linear transformation
    using a squared sine function added to the input.

    Parameters:
    - x (tf.Tensor): Input tensor.

    Returns:
    - tf.Tensor: Transformed tensor.
    """
    return x + K.square(K.sin(x))

# Function to normalize input data between 0 and 1 based on min and max values.
def normalization(x: tf.Tensor, xmax: tf.Tensor, xmin: tf.Tensor) -> tf.Tensor:
    """
    Normalizes input tensor `x` to a range between 0 and 1 using
    the provided minimum and maximum values.

    Parameters:
    - x (tf.Tensor): Input tensor to normalize.
    - xmax (tf.Tensor): Tensor containing the maximum values for normalization.
    - xmin (tf.Tensor): Tensor containing the minimum values for normalization.

    Returns:
    - tf.Tensor: Normalized tensor.
    """
    return (x - xmin) / (xmax - xmin)

# Function to reverse normalization, scaling data back to its original range.
def denormalization(x: tf.Tensor, xmax: tf.Tensor, xmin: tf.Tensor) -> tf.Tensor:
    """
    Reverts the normalization process, scaling the tensor `x` back to its original range
    defined by the provided minimum and maximum values.

    Parameters:
    - x (tf.Tensor): Normalized tensor to be denormalized.
    - xmax (tf.Tensor): Tensor containing the maximum values for denormalization.
    - xmin (tf.Tensor): Tensor containing the minimum values for denormalization.

    Returns:
    - tf.Tensor: Denormalized tensor.
    """
    return x * (xmax - xmin) + xmin

# Custom loss function that computes the mean squared error, ignoring specific predictions.
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

# Sinusoidal activation function that applies a square sine to the input.
def sinusoidal_activation(x: tf.Tensor) -> tf.Tensor:
    """
    Sinusoidal activation function that applies a square sine transformation to the input.

    Parameters:
    - x (tf.Tensor): Input tensor.

    Returns:
    - tf.Tensor: Transformed tensor.
    """
    return K.square(K.sin(x))

# Custom Keras layer that applies a Fourier transformation using learned sine and cosine kernels.
class FourierLayer(Layer):
    """
    Custom Keras layer that applies a Fourier transformation to the input tensor
    using learned sine and cosine kernels.

    Attributes:
    - output_dim (int): Number of output dimensions for the Fourier transformation.
    """

    def __init__(self, output_dim: int, **kwargs):
        """
        Initializes the FourierLayer with the specified output dimensions.

        Parameters:
        - output_dim (int): Number of output dimensions for the Fourier transformation.
        - kwargs: Additional keyword arguments for the Layer class.
        """
        self.output_dim = output_dim
        super(FourierLayer, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        """
        Builds the layer by initializing the sine and cosine kernels used
        for the Fourier transformation.

        Parameters:
        - input_shape (tf.TensorShape): Shape of the input tensor.
        """
        self.kernel_sin = self.add_weight(name='kernel_sin',
                                          shape=(self.output_dim,),
                                          initializer='glorot_uniform',
                                          trainable=True)
        self.kernel_cos = self.add_weight(name='kernel_cos',
                                          shape=(self.output_dim,),
                                          initializer='glorot_uniform',
                                          trainable=True)
        super(FourierLayer, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies the Fourier transformation to the input tensor using
        the learned sine and cosine kernels.

        Parameters:
        - x (tf.Tensor): Input tensor.

        Returns:
        - tf.Tensor: Transformed tensor.
        """
        result = tf.sin(tf.multiply(x, self.kernel_sin)) + tf.cos(tf.multiply(x, self.kernel_cos))
        return result

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """
        Computes the output shape of the layer, which matches the input shape.

        Parameters:
        - input_shape (tf.TensorShape): Shape of the input tensor.

        Returns:
        - tf.TensorShape: Shape of the output tensor.
        """
        return input_shape
    

def custom_activation(x: tf.Tensor) -> tf.Tensor:
    """
    Custom activation function combining linear and non-linear transformations.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Transformed tensor.
    """
    return x + K.square(K.sin(x))

# class Attention(Layer):
#     def __init__(self):
#         super(Attention, self).__init__()

#     def build(self, input_shape):
#         self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
#         self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
#         self.u = self.add_weight(name='context_vector', shape=(input_shape[-1],), initializer='random_normal', trainable=True)
#         super(Attention, self).build(input_shape)

#     def call(self, inputs):
#         score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
#         attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
#         context_vector = attention_weights * inputs
#         context_vector = tf.reduce_sum(context_vector, axis=1)
#         return context_vector

def getModel(
    params: Dict[str, Union[int, float, str, bool]], 
    num_inputs: int, 
    name: str, 
    num_outputs: int
) -> Model:
    """
    Builds and compiles a Keras model based on the specified architecture.

    Parameters:
    - params (Dict): A dictionary containing various parameters for the model.
    - num_inputs (int): Number of input features.
    - name (str): Name of the model architecture to build.
    - num_outputs (int): Number of output features.

    Returns:
    - model (Model): A compiled Keras model.
    """

    inputs = Input(shape=(None, num_inputs) if 'LSTM' in name else (num_inputs,))

    if name == "LSTM":
        a = LSTM(params['nodes'], return_sequences=True)(inputs)
        for _ in range(params['lay'] - 1):
            a = Dropout(params['dropout'])(a)
            a = LSTM(params['nodes'], return_sequences=True)(a)
        for _ in range(params['lay_dense']):
            a = Dense(params['nodes_dense'])(a)
        output = Dense(num_outputs, activation='linear')(a)

    elif name == "LSTM_support":
        a = LSTM(params['nodes'], return_sequences=True)(inputs)
        for _ in range(params['lay'] - 1):
            a = Dropout(params['dropout'])(a)
            a = LSTM(params['nodes'], return_sequences=True)(a)
        a = Dense(64, activation=sinusoidal_activation, kernel_initializer='uniform')(a)
        for _ in range(params['lay_dense']):
            a = Dense(params['nodes_dense'], activation=custom_activation)(a)
        output = Dense(num_outputs, activation='linear')(a)

    elif name == 'HF':
        hidden1 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2(params['l2weight']),
                        kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs, activation='linear', name='HF')(hidden1)

    elif name == 'LF':
        hidden1 = Dense(64, activation=custom_activation, kernel_initializer=params['kernel_init'],
                        kernel_regularizer=l2(0.001))(inputs)
        hidden2 = Dense(64, activation=custom_activation, kernel_initializer=params['kernel_init'],
                        kernel_regularizer=l2(0.001))(hidden1)
        fourier_layer1 = FourierLayer(output_dim=64)(hidden2)
        hidden3 = Dense(64, activation=custom_activation, kernel_initializer=params['kernel_init'],
                        kernel_regularizer=l2(0.001))(fourier_layer1)
        hidden3 = Dropout(0.05)(hidden3)
        hidden4 = Dense(64, activation=custom_activation, kernel_initializer=params['kernel_init'],
                        kernel_regularizer=l2(0.001))(hidden3)
        output = Dense(num_outputs, activation="linear", name="LF")(hidden4)

    elif name == 'Single':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'],
                        kernel_regularizer=l2(params['l2weight']))(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'],
                        kernel_regularizer=l2(params['l2weight']))(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'],
                        kernel_regularizer=l2(params['l2weight']))(hidden2)
        hidden4 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'],
                        kernel_regularizer=l2(params['l2weight']))(hidden3)
        output = Dense(num_outputs, activation='linear', name='Single')(hidden4)

    elif name == 'Hflin':
        hiddenlin = Dense(64, activation='linear', kernel_regularizer=l2(params['l2weight']),
                          kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs, activation='linear', name='HFlin')(hiddenlin)

    elif name == 'Hfper':
        hiddenlin = Dense(64, activation=custom_activation, kernel_regularizer=l2(params['l2weight']),
                          kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs, activation='linear', name='HFper')(hiddenlin)

    elif name == 'Inter':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden1)
        outputLF = Dense(1, activation='linear', name='LF')(hidden2)
        outputadd = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                          kernel_initializer=params['kernel_init'])(hidden2)
        merge = concatenate([outputLF, outputadd])
        hidden3 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                        kernel_initializer=params['kernel_init'])(merge)
        hidden4 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                        kernel_initializer=params['kernel_init'])(hidden3)
        outputHF = Dense(1, activation='linear', name='HF')(hidden4)
        model = Model(inputs=inputs, outputs=[outputHF, outputLF])
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1 - params['alpha']], optimizer=opti)
        return model

    # Compile and return the model for non-GP and non-Inter models
    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params['opt'], params['lr'])
    model.compile(loss='mse', optimizer=opti, metrics=['mse'])
    return model

# ELIMINATE?
def load_reaction_diffusion(  
    params: List[float], 
    fidelity: str, 
    path: str, 
    splitted: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load reaction-diffusion data from .mat files.

    Parameters:
    - params (List[float]): A list of parameter values used to identify the files.
    - fidelity (str): A string indicating the fidelity level used to construct the filenames.
    - path (str): The directory path where the .mat files are stored.
    - splitted (bool, optional): A boolean flag indicating if the data is split across multiple files. 
                                  Default is False.

    Returns:
    - Tuple containing:
        - data_u (np.ndarray): The loaded reaction-diffusion data stacked along the 4th axis.
        - x (np.ndarray): The spatial coordinate array.
        - t (np.ndarray): The temporal coordinate array.
    """
    
    u_list = []
    
    for param in params:
        # Construct the filename based on the parameter and fidelity level
        name = path + 'u_' + fidelity + '_' + "{:.3f}".format(param)
        
        if splitted:
            u_test_list = []
            # Load data from multiple files and concatenate along the 3rd axis
            for i in [1, 2]:
                u_test = scipy.io.loadmat(name + '_' + str(i) + '.mat')['u']
                u_test_list.append(u_test)
            u = np.concatenate(u_test_list, axis=2)
        else:
            # Load data from a single file
            u = scipy.io.loadmat(name + '.mat')['u']
        
        u_list.append(u)
    
    # Stack the loaded data along the 4th axis
    data_u = np.stack(u_list, axis=3)
    
    # Load the spatial and temporal coordinate arrays
    x = scipy.io.loadmat(path + 'x_' + fidelity + '.mat')['x']
    t = scipy.io.loadmat(path + 't_' + fidelity + '.mat')['t']
    
    # Return the data and flattened coordinate arrays
    return data_u, x.flatten(), t.flatten()





@njit
def process_data(datahf: np.ndarray, parameters: np.ndarray, t_eval: np.ndarray, Yhf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def HPO_tinyDA(
    datahf: np.ndarray, 
    mean_prior: np.ndarray,
    fwd_LSTM_folder: str,
    cov_prior: np.ndarray, 
    Yhf: np.ndarray, 
    parameters: np.ndarray, 
    rwmh_cov: np.ndarray, 
    rwmh_adaptive: bool, 
    iterations: int, 
    burnin: int, 
    n_chains: int, 
    final_model,  # The model class instance with a method 'param_inverse'
    algo: str, 
    levels: int, 
    force_sequential: bool,
    sigma_noise_bounds: Tuple[float, float],  # Bounds for sigma_noise as a tuple (min, max)
    n_data_bounds: Tuple[int, int],           # Bounds for n_data as a tuple (min, max)
    sigma_bounds: Tuple[float, float],        # Bounds for sigma as a tuple (min, max)
    rwmh_scaling_bounds: Tuple[float, float],  # Bounds for rwmh_scaling as a tuple (min, max)
    forward_low_fidelity: Optional[Callable] = None
) -> Tuple[float, int, float, float]:
    """
    Perform hyperparameter optimization using Bayesian Optimization to minimize the error
    in the param_inverse method of the given final_model.

    Args:
        datahf (np.ndarray): High-fidelity data array.
        mean_prior (np.ndarray): Prior mean vector.
        cov_prior (np.ndarray): Prior covariance matrix.
        Yhf (np.ndarray): High-fidelity observations.
        parameters (np.ndarray): Real parameter values for comparison.
        rwmh_cov (np.ndarray): Covariance matrix for the RWMH proposal distribution.
        rwmh_adaptive (bool): Flag to enable adaptive RWMH.
        iterations (int): Number of iterations for the RWMH algorithm.
        burnin (int): Number of burn-in iterations for the RWMH algorithm.
        n_chains (int): Number of MCMC chains.
        final_model: The model object with a method `param_inverse`.
        algo (str): Algorithm type for the optimization.
        levels (int): Number of levels in the model.
        force_sequential (bool): Flag to enforce sequential processing.
        sigma_noise_bounds (Tuple[float, float]): Bounds for the sigma_noise hyperparameter.
        n_data_bounds (Tuple[int, int]): Bounds for the n_data hyperparameter.
        sigma_bounds (Tuple[float, float]): Bounds for the sigma hyperparameter.
        rwmh_scaling_bounds (Tuple[float, float]): Bounds for the rwmh_scaling hyperparameter.

    Returns:
        Tuple[float, int, float, float]: The optimal values for sigma_noise, n_data, sigma, and rwmh_scaling.
    """
    
    def evaluate_model(sigma_noise: float, n_data: int, sigma: float, rwmh_scaling: float) -> float:
        """
        Objective function that runs the simulation and returns the negative error for minimization.

        Args:
            sigma_noise (float): Noise standard deviation for the covariance noise matrix.
            n_data (int): Number of data points to be used in the simulation.
            sigma (float): Parameter used in the covariance likelihood calculation.
            rwmh_scaling (float): Scaling factor for the Random Walk Metropolis-Hastings algorithm.

        Returns:
            float: The negative of the error to be minimized.
        """
        
        # Running the model's parameter inversion method to compute the error
        _, error, _ = final_model.param_inverse(
            mean_prior=mean_prior, 
            x_data = np.linspace(np.min(datahf[:,0]), np.max(datahf[:,0]), int(n_data)).reshape(-1, 1),  # Generate evaluation times
            max_par=max(datahf[:, 1]), 
            cov_prior=cov_prior, 
            cov_noise=sigma_noise,
            cov_likelihood=calculate_cov_likelihood(sigma, np.linspace(0., 5., int(n_data)).reshape(-1, 1)),
            y_obs=process_data(datahf, parameters, np.linspace(np.min(datahf[:,0]), np.max(datahf[:,0]), int(n_data)).reshape(-1, 1), Yhf)[1],
            x_real=parameters,
            number_chains=n_chains,
            N=iterations,
            burn_in=burnin,
            levels=levels, 
            diagnostic=True, 
            rwmh_cov=rwmh_cov, 
            rmwh_scaling=rwmh_scaling,
            rwmh_adaptive=rwmh_adaptive, 
            algo=algo, 
            forward_low_fidelity=forward_low_fidelity,
            force_sequential=force_sequential,
            fwd_LSTM_folder=fwd_LSTM_folder        )
        
        return -error  # Return the negative error for minimization

    # Define parameter bounds for the Bayesian Optimization based on user input
    pbounds = {
        'sigma_noise': sigma_noise_bounds,  # Bounds for sigma_noise
        'n_data': n_data_bounds,            # Bounds for n_data
        'sigma': sigma_bounds,              # Bounds for sigma
        'rwmh_scaling': rwmh_scaling_bounds # Bounds for rwmh_scaling
    }

    # Initialize the Bayesian Optimizer with the objective function and parameter bounds
    optimizer = BayesianOptimization(
        f=evaluate_model,
        pbounds=pbounds,
        random_state=42
    )

    # Run the optimization process
    optimizer.maximize(init_points=5, n_iter=25)

    # Retrieve the optimal parameter values
    best_params = optimizer.max['params']

    # Return the best parameters found
    return best_params['sigma_noise'], int(best_params['n_data']), best_params['sigma'], best_params['rwmh_scaling']


def run_param_inverse(
    final_model, 
    mean_prior: np.ndarray, 
    fwd_LSTM_folder: str,
    datahf: np.ndarray, 
    cov_prior: np.ndarray, 
    Yhf: np.ndarray, 
    parameters: np.ndarray, 
    sigma_noise: float, 
    n_data: int, 
    sigma: float, 
    rwmh_scaling: float, 
    rwmh_cov: np.ndarray, 
    rwmh_adaptive: bool, 
    iterations: int, 
    burnin: int, 
    n_chains: int, 
    levels: int, 
    algo: str, 
    force_sequential: bool,
    forward_low_fidelity: Optional[Callable] = None

) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Calls the `param_inverse` method of `final_model` with the given parameters.

    Args:
        final_model: The model object with a `param_inverse` method.
        mean_prior (np.ndarray): Prior mean vector.
        datahf (np.ndarray): High-fidelity data array.
        cov_prior (np.ndarray): Prior covariance matrix.
        Yhf (np.ndarray): High-fidelity observations.
        parameters (np.ndarray): Real parameter values for comparison.
        sigma_noise (float): Noise standard deviation for the covariance noise matrix.
        n_data (int): Number of data points to be used in the simulation.
        sigma (float): Parameter used in the covariance likelihood calculation.
        rwmh_scaling (float): Scaling factor for the Random Walk Metropolis-Hastings algorithm.
        rwmh_cov (np.ndarray): Covariance matrix for the RWMH proposal distribution.
        rwmh_adaptive (bool): Flag to enable adaptive RWMH.
        iterations (int): Number of iterations for the RWMH algorithm.
        burnin (int): Number of burn-in iterations for the RWMH algorithm.
        n_chains (int): Number of MCMC chains.
        levels (int): Number of levels in the model.
        algo (str): Algorithm type for the optimization.
        force_sequential (bool): Flag to enforce sequential processing.

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: The inferred parameters, error, and diagnostic data.
    """
    
    # Prepare the observational data and other required inputs for the param_inverse call
    t_eval = np.linspace(np.min(datahf[:,0]), np.max(datahf[:,0]), n_data).reshape(-1, 1)  # Generate evaluation times
    max_par = np.max(datahf[:, 1])
    
    # Calculate the covariance for the likelihood
    cov_likelihood = calculate_cov_likelihood(sigma, t_eval)
    
    # Process the data to get y_obs
    _, y_obs = process_data(datahf, parameters, t_eval, Yhf)
    
    # Call the `param_inverse` method from the `final_model`
    inferred_parameters, error, diagnostics = final_model.param_inverse(
        mean_prior=mean_prior,
        x_data=t_eval,
        max_par=max_par,
        cov_prior=cov_prior,
        cov_noise=sigma_noise,
        cov_likelihood=cov_likelihood,
        y_obs=y_obs,
        x_real=parameters,
        number_chains=n_chains,
        N=iterations,
        burn_in=burnin,
        levels=levels,
        diagnostic=True,
        rwmh_cov=rwmh_cov,
        rmwh_scaling=rwmh_scaling,
        rwmh_adaptive=rwmh_adaptive,
        algo=algo,
        forward_low_fidelity=forward_low_fidelity,
        force_sequential=force_sequential,
        fwd_LSTM_folder=fwd_LSTM_folder     
    )
    
    # Return the inferred parameters, error, and diagnostics
    return inferred_parameters, error, diagnostics


# def run_simulation( 
#                    datahf: np.ndarray, 
#                    mean_prior: np.ndarray, 
#                    fwd_LSTM_folder: str,
#                    cov_prior: np.ndarray, 
#                    Yhf: np.ndarray, 
#                    sigma_noise: List[float], 
#                    n_data: List[int], 
#                    parameters: np.ndarray, 
#                    sigma: np.ndarray, 
#                    rwmh_scaling: np.ndarray, 
#                    rwmh_cov: np.ndarray, 
#                    rwmh_adaptive: bool, 
#                    iterations: int, 
#                    burnin: int, 
#                    n_chains: int, 
#                    final_model: Any, 
#                    algo: str, 
#                    levels:int=1, 
#                    forward_low_fidelity: Optional[Callable] = None,
#                    force_sequential:bool=False

#                    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
#     """
#     Run a simulation to estimate parameters and calculate errors.
    
#     Parameters:
#     - datahf (np.ndarray): 2D array containing data  (t and parameter).
#     - mean_prior (np.ndarray): 1D array for the mean of the prior.
#     - fwd_LSTM_folder (str): name of the folder and files with collection of LSTM models with the relation (mu,t)->u_LF_POD
#     - cov_prior (np.ndarray): 2D array for the covariance of the prior.
#     - Yhf (np.ndarray): 1D array of observed values.
#     - sigma_noise (List[float]): List of noise levels.
#     - n_data (List[int]): List of number of data points along t.
#     - parameters (np.ndarray): 1D array of parameters.
#     - sigma (np.ndarray): 1D array of standard deviations for the likelihood.
#     - rwmh_scaling (np.ndarray): 1D array of scaling factors for the RWMH algorithm.
#     - rwmh_cov (np.ndarray): 2D array for the RWMH covariance.
#     - rwmh_adaptive (bool): Boolean indicating if RWMH is adaptive.
#     - iterations (int): Integer for the number of iterations.
#     - burnin (int): Integer for the burn-in period.
#     - n_chains (int): Integer for the number of chains.
#     - final_model (Any): The model object with the param_inverse method.
#     - algo (str): String indicating the algorithm to use.
#     - levels: number of levels of a Multilevel approach
#     - force_sequenntial (bool): True to avoid parallelization
#     - forward_low_fidelity (Optional[Callable]): Low fidelity forward model function (optional).
    
    
#     Returns:
#     - best_estimate (np.ndarray): The best parameter estimate.
#     - best_error (np.ndarray): The error corresponding to the best estimate.
#     - param_final (List[dict]): list of parameter of MCMC algorithm  

#     """
    
#     # Initialize error and estimate arrays
#     error_shape = (len(sigma_noise), len(n_data), len(sigma), len(rwmh_scaling))
#     error = np.zeros(error_shape)
#     estimates = np.zeros(error_shape)
#     param_final=[]
#     # Iterate over all combinations of parameters using itertools.product
#     for (i, noise), (k, n), (t, s), (j, r) in product(enumerate(sigma_noise), enumerate(n_data), enumerate(sigma), enumerate(rwmh_scaling)):
#         t_eval = np.linspace(np.min(datahf[:,0]), np.max(datahf[:,0]), n).reshape(-1, 1)  # Generate evaluation times
#         nearest_x, y_obs = process_data(datahf, parameters, t_eval, Yhf)  # Get nearest t values and observations
#         cov_likelihood = calculate_cov_likelihood(s, t_eval)  # Compute the covariance for the likelihood
        
#         # Perform parameter estimation and calculate error
#         estimates[i, k,t, j], error[i, k, t, j], par = final_model.param_inverse(
#             mean_prior=mean_prior, 
#             x_data=t_eval, 
#             max_par=max(datahf[:,1]),
#             cov_prior=cov_prior, 
#             rmwh_scaling=r, 
#             cov_noise=noise, 
#             cov_likelihood=cov_likelihood, 
#             y_obs=y_obs, 
#             x_real=parameters, 
#             number_chains=n_chains, 
#             N=iterations, 
#             burn_in=burnin, 
#             levels=levels, 
#             diagnostic=True, 
#             rwmh_cov=rwmh_cov, 
#             rwmh_adaptive=rwmh_adaptive, 
#             algo=algo, 
#             forward_low_fidelity=forward_low_fidelity,
#             force_sequential=force_sequential,
#             fwd_LSTM_folder=fwd_LSTM_folder

#         )
#         param_final.append(par)
#     # Identify the index of the minimum error
#     smallest_index = np.unravel_index(np.argmin(error), error.shape)
#     best_estimate = estimates[smallest_index]
#     best_error = error[smallest_index]
    
#     # Print the best parameters
#     print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, "
#           f"number of data={n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, "
#           f"rwmh_scaling={rwmh_scaling[smallest_index[3]]}")

#     return best_estimate, best_error, param_final