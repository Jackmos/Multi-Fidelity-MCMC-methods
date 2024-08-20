import logging
from typing import Dict, Any, List, Tuple, Union
from numba import njit
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, Nadam, Adamax
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from bayes_opt import BayesianOptimization

# Configure Python logging to suppress detailed Keras messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)


######################## VEDI SE ELIMINNARE

# Enable XLA JIT compilation
tf.config.optimizer.set_jit(True)
#######################################################àà


def sinusoidal_activation(x: tf.Tensor) -> tf.Tensor:
    """
    Apply a sinusoidal activation function to the input tensor.
    
    Parameters:
    - x: TensorFlow tensor, the input tensor.
    
    Returns:
    - A tensor with the same shape as the input, where each element is the square of the sine of the corresponding input element.
    """
    return tf.square(tf.sin(x))

@njit
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



def HPO_tinyDA(
    datahf: np.ndarray, 
    mean_prior: np.ndarray, 
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
    rwmh_scaling_bounds: Tuple[float, float]  # Bounds for rwmh_scaling as a tuple (min, max)
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
            x_data=np.linspace(0., 5., int(n_data)).reshape(-1, 1), 
            max_par=max(datahf[:, 1]), 
            cov_prior=cov_prior, 
            cov_noise=sigma_noise,
            cov_likelihood=calculate_cov_likelihood(sigma, np.linspace(0., 5., int(n_data)).reshape(-1, 1)),
            y_obs=process_data(datahf, parameters, np.linspace(0., 5., int(n_data)).reshape(-1, 1), Yhf)[1],
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
            force_sequential=force_sequential
        )
        
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
    force_sequential: bool
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
    x_obs = np.linspace(0., 5., int(n_data)).reshape(-1, 1)
    max_par = np.max(datahf[:, 1])
    
    # Calculate the covariance for the likelihood
    cov_likelihood = calculate_cov_likelihood(sigma, x_obs)
    
    # Process the data to get y_obs
    _, y_obs = process_data(datahf, parameters, x_obs, Yhf)
    
    # Call the `param_inverse` method from the `final_model`
    inferred_parameters, error, diagnostics = final_model.param_inverse(
        mean_prior=mean_prior,
        x_data=x_obs,
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
        force_sequential=force_sequential
    )
    
    # Return the inferred parameters, error, and diagnostics
    return inferred_parameters, error, diagnostics


def HPO_cuqi(
    data: dict,
    mean_prior: np.ndarray,
    parameters: np.ndarray,
    iterations: int,
    burn_in: int,
    cov_prior: np.ndarray,
    sd_noise_bounds: Tuple[float, float],
    proposal_sd_bounds: Tuple[float, float],
    n_data_bounds: Tuple[int, int],
    scale_bounds: Tuple[float, float],
    adapt: bool,
    number_chains: int,
    algo: str,
    fwd_model,
    parallel: bool,
    init_points: int = 5,
    n_iter: int = 25
) -> Tuple[float, float, int, float, float, float]:
    """
    Run a CUQI simulation to estimate parameters and compute error using Bayesian Optimization.
    
    Parameters:
    - data: Dictionary containing high-fidelity data (keys: "xhf" and "Yhf").
    - mean_prior: 1D numpy array for the mean of the prior.
    - parameters: 1D numpy array of parameter values.
    - iterations: Integer for the number of iterations (samples).
    - burn_in: Integer for the number of burn-in samples.
    - cov_prior: 2D numpy array for the covariance of the prior.
    - sd_noise_bounds: Tuple of (min, max) bounds for noise standard deviation.
    - proposal_sd_bounds: Tuple of (min, max) bounds for proposal standard deviation.
    - n_data_bounds: Tuple of (min, max) bounds for the number of data points.
    - scale_bounds: Tuple of (min, max) bounds for the scaling factor.
    - adapt: Boolean indicating whether to use adaptation in the algorithm.
    - number_chains: Integer for the number of MCMC chains.
    - algo: String indicating the algorithm to use for MCMC.
    - fwd_model: Forward model object with the inverse_cuqi method.
    - parallel: Boolean indicating whether to run MCMC chains in parallel.
    - init_points: Number of initial random points for Bayesian Optimization.
    - n_iter: Number of iterations for the optimization process.

    Returns:
    - Best parameters found through Bayesian Optimization: (best_estimate, best_error, best_sd_noise, best_proposal_sd, best_n_data, best_scale).
    """
    
    def evaluate_model(sd_noise: float, proposal_sd: float, n_data: int, scale: float) -> float:
        """
        Objective function that runs the simulation and returns the negative error for minimization.
        
        Args:
            sd_noise (float): Noise standard deviation for the covariance noise matrix.
            proposal_sd (float): Proposal standard deviation for the MCMC algorithm.
            n_data (int): Number of data points to be used in the simulation.
            scale (float): Scaling factor for the Random Walk Metropolis-Hastings algorithm.
            
        Returns:
            float: The negative of the error to be minimized.
        """
        
        # Generate evaluation points in the input domain based on the number of data points
        x_data = np.linspace(0., 5., int(n_data)).reshape(-1, 1)
        _, y_obs = process_data(data["xhf"], parameters, x_data, data["Yhf"])

        # Perform parameter estimation and calculate the error
        _, error, _ = fwd_model.inverse_cuqi(
            mean_prior=mean_prior,
            x_real=parameters,
            max_par=max(data["xhf"][:,1]),
            y_obs=y_obs,
            N=iterations,
            burn_in=burn_in,
            cov_prior=cov_prior,
            sd_noise=sd_noise,
            adapt=adapt,
            scale=scale,
            proposal_sd=proposal_sd,
            number_chains=number_chains,
            algo=algo,
            x_data=x_data,
            parallel=parallel
        )
        
        return -error  # Returning negative error for minimization by Bayesian Optimization

    # Define the bounds for each parameter in the optimization process
    pbounds = {
        'sd_noise': sd_noise_bounds,
        'proposal_sd': proposal_sd_bounds,
        'n_data': (n_data_bounds[0], n_data_bounds[1]),
        'scale': scale_bounds
    }

    # Initialize Bayesian Optimizer with the objective function and bounds
    optimizer = BayesianOptimization(
        f=evaluate_model,
        pbounds=pbounds,
        random_state=42
    )

    # Run the optimization process
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # Retrieve the optimal parameter values
    best_params = optimizer.max['params']
    
    # Extract the best parameters found
    best_sd_noise = best_params['sd_noise']
    best_proposal_sd = best_params['proposal_sd']
    best_n_data = int(best_params['n_data'])
    best_scale = best_params['scale']
    
    # Rerun the model with the best parameters to get the final estimates and error
    x_data = np.linspace(0., 5., best_n_data).reshape(-1, 1)
    nearest_x, y_obs = process_data(data["xhf"], parameters, x_data, data["Yhf"])

    best_estimate, best_error, params_result = fwd_model.inverse_cuqi(
        mean_prior=mean_prior,
        x_real=parameters,
        max_par=max(data["xhf"][:,1]),
        y_obs=y_obs,
        N=iterations,
        burn_in=burn_in,
        cov_prior=cov_prior,
        sd_noise=best_sd_noise,
        adapt=adapt,
        scale=best_scale,
        proposal_sd=best_proposal_sd,
        number_chains=number_chains,
        algo=algo,
        x_data=x_data,
        parallel=parallel
    )

    # Print the best parameters
    print(f"Best parameters found: sd_noise={best_sd_noise}, proposal_sd={best_proposal_sd}, "
          f"n_data={best_n_data}, scale={best_scale}")

    return best_estimate, best_error, best_sd_noise, best_proposal_sd, best_n_data, best_scale


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

def getOpti(name: str, lr: float) -> Union[str, Adam, Nadam, Adamax]:
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
        'standardadam': 'adam'
    }
    
    if name not in optimizers:
        raise ValueError(f"Optimizer name '{name}' is not recognized. Valid options are: {list(optimizers.keys())}")
    
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
    
    # Define the input layer with the given number of input features
    inputs = Input(shape=(num_inputs,))

    # High-Fidelity (HF) Model
    if name == 'HF':
        hidden1 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2(params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HF')(hidden1)

    # Low-Fidelity (LF) Model
    elif name == 'LF':
        hidden = inputs
        for i in range(4):
            hidden = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden{i+1}')(hidden)
        output = Dense(num_outputs, activation='linear', name='LF')(hidden)

    # Single Model
    elif name == 'Single':
        hidden = inputs
        for i in range(4):
            hidden = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'],
                           kernel_regularizer=l2(params['l2weight']), name=f'{name}_hidden{i+1}')(hidden)
        output = Dense(num_outputs, activation='linear', name='Single')(hidden)

    # High-Fidelity Linear (Hflin) Model
    elif name == 'Hflin':
        hidden = Dense(64, activation='linear', kernel_regularizer=l2(params['l2weight']),
                       kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HFlin')(hidden)

    # High-Fidelity Periodic (Hfper) Model
    elif name == "Hfper":
        hidden = inputs
        for i in range(2):
            hidden = Dense(64, activation=sinusoidal_activation, kernel_regularizer=l2(params["l2weight"]),
                           kernel_initializer=params["kernel_init"], name=f'{name}_hidden{i+1}')(hidden)
        output = Dense(num_outputs, activation="linear", name="HFper")(hidden)

    # Intermediate (Inter) Model
    elif name == 'Inter':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
        outputLF = Dense(1, activation='linear', name='LF')(hidden2)
        outputadd = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                          kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(hidden2)
        merge = concatenate([outputLF, outputadd])
        hidden3 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(merge)
        outputHF = Dense(1, activation='linear', name='HF')(hidden3)
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





















# #######################################à

# from bayes_opt import BayesianOptimization
# import numpy as np

# def optimize_parameters(datahf, 
#                         mean_prior, 
#                         cov_prior, 
#                         Yhf, 
#                         parameters, 
#                         rwmh_cov, 
#                         rwmh_adaptive, 
#                         iterations, 
#                         burnin, 
#                         n_chains, 
#                         final_model, 
#                         algo, 
#                         levels, 
#                         force_sequential
#                         ):
    
#     def evaluate_model(sigma_noise, n_data, sigma, rwmh_scaling):
#         # Funzione obiettivo che esegue la simulazione e restituisce l'errore negativo (per minimizzazione)

#         _, error, _ = final_model.param_inverse(
#             mean_prior, 
#             np.linspace(0., 5., int(n_data)).reshape(-1, 1), 
#             max_par=max(datahf[:,1]),
#             cov_prior=cov_prior, 
#             cov_noise=sigma_noise,
#             cov_likelihood=calculate_cov_likelihood(sigma, np.linspace(0., 5., int(n_data)).reshape(-1, 1)),
#             y_obs=process_data(datahf, parameters, np.linspace(0., 5., int(n_data)).reshape(-1, 1), Yhf)[1],
#             x_real=parameters,
#             number_chains=n_chains,
#             N=iterations,
#             burn_in=burnin,
#             levels=levels, 
#             diagnostic=True, 
#             rwmh_cov=rwmh_cov, 
#             rmwh_scaling=rwmh_scaling,
#             rwmh_adaptive=rwmh_adaptive, 
#             algo=algo, 
#             force_sequential=force_sequential
#         )
#         return -error  # Minimizzazione dell'errore

#     # Definizione dei limiti dei parametri
#     pbounds = {
#         'sigma_noise': (0.001, 1.0),  # Limiti per sigma_noise
#         'n_data': (10, 1000),  # Limiti per n_data
#         'sigma': (0.001, 1.0),  # Limiti per sigma
#         'rwmh_scaling': (0.01, 2.0)  # Limiti per rwmh_scaling
#     }

#     # Inizializzazione dell'ottimizzatore bayesiano
#     optimizer = BayesianOptimization(
#         f=evaluate_model,
#         pbounds=pbounds,
#         random_state=42
#     )

#     # Esecuzione dell'ottimizzazione
#     optimizer.maximize(init_points=5, n_iter=25)

#     # Ottieni i parametri ottimali
#     best_params = optimizer.max['params']
    
#     return best_params['sigma_noise'], int(best_params['n_data']), best_params['sigma'], best_params['rwmh_scaling']







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
#     force_sequential: bool = False
# ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    
#     # Invece del ciclo for, chiamiamo la funzione di ottimizzazione
#     best_sigma_noise, best_n_data, best_sigma, best_rwmh_scaling = optimize_parameters(
#         datahf, mean_prior, cov_prior, Yhf, parameters, rwmh_cov, rwmh_adaptive, iterations, burnin, n_chains, final_model, algo, levels, force_sequential
#     )

#     # Eseguire la simulazione con i parametri ottimizzati
#     best_estimate, best_error, param_results = final_model.param_inverse(
#         mean_prior, 
#         np.linspace(0., 5., best_n_data).reshape(-1, 1), 
#         max_par=max(datahf[:,1]),
#         cov_prior=cov_prior, 
#         cov_noise=best_sigma_noise,
#         cov_likelihood=calculate_cov_likelihood(best_sigma, np.linspace(0., 5., best_n_data).reshape(-1, 1)),
#         y_obs=process_data(datahf, parameters, np.linspace(0., 5., best_n_data).reshape(-1, 1), Yhf)[1],
#         x_real=parameters,
#         number_chains=n_chains,
#         N=iterations,
#         burn_in=burnin,
#         levels=levels, 
#         diagnostic=True, 
#         rwmh_cov=rwmh_cov, 
#         rmwh_scaling=best_rwmh_scaling,
#         rwmh_adaptive=rwmh_adaptive, 
#         algo=algo, 
#         force_sequential=force_sequential
#     )

#     # Stampa i migliori parametri trovati
#     print(f"Migliori parametri trovati: sigma_noise={best_sigma_noise}, n_data={best_n_data}, sigma={best_sigma}, rwmh_scaling={best_rwmh_scaling}")
    
#     return best_estimate, best_error, param_results





















