import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from bayes_opt import BayesianOptimization

from typing import List, Tuple, Union, Dict
from numba import njit



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
    domain:Tuple[float,float],
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
        domain (Tuple[float, float]): domain of the considered problem
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
            x_data=np.linspace(domain[0], domain[1], int(n_data)).reshape(-1, 1), 
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
    force_sequential: bool,
    domain:Tuple[float,float]

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
        domain (Tuple[float, float]): domain of the considered problem

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: The inferred parameters, error, and diagnostic data.
    """
    
    # Prepare the observational data and other required inputs for the param_inverse call
    x_obs = np.linspace(domain[0], domain[1], int(n_data)).reshape(-1, 1)
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
    domain:Tuple[float,float],
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
    - domain (Tuple[float, float]): domain of the considered problem

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
        x_data = np.linspace(domain[0], domain[1], int(n_data)).reshape(-1, 1)
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





def run_param_inverse_cuqi(
    fwd_model, 
    data: Dict[str, np.ndarray], 
    parameters: np.ndarray, 
    mean_prior: np.ndarray, 
    cov_prior: np.ndarray, 
    iterations: int, 
    burn_in: int, 
    adapt: bool, 
    number_chains: int, 
    algo: str, 
    parallel: bool, 
    sd_noise: float, 
    proposal_sd: float, 
    domain:Tuple[float,float],
    n_data: int, 
    scale: float
) -> float:
    """
    Objective function that runs the simulation and returns the negative error for minimization.

    Args:
        fwd_model: The model object with an `inverse_cuqi` method.
        data (Dict[str, np.ndarray]): Dictionary containing the high-fidelity data arrays ("xhf" and "Yhf").
        parameters (np.ndarray): Real parameter values for comparison.
        mean_prior (np.ndarray): Prior mean vector.
        cov_prior (np.ndarray): Prior covariance matrix.
        iterations (int): Number of iterations for the MCMC algorithm.
        burn_in (int): Number of burn-in iterations for the MCMC algorithm.
        adapt (bool): Flag to enable adaptive MCMC.
        number_chains (int): Number of MCMC chains.
        algo (str): Algorithm type for the optimization.
        parallel (bool): Flag to enable parallel processing.
        sd_noise (float): Noise standard deviation for the covariance noise matrix.
        proposal_sd (float): Proposal standard deviation for the MCMC algorithm.
        n_data (int): Number of data points to be used in the simulation.
        scale (float): Scaling factor for the Random Walk Metropolis-Hastings algorithm.

    Returns:
        float: The negative of the error to be minimized.
    """
    
    # Generate evaluation points in the input domain based on the number of data points
    x_data = np.linspace(domain[0], domain[1], n_data).reshape(-1, 1)

    # Process the data to obtain the observed values
    _, y_obs = process_data(data["xhf"], parameters, x_data, data["Yhf"])

    # Perform parameter estimation and calculate the error using the inverse_cuqi method
    _, error, _ = fwd_model.inverse_cuqi(
        mean_prior=mean_prior,
        x_real=parameters,
        max_par=np.max(data["xhf"][:, 1]),
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
    
    # Return the negative of the error for minimization
    return -error




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

