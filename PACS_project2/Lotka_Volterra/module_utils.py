import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from bayes_opt import BayesianOptimization
from tensorflow.keras.optimizers import Adam, Nadam, Adamax

from typing import List, Tuple, Union, Dict, Optional
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


# def multilevel_observations(levels:int,
#                             y_obs:np.ndarray,
#                             parameters:np.ndarray,
#                             t_eval:np.ndarray,
#                             final_model
#                             ) -> Union[np.ndarray, List[np.ndarray]]:
#     """
#     In a multilevel case scenario create proper observations for each level
#     Parameters:
#     - levels: number of levels of the ML algorithm
#     - y_obs: numpy array containing "real" observations.
#     - parameters: 1D numpy array of parameter values to find in datahf.
#     - t_eval: 2D numpy array of evaluation times.
#     - final_model: The model object with a method `param_inverse`.

#     Returns:
#     - y_obs: numpy array  or List (if multilevel) of corresponding y values from Yhf.
#     """
#     y=[]

#     for l in range(levels-1):
#         y.append(final_model.model_list[l]._wrapper_prediction(np.hstack(t_eval,np.tile(parameters,t_eval.shape))))                   #################
   
#     return y.append(y_obs)




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


def get_algorithm_specific_bounds_tiny(algo: str, levels: int, rwmh_scaling, rwmh_cov, subsampling_rate) -> dict:
    """
    In HPO_tinyDA, manage algo dependent parameters .

    Args:
        algo (str): Algorithm type for the optimization.
        levels (int): Number of levels in the model.
        rwmh_scaling: hyperparameter, can be a float (fixed), a tuple (bounds), or None.
        rwmh_cov:  Covariance matrix for the RWMH proposal distribution.
        subsampling_rate:  Optional Rate or rates of subsampling the posterior. Default is 1

    Returns:
        dict: Dictionary with required parameters depending on the algorithm
    """
    algo_dependent_params = {}

    if algo == 'MH':  
        if rwmh_scaling is None:
            raise ValueError(f"The algorithm '{algo}' requires the parameter 'rwmh_scaling' which is missing.")
        if rwmh_cov is None:
            raise ValueError(f"The algorithm '{algo}' requires the parameter 'rwmh_cov' which is missing.")
        algo_dependent_params['rwmh_scaling'] = rwmh_scaling
        algo_dependent_params['rwmh_cov'] = rwmh_cov

    elif algo == 'AM':  
        if rwmh_cov is None:
            raise ValueError(f"The algorithm '{algo}' requires the parameter 'rwmh_cov' which is missing.")
        algo_dependent_params['rwmh_cov'] = rwmh_cov
        algo_dependent_params['rwmh_scaling'] = 1.0  # generic value

    elif algo == 'CN':  
        if rwmh_scaling is None:
            raise ValueError(f"The algorithm '{algo}' requires the parameter 'rwmh_scaling' which is missing.")
        algo_dependent_params['rwmh_scaling'] = rwmh_scaling
        algo_dependent_params['rwmh_cov'] = 1.  # generic value

    elif algo == 'DREAMZ':  
        algo_dependent_params['rwmh_scaling'] = 1.0  # generic value
        algo_dependent_params['rwmh_cov'] =1.  # generic value
    else:
        raise ValueError(f"Unrecognized algorithm '{algo}' specified.")

   
    if levels > 1:
        if subsampling_rate is None:
            raise ValueError(f"With levels > 1, 'subsampling_rate' is required for the algorithm '{algo}'.")
        algo_dependent_params['subsampling_rate'] = subsampling_rate
    else:
        algo_dependent_params['subsampling_rate'] = 1  # generic value

    return algo_dependent_params




def multilevel_observations(levels:int,y_obs:np.ndarray,parameters:np.ndarray,t_eval:np.ndarray,final_model)-> List[np.ndarray] :
    y=[]

    y.append(final_model.model_list[0].prediction(np.hstack((t_eval,np.tile(parameters,t_eval.shape)))))                   #################
    for l in range(1,levels-1):
        y.append(final_model.model_list[l].prediction( np.hstack( (np.hstack( (t_eval,np.tile(parameters,t_eval.shape) ) ) ,y[l-1] ) )))
    y.append(y_obs)
    return y




def HPO_tinyDA(
    datahf: np.ndarray, 
    mean_prior: np.ndarray,
    Yhf: np.ndarray, 
    parameters: np.ndarray, 
    rwmh_adaptive: bool, 
    iterations: int, 
    burnin: int, 
    n_chains: int, 
    final_model,                          # The model class instance with a method 'param_inverse'
    algo: str, 
    levels: int, 
    domain:Tuple[float,float],
    force_sequential: bool,
    cov_prior:  np.ndarray,  # Required parameter
    n_data: Union[int, Tuple[int, int]],  # Required parameter
    sigma_noise: Union[float, Tuple[float, float]],  # Required parameter
    sigma: Union[float, Tuple[float, float]],  # Required parameter
    rwmh_scaling: Optional[Union[float, Tuple[float, float]]] = None,  # Optional parameter
    rwmh_cov: Optional[Union[float, Tuple[float, float]]] = None,  # Optional parameter
    subsampling_rate:Optional[Union[Union[int,List[int]], Union[Tuple[int, int],Tuple[List[int],List[int]]]]] = None,
    init_points:int=5,
    n_iter:int=25
) -> dict:
    """
    Perform hyperparameter optimization using Bayesian Optimization to minimize the error
    in the param_inverse method of the given final_model.

    Args:
        datahf (np.ndarray): High-fidelity data array.
        mean_prior (np.ndarray): Prior mean vector.
        cov_prior (np.ndarray): Prior covariance matrix.
        Yhf (np.ndarray): High-fidelity observations.
        parameters (np.ndarray): Real parameter values for comparison.
        rwmh_adaptive (bool): Flag to enable adaptive RWMH.
        iterations (int): Number of iterations for the RWMH algorithm.
        burnin (int): Number of burn-in iterations for the RWMH algorithm.
        n_chains (int): Number of MCMC chains.
        final_model: The model object with a method `param_inverse`.
        algo (str): Algorithm type for the optimization.
        levels (int): Number of levels in the model.
        domain (Tuple[float, float]): domain of the considered problem
        force_sequential (bool): Flag to enforce sequential processing.

        n_data: Required hyperparameter, can be an integer (fixed) or a tuple (bounds for optimization).
        sigma_noise: Required hyperparameter, can be a float (fixed) or a tuple (bounds for optimization).
        sigma: Required hyperparameter, can be a float (fixed) or a tuple (bounds for optimization).
        rwmh_scaling: Optional hyperparameter, can be a float (fixed), a tuple (bounds), or None.
        rwmh_cov: Optional hyperparameter,  Covariance of the RWMH proposal distribution can be a matrix (fixed), a tuple of matrices (bounds), or None
        subsampling_rate: Optional Rate or rates of subsampling the posterior. Default is 1.

        init_points: Number of initial random points for Bayesian Optimization.
        n_iter: Number of iterations for the optimization process.
    Returns:
        dict: The optimal or fixed values for the hyperparameters.
    """
    
    # List of required parameters
    required_params = {
        'n_data': n_data,
        'sigma_noise': sigma_noise,
        'sigma': sigma
    }
    
    # Checking if required parameters are provided
    for param_name, param_value in required_params.items():
        if param_value is None:
            raise ValueError(f"The required parameter '{param_name}' is missing.")
    
    # Check algorithm-dependent parameters
    algo_dependent_params = get_algorithm_specific_bounds_tiny(algo, levels, rwmh_scaling, rwmh_cov, subsampling_rate)

    def evaluate_model(**model_kwargs) -> float:
        """
        Objective function that runs the simulation and returns the negative error for minimization.

        Args:
            **model_kwargs: Hyperparameters passed to the param_inverse method.

        Returns:
            float: The negative of the error to be minimized.
        """
        y_obs=process_data(datahf, parameters, 
                               np.linspace(domain[0], domain[1], int(model_kwargs.get('n_data',  required_params["n_data"]))).reshape(-1, 1), Yhf)[1]
        
        if final_model.is_istance_MF():
            y_obs=multilevel_observations(levels,y_obs,parameters,np.linspace(domain[0], domain[1], int(model_kwargs.get('n_data',  required_params["n_data"]))).reshape(-1, 1), final_model) 

        # Running the model's parameter inversion method to compute the error
        _, error, _ = final_model.param_inverse(
            mean_prior=mean_prior, 
            x_data=np.linspace(domain[0], domain[1], int(model_kwargs.get('n_data', required_params["n_data"]))).reshape(-1, 1),  
            max_par=max(datahf[:, 1]), 
            cov_prior=cov_prior,
            cov_noise=model_kwargs.get('sigma_noise', required_params["sigma_noise"]),
            cov_likelihood=calculate_cov_likelihood(model_kwargs.get('sigma',  required_params["sigma"]), 
                                                    np.linspace(domain[0], domain[1], int(model_kwargs.get('n_data',  required_params["n_data"]))).reshape(-1, 1)),
            y_obs=y_obs,
            x_real=parameters,
            number_chains=n_chains,
            N=iterations,
            burn_in=burnin,
            levels=levels, 
            diagnostic=True, 
            rwmh_cov=model_kwargs.get('rwmh_cov', algo_dependent_params["rwmh_cov"]), 
            rmwh_scaling=model_kwargs.get('rwmh_scaling', algo_dependent_params["rwmh_scaling"]),
            subsampling_rate=model_kwargs.get('subsampling_rate', algo_dependent_params["subsampling_rate"]),
            rwmh_adaptive=rwmh_adaptive, 
            algo=algo, 
            force_sequential=force_sequential
        )
        
        return -error  # Return the negative error for minimization

    # Define parameter bounds for the Bayesian Optimization based on **kwargs
    pbounds = {k: v for k, v in required_params.items() if isinstance(v, tuple)}  
    pbounds.update({k: v for k, v in algo_dependent_params.items() if isinstance(v, tuple)})

    # Check if there are parameters to optimize
    if pbounds:
        # Initialize the Bayesian Optimizer with the objective function and parameter bounds
        optimizer = BayesianOptimization(
            f=evaluate_model,
            pbounds=pbounds,
            random_state=42
        )

        # Run the optimization process
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        # Retrieve the optimal parameter values
        best_params = optimizer.max['params']

        # Merge optimized parameters with fixed ones from kwargs
        best_params = {**best_params, **{k: v for k, v in required_params.items() if not isinstance(v, tuple)}}  
        best_params.update({k: v for k, v in algo_dependent_params.items() if not isinstance(v, tuple)})  
    else:
        # No optimization required, use fixed values from kwargs
        best_params = {**required_params, **algo_dependent_params}

    return best_params



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
    domain:Tuple[float,float],
    subsampling_rate:Optional[Union[Union[int,List[int]], Union[Tuple[int, int],Tuple[List[int],List[int]]]]] = None,


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
        subsampling_rate: Optional Rate or rates of subsampling the posterior. Default is 1.

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
        force_sequential=force_sequential,
        subsampling_rate=subsampling_rate
    )
    
    # Return the inferred parameters, error, and diagnostics
    return inferred_parameters, error, diagnostics




def HPO_cuqi(
    data: np.ndarray,
    Yhf: np.ndarray, 
    mean_prior: np.ndarray,
    parameters: np.ndarray,
    iterations: int,
    burn_in: int,
    cov_prior: np.ndarray,
    domain:Tuple[float,float],
    sd_noise: Union[float, Tuple[float, float]],  # Mandatory, optimize if tuple
    proposal_sd: Union[float, Tuple[float, float]],  # Mandatory, optimize if tuple
    n_data: Union[int, Tuple[int, int]],  # Mandatory, optimize if tuple
    scale: Optional[Union[float, Tuple[float, float]]] = None,  # Optional, optimize if tuple and algo == 'MH'
    adapt: bool = False,
    number_chains: int = 1,
    algo: str = "MH",
    fwd_model=None,
    parallel: bool = False,
    init_points: int = 5,
    n_iter: int = 25
) -> Tuple[float, float, float, float, int, float]:
    
    """
    Run a CUQI simulation to estimate parameters and compute error using Bayesian Optimization.
    
    Parameters:
    - data: Dictionary containing high-fidelity data (keys: "xhf" and "Yhf").
    - mean_prior: 1D numpy array for the mean of the prior.
    - parameters: 1D numpy array of parameter values.
    - iterations: Integer for the number of iterations (samples).
    - burn_in: Integer for the number of burn-in samples.
    - cov_prior: 2D numpy array for the covariance of the prior.
    - domain (Tuple[float, float]): domain of the considered problem
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
        """
        
        # Generate evaluation points in the input domain based on the number of data points
        x_data = np.linspace(domain[0], domain[1], int(n_data)).reshape(-1, 1)
        _, y_obs = process_data(data, parameters, x_data, Yhf)

        # Perform parameter estimation and calculate the error
        _, error, _ = fwd_model.inverse_cuqi(
            mean_prior=mean_prior,
            x_real=parameters,
            max_par=max(data[:, 1]),
            y_obs=y_obs,
            N=iterations,
            burn_in=burn_in,
            cov_prior=cov_prior,  # Fixed if scalar
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

    # Get the parameter bounds based on the algorithm
    pbounds = get_algorithm_specific_bounds_cuqi(
        algo=algo,
        sd_noise=sd_noise,
        proposal_sd=proposal_sd,
        n_data=n_data,
        scale=scale
    )

    # Initialize Bayesian Optimizer with the objective function and bounds if any optimization is required
    if pbounds:
        optimizer = BayesianOptimization(
            f=evaluate_model,
            pbounds=pbounds,
            random_state=42
        )

        # Run the optimization process
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        # Retrieve the optimal parameter values
        best_params = optimizer.max['params']
    else:
        # No optimization needed, use fixed parameters
        best_params = {}

    # Extract the best parameters found or use fixed values
    best_sd_noise = best_params.get('sd_noise', sd_noise)
    best_proposal_sd = best_params.get('proposal_sd', proposal_sd)
    best_n_data = int(best_params.get('n_data', n_data))
    best_scale = best_params.get('scale', scale if scale is not None else 1.0)  # Default scale if not provided

    # Rerun the model with the best parameters to get the final estimates and error
    x_data = np.linspace(domain[0], domain[1], best_n_data).reshape(-1, 1)
    _, y_obs = process_data(data["xhf"], parameters, x_data, data["Yhf"])

    best_estimate, best_error, params_result = fwd_model.inverse_cuqi(
        mean_prior=mean_prior,
        x_real=parameters,
        max_par=max(data["xhf"][:, 1]),
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

def run_inverse_cuqi(
    data: np.ndarray,
    Yhf: np.ndarray,
    mean_prior: np.ndarray,
    parameters: np.ndarray,
    iterations: int,
    burn_in: int,
    domain:Tuple[float,float],
    cov_prior: np.ndarray,
    sd_noise: float,
    proposal_sd: float,
    n_data: int,
    scale: Optional[float] = None,
    adapt: bool = False,
    number_chains: int = 1,
    algo: str = "MH",
    fwd_model=None,
    parallel: bool = False
) -> Tuple[float, float, dict]:
    """
    Run a single CUQI simulation with given parameters to estimate parameters and compute error.
    
    Parameters:
    - data: 2D numpy array containing high-fidelity data (keys: "xhf" and "Yhf").
    - Yhf: 2D numpy array of high-fidelity outputs corresponding to `data`.
    - mean_prior: 1D numpy array for the mean of the prior.
    - parameters: 1D numpy array of parameter values.
    - iterations: Integer for the number of iterations (samples).
    - burn_in: Integer for the number of burn-in samples.
    - domain (Tuple[float, float]): domain of the considered problem
    - cov_prior: 2D numpy array for the covariance of the prior.
    - sd_noise: Float for the noise standard deviation.
    - proposal_sd: Float for the proposal standard deviation.
    - n_data: Integer for the number of data points.
    - scale: Float for the scaling factor (optional).
    - adapt: Boolean indicating whether to use adaptation in the algorithm.
    - number_chains: Integer for the number of MCMC chains.
    - algo: String indicating the algorithm to use for MCMC.
    - fwd_model: Forward model object with the `inverse_cuqi` method.
    - parallel: Boolean indicating whether to run MCMC chains in parallel.
    
    Returns:
    - A tuple containing the best estimate, the associated error, and additional parameter results.
    """

    # Generate evaluation points in the input domain based on the number of data points
    x_data = np.linspace(domain[0], domain[1], n_data).reshape(-1, 1)
    
    # Process the data
    _, y_obs = process_data(data, parameters, x_data, Yhf)

    # Perform parameter estimation using the inverse_cuqi method
    best_estimate, best_error, params_result = fwd_model.inverse_cuqi(
        mean_prior=mean_prior,
        x_real=parameters,
        max_par=max(data[:, 1]),
        y_obs=y_obs,
        N=iterations,
        burn_in=burn_in,
        cov_prior=cov_prior,
        sd_noise=sd_noise,
        adapt=adapt,
        scale=scale if scale is not None else 1.0,
        proposal_sd=proposal_sd,
        number_chains=number_chains,
        algo=algo,
        x_data=x_data,
        parallel=parallel
    )

    # Print the parameters used
    print(f"Parameters used: sd_noise={sd_noise}, proposal_sd={proposal_sd}, "
          f"n_data={n_data}, scale={scale if scale is not None else 1.0}")

    return best_estimate, best_error, params_result

def get_algorithm_specific_bounds_cuqi(
    algo: str,
    sd_noise: Union[float, Tuple[float, float]],
    proposal_sd: Union[float, Tuple[float, float]],
    n_data: Union[int, Tuple[int, int]],
    scale: Optional[Union[float, Tuple[float, float]]] = None
) -> dict:
    """
    Determines the bounds for optimization based on the algorithm and input parameters.
    
    Parameters:
    - algo: The algorithm to be used (e.g., 'MH', 'AM', etc.).
    - sd_noise: The noise standard deviation (either a fixed value or a tuple of bounds).
    - proposal_sd: The proposal standard deviation (either a fixed value or a tuple of bounds).
    - n_data: The number of data points (either a fixed value or a tuple of bounds).
    - scale: The scaling factor (either a fixed value or a tuple of bounds).
    
    Returns:
    - A dictionary containing the bounds for the parameters that should be optimized.
    """

    pbounds = {
        'sd_noise': sd_noise if isinstance(sd_noise, tuple) else None,
        'proposal_sd': proposal_sd if isinstance(proposal_sd, tuple) else None,
        'n_data': n_data if isinstance(n_data, tuple) else None
    }

    # Add scale to pbounds only if algo is 'MH' and scale is provided as a tuple
    if algo == 'MH' and isinstance(scale, tuple):
        pbounds['scale'] = scale

    # Remove None values from pbounds (those parameters that should not be optimized)
    pbounds = {k: v for k, v in pbounds.items() if v is not None}

    return pbounds



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

