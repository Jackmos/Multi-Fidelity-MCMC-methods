from utils.functions_to_ray import *
from utils.helper_functions import Clean
import arviz as az
import logging
import numpy as np
import os
import ray
import tensorflow as tf
import tinyDA as tda
import uuid
import warnings
from bayes_opt import BayesianOptimization
import warnings


from cuqi.distribution import JointDistribution
from cuqi.sampler import MH, NUTS, pCN
from datetime import datetime
from matplotlib import pyplot as plt
from numba import njit
from tinyDA import AdaptiveMetropolis, CrankNicolson, DREAMZ, GaussianRandomWalk, get_MAP, sample, to_inference_data
from typing import Any, List, Tuple, Union, Callable, Optional, Dict
from abc import ABC, abstractmethod


# Suppress specific UserWarnings and RuntimeWarnings
warnings.filterwarnings("ignore", category=UserWarning, message="qoi group is not defined in the InferenceData scheme")
warnings.filterwarnings("ignore", category=UserWarning, message="Your data appears to have a single value or no finite values")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")



def create_folder_name(base_name: str, max_length: int = 255) -> str:
    """
    Creates a unique folder name based on the base name.
    
    Ensures the folder name is valid for most file systems by replacing 
    forbidden characters and ensuring the name doesn't exceed the max length.

    Args:
        base_name (str): The base name for the folder.
        max_length (int): Maximum length for the folder name (default is 255).

    Returns:
        str: The path to the created folder.
    """
    # Replace forbidden characters
    folder_name = base_name.replace(" ", "").replace("[", "").replace("]", "").replace(".", "p").replace(",", "_")
    
    # Ensure folder name doesn't exceed max length
    if len(folder_name) > max_length:
        folder_name = folder_name[:max_length - 9] + "_" + uuid.uuid4().hex[:8]
    
    # Create the directory if it doesn't exist
    # os.makedirs(folder_name, exist_ok=True)
    
    return folder_name




class INetwork_protection:
    """
    A wrapper class for INetwork that protects its attributes from being modified.

    This class delegates attribute access and method calls to the underlying INetwork
    instance (`_forward_NN`). It ensures that the internal state of the network cannot
    be altered after initialization.
    """

    def __init__(self, forward_NN: 'INetwork'):
        """
        Initializes the INetwork_protection with a given INetwork instance.

        :param forward_NN: An instance of INetwork that this class will protect.
        """
        self._forward_NN = forward_NN

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access and method calls to the underlying forward model.

        :param name: The attribute name or method to access.
        :return: The corresponding attribute or method from the `_forward_NN`.
        """
        return getattr(self._forward_NN, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Prevents modification of attributes, except for the `_forward_NN` itself.

        :param name: The attribute name.
        :param value: The value to set.
        :raises AttributeError: If attempting to modify any attribute other than `_forward_NN`.
        """
        if name == "_forward_NN":
            super().__setattr__(name, value)
        else:
            raise AttributeError("Cannot modify attributes of the network")


class BayesianLibraryBase:
    """
    A base class for Bayesian libraries that handles common operations like HPO (Hyperparameter Optimization) and Run.

    This class is designed to be subclassed, where specific algorithm implementations
    should define `_hpo_specific` and `_run_specific` methods.
    """

    __slots__ = ['algorithm', 'forward_NN','best_params']

    def __init__(self, algorithm: str, forward_NN: 'INetwork'):
        """
        Initializes the BayesianLibraryBase with a specific algorithm and a forward neural network.

        :param algorithm: The name of the algorithm being used.
        :param forward_NN: An instance of INetwork to be used in the library.
        """
        self.algorithm = algorithm
        self.forward_NN = forward_NN

    def hpo(self, *args: Any, **kwargs: Any) -> None:
        """
        Executes Hyperparameter Optimization (HPO) specific to the algorithm.

        :param args: Positional arguments passed to the specific HPO method.
        :param kwargs: Keyword arguments passed to the specific HPO method.
        """
        self._hpo_specific(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> None:
        """
        Executes the run operation specific to the algorithm.

        :param args: Positional arguments passed to the specific run method.
        :param kwargs: Keyword arguments passed to the specific run method.
        """
        self._run_specific(*args, **kwargs)

    def _hpo_specific(self, *args: Any, **kwargs: Any) -> None:
        """
        Placeholder for HPO logic that should be implemented by subclasses.

        :raises NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Metodo specifico non implementato")

    def _run_specific(self, *args: Any, **kwargs: Any) -> None:
        """
        Placeholder for run logic that should be implemented by subclasses.

        :raises NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Metodo specifico non implementato")
    
    @staticmethod
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


    @staticmethod
    @njit
    def process_data(inputs_HF: np.ndarray, real_parameter: np.ndarray, 
                     x_domain_grid: np.ndarray, output_HF: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the elements of a dataset nearest to the ones given.

        This method finds the x-values in the dataset that are closest to a specified grid of domain values,
        and retrieves the corresponding y-values (observations).

        :param inputs_HF: 2D numpy array where inputs_HF[:,1] contains parameter values.
        :param real_parameter: 1D numpy array of parameter values to find in inputs_HF.
        :param x_domain_grid: 2D numpy array of evaluation times (domain values).
        :param output_HF: 1D numpy array with output data observations.
        :return: A tuple containing:
            - nearest_x: 1D numpy array of x values closest to each t_eval.
            - y_obs: 1D numpy array of corresponding y values from output_HF.
        :raises ValueError: If no observations related to the parameter are found.
        """
        # Create a boolean array to store matching rows
        match_condition = np.ones(inputs_HF.shape[0], dtype=np.bool_)

        # Loop through each parameter and apply the matching condition (check if available observations related to parameter)
        for i in range(real_parameter.shape[0]):
            match_condition &= (inputs_HF[:, i + 1] == real_parameter[i])

        # Get the indices where all parameter values match
        indices = np.where(match_condition)[0]
        if len(indices) == 0:
            raise ValueError(f"No observations related to parameter: {real_parameter[0]}")
        
        # Find values on the domain as similar as possible to a uniform grid in the data
        datahf_values = inputs_HF[indices, 0].reshape(-1, 1)
        t_eval_values = x_domain_grid.reshape(1, -1)
        differences = np.abs(datahf_values - t_eval_values)
        closest_indices = np.argmin(differences, axis=0)
        nearest_x = inputs_HF[indices[closest_indices], 0]
        selected_observations = output_HF[indices[closest_indices]]
        
        return nearest_x, selected_observations

    @staticmethod
    def setup_optimizer(evaluate_model: Callable[..., float], pbounds: Dict[str, Tuple[float, float]], 
                        initial_points_optimizer: int, iterations_optimizer: int) -> Dict[str, float]:
        """
        Set up and execute the Bayesian Optimization process.

        :param evaluate_model: A callable that evaluates the model and returns a score.
        :param pbounds: A dictionary defining the bounds for each parameter.
        :param initial_points_optimizer: The number of initial random points to explore.
        :param iterations_optimizer: The number of optimization iterations to perform.
        :return: A dictionary of the best parameters found during optimization.
        """
        optimizer = BayesianOptimization(
            f=evaluate_model,
            pbounds=pbounds,
            random_state=42
        )
        optimizer.maximize(init_points=initial_points_optimizer, n_iter=iterations_optimizer)
        return optimizer.max['params']

    def update_best_params(self, best_params: dict, iterations: int, n_chains: int)->None:
        """
        Update the dictionary best_params. If an entry with the same
        iterations, n_chains, and algo exists, it will be replaced.
        Otherwise, a new entry will be added.

        Parameters:
        - best_params (dict): Dictionary containing parameters including 'iterations', 'n_chains', and 'algo'.
        - iterations (int): Number of iterations.
        - n_chains (int): Number of chains.

        Returns:
        - None
        """
        # Create a tuple key based on iterations, n_chains, and algo
        key = (self.algorithm,n_chains)
        
        # Update the dictionary with the specific parameters
        best_params["iterations"] = iterations
        best_params["num_chains"] = n_chains
        best_params["algorithm"] = self.algorithm

        # Insert or update the entry with the key
        self.best_params[key] = best_params

    def check_saved_param(self, algo: str, n: int, param_name: str, param_value):
        """
        Return the parameter value from self.best_params if the provided value is None.
        The key in self.best_params is a tuple (algo, n).
        
        Args:
            algo (str): The algorithm identifier.
            n (int): The integer identifier.
            param_name (str): The name of the parameter to check.
            param_value: The current value of the parameter.
            
        Returns:
            The parameter value from self.best_params if param_value is None, 
            otherwise the provided param_value.
        """
        key = (algo, n)
        if param_value is not None:
            return param_value
        
        if key in self.best_params:
            return self.best_params[key].get(param_name)
        
        return None

class Inversion_tinyDA(BayesianLibraryBase):
    """
    A class for performing Bayesian inversion using the tinyDA library.

    This class specializes in hyperparameter optimization (HPO) and running
    the inversion process using specific algorithms supported by the tinyDA library.
    """

    __slots__ = ["best_params"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call parent class __init__ 

        # Initialize best_params if it has not been initialized
        if not hasattr(self, 'best_params'):
            self.best_params = {}            # has to be initialized for HPO

    def _hpo_specific(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """
        Executes the specific HPO (Hyperparameter Optimization) for the algorithm.

        :param args: Positional arguments for the HPO method.
        :param kwargs: Keyword arguments for the HPO method.
        :return: A dictionary of the best hyperparameters found.
        """
        print(f"Executing specific HPO for {self.algorithm} with Library TinyDA")
        return self.HPO_tinyDA(*args, **kwargs)

    def _run_specific(self, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Executes the specific run process for the algorithm.

        :param args: Positional arguments for the run method.
        :param kwargs: Keyword arguments for the run method.
        :return: A tuple containing the inferred parameters, error, and diagnostic data.
        """
        print(f"Executing specific run for {self.algorithm} with Library TinyDA")
        return self.run_inverse_tiny(*args, **kwargs)


    def multilevel_observations(self,levels:int,observations:np.ndarray,parameters:np.ndarray,x_data:np.ndarray)-> List[np.ndarray] :
        y=[]

        y.append(self.forward_NN.model_list[0].prediction(np.hstack((x_data,np.tile(parameters,x_data.shape)))))                   #################
        for l in range(1,levels-1):
            y.append(self.forward_NN.model_list[l].prediction( np.hstack( (np.hstack( (x_data,np.tile(parameters,x_data.shape) ) ) ,y[l-1] ) )))
        y.append(observations)
        return y


    def HPO_tinyDA(self, 
                   inputs_HF: np.ndarray, 
                   mean_prior: np.ndarray,
                   output_HF: np.ndarray, 
                   real_parameters: np.ndarray,
                   rwmh_adaptive: bool, 
                   iterations: int, 
                   burn_in: int,
                   domain_bounds: Tuple[float, float], 
                   force_sequential: bool,
                   cov_prior: np.ndarray, 
                   number_data: Union[int, Tuple[int, int]],
                   sigma_noise: Union[float, Tuple[float, float]],
                   sigma: Union[float, Tuple[float, float]],
                   levels: int=1,
                   n_chains: int=1, 
                   rwmh_scaling: Optional[Union[float, Tuple[float, float]]] = None,
                   rwmh_covariance: Optional[Union[float, Tuple[float, float]]] = None,
                   subsampling_rate: Optional[Union[int, List[int], Union[Tuple[int, int], Tuple[List[int], List[int]]]]] = None,
                   initial_points_optimizer: int = 5,
                   iterations_optimizer: int = 25,
                   **kwargs: Any) -> Dict[str, float]:
        """
        Perform hyperparameter optimization using Bayesian Optimization.

        :param inputs_HF: High-fidelity input data.
        :param mean_prior: Prior mean vector.
        :param output_HF: High-fidelity output data.
        :param real_parameters: Real parameter values for comparison.
        :param rwmh_adaptive: Boolean flag for adaptive Random Walk Metropolis-Hastings (RWMH).
        :param iterations: Number of iterations for the RWMH algorithm.
        :param burn_in: Number of burn-in iterations for the RWMH algorithm.
        :param n_chains: Number of MCMC chains.
        :param domain_bounds: Bounds for the domain grid.
        :param levels: Number of levels in the model.
        :param force_sequential: Boolean flag to enforce sequential processing.
        :param cov_prior: Prior covariance matrix.
        :param number_data: Number of data points to use in the simulation.
        :param sigma_noise: Noise standard deviation for the covariance noise matrix.
        :param sigma: Parameter used in the covariance likelihood calculation.
        :param rwmh_scaling: Optional scaling factor for the RWMH algorithm.
        :param rwmh_covariance: Optional covariance matrix for the RWMH proposal distribution.
        :param subsampling_rate: Optional rate or rates of subsampling the posterior.
        :param initial_points_optimizer: Number of initial points for the optimizer.
        :param iterations_optimizer: Number of iterations for the optimizer.
        :param kwargs: for LSTM to pass support elements
        :return: A dictionary of the best hyperparameters found.
        """
        
        # Check required parameters
        required_params = {
            'n_data': number_data, 
            'sigma_noise': sigma_noise, 
            'sigma': sigma
        }
        self._check_required_params(required_params)


        # Define parameter bounds for Bayesian Optimization
        pbounds,support = self._get_pbounds(required_params, rwmh_scaling, rwmh_covariance, subsampling_rate, levels)



        def evaluate_model(**model_kwargs: Any) -> float:
            """
            Objective function that runs the simulation and returns the negative error for minimization.

            :param model_kwargs: Hyperparameters passed to the param_inverse method.
            :return: The negative of the error to be minimized.
            """
            x_data,observations=self.process_data(inputs_HF, real_parameters, 
                                np.linspace(domain_bounds[0], domain_bounds[1], int(model_kwargs.get('n_data',  support["n_data"]))).reshape(-1, 1), output_HF)
            # Running the model's parameter inversion method to compute the error


            if self.forward_NN.is_instance_MF() and levels>1:
                observations=self.multilevel_observations(levels,observations,real_parameters,np.linspace(domain_bounds[0], domain_bounds[1], int(model_kwargs.get('n_data', support["n_data"]))).reshape(-1, 1)) 


            _, error, _ = self.forward_NN.param_inverse(
                mean_prior=mean_prior, 
                x_data= np.linspace(domain_bounds[0], domain_bounds[1], int(model_kwargs.get('n_data',  support["n_data"]))).reshape(-1, 1), 
                max_parameter=np.max(inputs_HF[:, 1:]), 
                cov_prior=cov_prior,
                cov_noise=model_kwargs.get('sigma_noise', support["sigma_noise"]),
                cov_likelihood=self.calculate_cov_likelihood(model_kwargs.get('sigma', support["sigma"]), 
                                                        np.linspace(domain_bounds[0], domain_bounds[1], int(model_kwargs.get('n_data', support["n_data"]))).reshape(-1, 1)),
                observation=observations,
                x_real=real_parameters,
                number_chains=n_chains,
                iterations=iterations,
                burn_in=burn_in,
                levels=levels, 
                diagnostic=True,
                plot_result=False, 
                rwmh_covariance=model_kwargs.get('rwmh_cov', support["rwmh_cov"]), 
                rmwh_scaling=model_kwargs.get('rwmh_scaling', support["rwmh_scaling"]),
                subsampling_rate=int(model_kwargs.get('subsampling_rate', support["subsampling_rate"])),
                rwmh_adaptive=rwmh_adaptive, 
                proposal_algorithm=self.algorithm, 
                force_sequential=force_sequential,
                **kwargs
            )
            
            return -error  # Return the negative error for minimization



        # Optimize or use fixed parameters
        if pbounds:
            best_params =BayesianLibraryBase.setup_optimizer(evaluate_model, pbounds, initial_points_optimizer, iterations_optimizer)
        else:
            best_params = {**required_params}
        best_params["n_data"]=int(best_params["n_data"])


        self.update_best_params(best_params,iterations,n_chains)



        return best_params
    


    def get_params_by_algo(self, algo: str) -> list:
        """
        Retrieve all dictionary entries where the 'algorithm' key matches the provided string.

        :param algo : The algorithm name to search for.

        :raises results : A list of dictionaries where the 'algorithm' matches the provided algo.
        """
        results = []
        for key, value in self.best_params.items():
            if value['algorithm'] == algo:
                results.append(value)
        return results

    def _check_required_params(self, required_params: Dict[str, Union[int, float, Tuple[float, float]]]) -> None:
        """
        Ensure all required parameters are provided.

        :param required_params: A dictionary of required parameters and their values.
        :raises ValueError: If any required parameter is missing.
        """
        for param_name, param_value in required_params.items():
            if param_value is None:
                raise ValueError(f"The required parameter '{param_name}' is missing.")

    def _get_pbounds(self,
                     required_params: Dict[str, Union[int, Tuple[float, float]]], 
                     rwmh_scaling: Optional[Union[float, Tuple[float, float]]],
                     rwmh_covariance: Optional[Union[float, Tuple[float, float]]], 
                     subsampling_rate: Optional[Union[int, List[int], Union[Tuple[int, int], Tuple[List[int], List[int]]]]],
                     levels: int = 1) -> Dict[str, Tuple[float, float]]:
        """
        Define the parameter bounds for Bayesian Optimization.

        :param required_params: A dictionary of required parameters with potential bounds.
        :param rwmh_scaling: Optional scaling factor for RWMH.
        :param rwmh_covariance: Optional covariance matrix for RWMH.
        :param subsampling_rate: Optional rate or rates of subsampling.
        :param levels: Number of levels in the model.
        :return: A dictionary of parameter bounds for optimization.
        """
        pbounds = {k: v for k, v in required_params.items() if isinstance(v, tuple)}
        support = {k: v for k, v in required_params.items()}
        

        algo_dependent_params = Inversion_tinyDA._get_algorithm_specific_bounds_tiny(self.algorithm, levels, rwmh_scaling, rwmh_covariance, subsampling_rate)
        pbounds.update({k: v for k, v in algo_dependent_params.items() if isinstance(v, tuple)})
        support.update({k: v for k, v in algo_dependent_params.items() if isinstance(v, tuple)})

        support.update({k: v for k, v in algo_dependent_params.items() if np.isscalar(v)})
        support.update({k: 1 for k, v in algo_dependent_params.items() if not isinstance(v, tuple)and not np.isscalar(v)})    # check se ambiguo con passi dopo 

        return pbounds, support
    
    def run_inverse_tiny(self, 
                         mean_prior: np.ndarray, 
                         inputs_HF: np.ndarray, 
                         cov_prior: np.ndarray, 
                         output_HF: np.ndarray, 
                         domain_bounds: Tuple[float, float],
                         real_parameters: np.ndarray, 
                         iterations: int, 
                         burn_in: int, 
                         n_chains: int, 
                         levels: int, 
                         force_sequential: bool,
                         sigma_noise: float=None, 
                         number_data: int=None, 
                         sigma: float=None,                          
                         rwmh_scaling: float=None, 
                         rwmh_covariance: np.ndarray=None, 
                         rwmh_adaptive: bool=None,                          
                         subsampling_rate: Union[int, List[int]]=None,
                         **kwargs: Any
                         ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Calls the `param_inverse` method of the forward model with the given parameters.

        :param mean_prior: Prior mean vector.
        :param inputs_HF: High-fidelity input data.
        :param cov_prior: Prior covariance matrix.
        :param output_HF: High-fidelity output data.
        :param domain_bounds: Bounds for the domain grid.
        :param real_parameters: Real parameter values for comparison.
        :param sigma_noise: Noise standard deviation for the covariance noise matrix.
        :param number_data: Number of data points to be used in the simulation.
        :param sigma: Parameter used in the covariance likelihood calculation.
        :param rwmh_scaling: Scaling factor for the Random Walk Metropolis-Hastings algorithm.
        :param rwmh_covariance: Covariance matrix for the RWMH proposal distribution.
        :param rwmh_adaptive: Boolean flag for adaptive RWMH.
        :param iterations: Number of iterations for the RWMH algorithm.
        :param burn_in: Number of burn-in iterations for the RWMH algorithm.
        :param n_chains: Number of MCMC chains.
        :param levels: Number of levels in the model.
        :param subsampling_rate: Rate or rates of subsampling the posterior.
        :param force_sequential: Boolean flag to enforce sequential processing.
        :return: A tuple containing the inferred parameters, error, and diagnostic data.
        """
        # Use self.get_param to handle parameters potentially in self.best_params
        sigma_noise = self.check_saved_param(self.algorithm,n_chains,'sigma_noise', sigma_noise)
        number_data = self.check_saved_param(self.algorithm,n_chains,'n_data', number_data)
        sigma = self.check_saved_param(self.algorithm,n_chains,'sigma', sigma)
        rwmh_scaling = self.check_saved_param(self.algorithm,n_chains,'rwmh_scaling', rwmh_scaling)
        rwmh_covariance = self.check_saved_param(self.algorithm,n_chains,'rwmh_covariance', rwmh_covariance)
        rwmh_adaptive = self.check_saved_param(self.algorithm,n_chains,'rwmh_adaptive', rwmh_adaptive)
        subsampling_rate = self.check_saved_param(self.algorithm,n_chains,'subsampling_rate', subsampling_rate)

        # Prepare the observational data and other required inputs for the param_inverse call
        x_obs = np.linspace(domain_bounds[0], domain_bounds[1], int(number_data)).reshape(-1, 1)
        max_parameter = np.max(inputs_HF[:, 1:])

        # Calculate the covariance for the likelihood
        cov_likelihood = self.calculate_cov_likelihood(sigma, x_obs)
        
        # Process the data to get y_obs
        _, observations = Inversion_tinyDA.process_data(inputs_HF, real_parameters, x_obs, output_HF)
        

        if self.forward_NN.is_instance_MF() and levels>1:
            observations=self.multilevel_observations(levels,observations,real_parameters,np.linspace(domain_bounds[0], domain_bounds[1], number_data).reshape(-1, 1)) 


        # Call the `param_inverse` method from the forward model
        inferred_parameters, error, diagnostics = self.forward_NN.param_inverse(
            mean_prior=mean_prior,
            x_data=x_obs,
            max_parameter=max_parameter,
            cov_prior=cov_prior,
            cov_noise=sigma_noise,
            cov_likelihood=cov_likelihood,
            observation=observations,
            x_real=real_parameters,
            number_chains=n_chains,
            iterations=iterations,
            burn_in=burn_in,
            levels=levels,
            diagnostic=True,
            plot_result=True,
            rwmh_covariance=rwmh_covariance,
            rmwh_scaling=rwmh_scaling,
            rwmh_adaptive=rwmh_adaptive,
            proposal_algorithm=self.algorithm,
            subsampling_rate=subsampling_rate,
            force_sequential=force_sequential,
            **kwargs
        )
        
        # Return the inferred parameters, error, and diagnostics
        return inferred_parameters, error, diagnostics

    @staticmethod
    def _get_algorithm_specific_bounds_tiny(algorithm: str, levels: int, rwmh_scaling: Optional[Union[float, Tuple[float, float]]] = None,
                                            rwmh_cov: Optional[Union[float, Tuple[float, float]]] = None,
                                            subsampling_rate: Optional[Union[int, List[int], Union[Tuple[int, int], Tuple[List[int], List[int]]]]] = None) -> Dict[str, Tuple[float, float]]:
        """
        Retrieve algorithm-specific parameter bounds for Bayesian Optimization.

        :param algorithm: The name of the algorithm.
        :param levels: Number of levels in the model.
        :param rwmh_scaling: Optional scaling factor for RWMH.
        :param rwmh_cov: Optional covariance matrix for RWMH.
        :param subsampling_rate: Optional rate or rates of subsampling.
        :return: A dictionary of parameter bounds.
        """
        algo_instance = ProposalAlgorithmFactory.create_algorithm(algorithm, levels, subsampling_rate)
        return algo_instance.get_parameters(rwmh_scaling=rwmh_scaling, rwmh_cov=rwmh_cov)




class Inversion_cuqi(BayesianLibraryBase):
    __slots__ = ["best_params"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call parent class __init__ 

        # Initialize best_params if it has not been initialized
        if not hasattr(self, 'best_params'):
            self.best_params = {}            # has to be initialized for HPO

    def _hpo_specific(self, *args, **kwargs):
        print(f"Executing specific HPO for {self.algorithm} with Library CuqiPy")

        return self.HPO_cuqi( *args, **kwargs)
    def _run_specific(self, *args, **kwargs):
        print(f"Executing specific run for {self.algorithm} with Library CuqiPy")

        return self.run_inverse_cuqi(*args, **kwargs)
        



    def HPO_cuqi(self,
        inputs_HF: np.ndarray,
        mean_prior: np.ndarray,
        output_HF: np.ndarray, 
        real_parameters: np.ndarray,
        iterations: int,
        burn_in: int,
        domain_bounds:Tuple[float,float],
        cov_prior: np.ndarray,
        sd_noise: Union[float, Tuple[float, float]],  # Mandatory, optimize if tuple
        proposal_sd: Union[float, Tuple[float, float]],  # Mandatory, optimize if tuple
        num_data: Union[int, Tuple[int, int]],  # Mandatory, optimize if tuple
        scale: Optional[Union[float, Tuple[float, float]]] = None,  # Optional, optimize if tuple and algo == 'MH'
        adapt: bool = False,
        number_chains: int = 1,
        parallel: bool = False,
        initial_points_optimizer: int = 5,
        iterations_optimizer: int = 25
    ) -> dict:
        
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
        # Check required parameters
        required_params = {
            'n_data': num_data, 
            'sd_noise': sd_noise, 
            'proposal_sd': proposal_sd
        }
        
        self._check_required_params(required_params)


        def evaluate_model(**model_kwargs: Any) -> float:
            """
            Objective function that runs the simulation and returns the negative error for minimization.
            """
            
            # Generate evaluation points in the input domain based on the number of data points
            x_data,observations=self.process_data(inputs_HF, real_parameters, 
                                np.linspace(domain_bounds[0], domain_bounds[1], int(model_kwargs.get('n_data', 100))).reshape(-1, 1), output_HF)
    
            # Perform parameter estimation and calculate the error
            _, error, _ = self.forward_NN.inverse_cuqi(
                mean_prior=mean_prior,
                x_real=real_parameters,
                max_parameter=max(inputs_HF[:, 1]),
                observation=observations,
                iterations=iterations,
                burn_in=burn_in,
                diagnostic=True,
                plot_result=False, 
                cov_prior=cov_prior,  # Fixed if scalar
                sd_noise=model_kwargs.get('sd_noise', 0.1),
                adapt=adapt,
                scale=model_kwargs.get('scale', 0.3),
                proposal_sd=model_kwargs.get('proposal_sd', 0.5),
                number_chains=number_chains,
                proposal_algorithm=self.algorithm,
                x_data=x_data.reshape(-1,1),
                parallel=parallel
            )
            
            return -error  # Returning negative error for minimization by Bayesian Optimization

        # Get the parameter bounds based on the algorithm
        pbounds = self._get_pbounds(required_params, scale )

        # Initialize Bayesian Optimizer with the objective function and bounds if any optimization is required
        pbounds = self._get_pbounds(required_params,scale)

        # Optimize or use fixed parameters
        if pbounds:
            best_params = Inversion_cuqi.setup_optimizer(evaluate_model, pbounds, initial_points_optimizer, iterations_optimizer)
        else:
            best_params = {**required_params}
        best_params["n_data"]=int(best_params["n_data"])

        self.update_best_params(best_params,iterations,number_chains)
        return best_params
    

    def _check_required_params(self, required_params) -> None:
        """
        Ensure all required parameters are provided.

        :param required_params: A dictionary of required parameters and their values.
        :raises ValueError: If any required parameter is missing.
        """
        for param_name, param_value in required_params.items():
            if param_value is None:
                raise ValueError(f"The required parameter '{param_name}' is missing.")

    def _get_pbounds(self,
                     required_params: Dict[str, Union[int, Tuple[float, float]]], 
                     scale: float = 1) -> Dict[str, Tuple[float, float]]:

        pbounds = {k: v for k, v in required_params.items() if isinstance(v, tuple)}
        algo_dependent_params = Inversion_cuqi._get_algorithm_specific_bounds_cuqi(self.algorithm, scale)
        pbounds.update({k: v for k, v in algo_dependent_params.items() if isinstance(v, tuple)})
        return pbounds



    def run_inverse_cuqi(self,
        inputs_HF: np.ndarray,
        output_HF: np.ndarray,
        mean_prior: np.ndarray,
        real_parameters: np.ndarray,
        domain_bounds:Tuple[float,float],
        iterations: int,
        burn_in: int,
        cov_prior: np.ndarray,
        sd_noise: float=None,
        proposal_sd: float=None,
        num_data: int=None,
        scale: Optional[float] = None,
        adapt: bool = False,
        number_chains: int = 1,
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

        sd_noise = self.check_saved_param(self.algorithm,number_chains,'sd_noise', sd_noise)
        number_data = self.check_saved_param(self.algorithm,number_chains,'n_data', num_data)
        proposal_sd = self.check_saved_param(self.algorithm,number_chains,'proposal_sd', proposal_sd)
        scale = self.check_saved_param(self.algorithm,number_chains,'scale', scale)
        x_data = np.linspace(domain_bounds[0], domain_bounds[1], int(number_data)).reshape(-1, 1)

        # Process the data
        x_data, observations = Inversion_cuqi.process_data(inputs_HF, real_parameters, x_data, output_HF)

    
        # Perform parameter estimation using the inverse_cuqi method
        inferred_parameters, error, diagnostics = self.forward_NN.inverse_cuqi(
            mean_prior=mean_prior,
            x_real=real_parameters,
            max_parameter=np.max(inputs_HF[:, 1:]),
            observation=observations,
            iterations=iterations,
            burn_in=burn_in,
            cov_prior=cov_prior,
            sd_noise=sd_noise,
            diagnostic=True,
            plot_result=True, 
            adapt=adapt,
            scale=scale if scale is not None else 1.0,
            proposal_sd=proposal_sd,
            number_chains=number_chains,
            proposal_algorithm=self.algorithm,
            x_data=x_data.reshape(-1,1),
            parallel=parallel
        )


        return inferred_parameters, error, diagnostics

    @staticmethod
    def _get_algorithm_specific_bounds_cuqi(algorithm: str, proposal_sd, scale=None) -> dict:
        
        algo_instance = ProposalAlgorithmFactory.create_algorithm(algorithm)


        return algo_instance.get_parameters(proposal_sd=proposal_sd,scale=scale)


  



class LibraryConnection:
    """
    A factory class to create the appropriate algorithm library based on the given algorithm name.

    This class maps algorithm names to their respective library classes and creates an instance of the library.
    """

    __slots__ = []

    @staticmethod
    def create_algorithm(algorithm_name: str, forward_NN: 'INetwork') -> BayesianLibraryBase:
        """
        Creates an instance of the appropriate library class based on the algorithm name.

        :param algorithm_name: The name of the algorithm to use.
        :param forward_NN: An instance of INetwork to be used in the library.
        :return: An instance of a subclass of BayesianLibraryBase corresponding to the algorithm.
        :raises ValueError: If the algorithm name is not supported.
        """
        algorithm_to_library_map = {
            "AM": Inversion_tinyDA,
            "CN": Inversion_tinyDA,
            "DREAMZ": Inversion_tinyDA,
            "MH_cuqi": Inversion_cuqi,
            "MH_tiny": Inversion_tinyDA,
            "NUTS": Inversion_cuqi,
            "pCN": Inversion_cuqi
        }

        if algorithm_name not in algorithm_to_library_map:
            raise ValueError(f"Algorithm {algorithm_name} not supported")
        
        # Select the correct library class based on the algorithm
        library_class = algorithm_to_library_map[algorithm_name]

        return library_class(algorithm_name, forward_NN)


class BayesianInverseProblem_NN:
    """
    A high-level class for solving Bayesian Inverse Problems using a specified algorithm and a neural network.

    This class acts as a facade, allowing users to set an algorithm and then run the HPO and Run methods.
    """

    __slots__ = ['strategy', 'forward_NN']

    def __init__(self, algorithm_name: str, forward_NN: 'INetwork'):
        """
        Initializes the BayesianInverseProblem_NN with a specific algorithm and neural network.

        :param algorithm_name: The name of the algorithm to use.
        :param forward_NN: An instance of INetwork to be used for the Bayesian inverse problem.
        """
        self.forward_NN = INetwork_protection(forward_NN)
        self.set_algorithm(algorithm_name, forward_NN)

    def set_algorithm(self, algorithm_name: str, forward_NN: 'INetwork' = None) -> None:
        """
        Sets or updates the algorithm to be used for solving the Bayesian inverse problem.

        :param algorithm_name: The name of the algorithm to use.
        :param forward_NN: An optional new INetwork instance to use (if provided).
        """
        self.strategy = LibraryConnection.create_algorithm(algorithm_name, forward_NN)

    def hpo(self, *args: Any, **kwargs: Any) -> None:
        """
        Executes the HPO (Hyperparameter Optimization) using the currently set algorithm.

        :param args: Positional arguments for the HPO method.
        :param kwargs: Keyword arguments for the HPO method.
        """
        return self.strategy.hpo(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> None:
        """
        Executes the run process using the currently set algorithm.

        :param args: Positional arguments for the run method.
        :param kwargs: Keyword arguments for the run method.
        """
        return self.strategy.run(*args, **kwargs)


####################################à
#CLASSES FOR ALGORITHMS
# created for mantainability 
# Base class for all proposal algorithms
class ProposalAlgorithm(ABC):
    """
    Abstract base class for proposal algorithms used in Bayesian optimization and MCMC sampling.

    This class defines the structure for algorithms that generate proposals for 
    the next steps in the sampling process, with provisions for setting up 
    specific parameters and ensuring subsampling rate consistency.
    """

    def __init__(self, levels: int=1, subsampling_rate: Optional[Union[int, Tuple[int, int]]] = None):
        """
        Initialize the ProposalAlgorithm with the number of levels and an optional subsampling rate.

        :param levels: Number of levels in the model.
        :param subsampling_rate: Optional subsampling rate, required if levels > 1.
        """
        self.levels = levels
        self.subsampling_rate = subsampling_rate

    @abstractmethod
    def get_parameters(self, *args, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        """
        Abstract method to retrieve algorithm-specific parameters.

        Subclasses must implement this method to return the specific parameters 
        required by the algorithm.
        """
        pass
    
    @abstractmethod
    def setup_proposal(self, *args, **kwargs) -> object:
        """
        Abstract method to set up the proposal distribution based on algorithm-specific requirements.

        Subclasses must implement this method to initialize the proposal distribution.
        """
        pass

    def _check_subsampling_rate(self) -> Union[int, Tuple[int, int]]:
        """
        Check the subsampling rate to ensure it is set if required by the levels.

        :raises ValueError: If levels > 1 and subsampling_rate is not provided.
        :return: The subsampling rate if levels > 1, otherwise 1.
        """
        if self.levels > 1 and self.subsampling_rate is None:
            raise ValueError("With levels > 1, 'subsampling_rate' is required.")
        return self.subsampling_rate if self.levels > 1 else 1


class MH_tinyAlgorithm(ProposalAlgorithm):
    """
    Implementation of the Metropolis-Hastings (MH) algorithm with a tiny proposal distribution.
    """

    def get_parameters(self, rwmh_scaling: float, rwmh_cov: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Retrieve the parameters specific to the MH_tiny algorithm.

        :param rwmh_scaling: Scaling factor for the RWMH algorithm.
        :param rwmh_cov: Covariance matrix for the RWMH proposal distribution.
        :return: A dictionary containing the relevant parameters.
        :raises ValueError: If required parameters are missing.
        """
        if rwmh_scaling is None:
            raise ValueError("The algorithm 'MH_tiny' requires the parameter 'rwmh_scaling' which is missing.")
        if rwmh_cov is None:
            raise ValueError("The algorithm 'MH_tiny' requires the parameter 'rwmh_cov' which is missing.")
        
        return {
            'rwmh_scaling': rwmh_scaling,
            'rwmh_cov': rwmh_cov,
            'subsampling_rate': self._check_subsampling_rate()
        }

    def setup_proposal(self, *args, **kwargs) -> object:
        """
        Set up the proposal distribution for the MH_tiny algorithm.

        :param args: Positional arguments for setup.
        :param kwargs: Keyword arguments, expected to contain 'rwmh_scaling', 'rwmh_cov', and 'adaptive'.
        :return: The initialized proposal distribution object.
        :raises Warning: If required parameters are missing.
        """
        rwmh_scaling = kwargs.get('rwmh_scaling')
        rwmh_cov = kwargs.get('rwmh_cov')
        adaptive = kwargs.get('adaptive', False)

        if rwmh_scaling is None or rwmh_cov is None:
            warnings.warn("'rwmh_scaling' or 'rwmh_cov' is missing in MH_tinyAlgorithm's setup_proposal.")
        
        return GaussianRandomWalk(C=rwmh_cov * np.eye(kwargs.get('num_params')), scaling=rwmh_scaling, adaptive=adaptive)


class AMAlgorithm(ProposalAlgorithm):
    """
    Implementation of the Adaptive Metropolis (AM) algorithm.
    """

    def get_parameters(self, rwmh_scaling: float, rwmh_cov: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Retrieve the parameters specific to the AM algorithm.

        :param rwmh_scaling: Scaling factor for the RWMH algorithm.
        :param rwmh_cov: Covariance matrix for the RWMH proposal distribution.
        :return: A dictionary containing the relevant parameters.
        :raises ValueError: If required parameters are missing.
        """
        if rwmh_cov is None:
            raise ValueError("The algorithm 'AM' requires the parameter 'rwmh_cov' which is missing.")
        
        return {
            'rwmh_cov': rwmh_cov,
            'rwmh_scaling': 1.0,  # Generic value for AM algorithm
            'subsampling_rate': self._check_subsampling_rate()
        }

    def setup_proposal(self, *args, **kwargs) -> object:
        """
        Set up the proposal distribution for the AM algorithm.

        :param args: Positional arguments for setup.
        :param kwargs: Keyword arguments, expected to contain 'rwmh_cov', 'adaptive', 'period', and 't0'.
        :return: The initialized proposal distribution object.
        :raises Warning: If required parameters are missing.
        """
        rwmh_cov = kwargs.get('rwmh_cov')
        adaptive = kwargs.get('adaptive', False)
        period = kwargs.get('period')
        t0 = kwargs.get('t0')

        if rwmh_cov is None or period is None or t0 is None:
            warnings.warn("'rwmh_cov', 'period', or 't0' is missing in AMAlgorithm's setup_proposal.")

        # Assuming AdaptiveMetropolis is defined elsewhere in your code
        return AdaptiveMetropolis(C0=rwmh_cov * np.eye(kwargs.get('num_params')), adaptive=adaptive, period=period, t0=t0)


class CNAlgorithm(ProposalAlgorithm):
    """
    Implementation of the Crank-Nicolson (CN) algorithm.
    """

    def get_parameters(self, rwmh_scaling: float, rwmh_cov: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Retrieve the parameters specific to the CN algorithm.

        :param rwmh_scaling: Scaling factor for the RWMH algorithm.
        :param rwmh_cov: Covariance matrix for the RWMH proposal distribution.
        :return: A dictionary containing the relevant parameters.
        :raises ValueError: If required parameters are missing.
        """
        if rwmh_scaling is None:
            raise ValueError("The algorithm 'CN' requires the parameter 'rwmh_scaling' which is missing.")
        
        return {
            'rwmh_scaling': rwmh_scaling,
            'rwmh_cov': np.eye(1),  # Generic value for CN algorithm
            'subsampling_rate': self._check_subsampling_rate()
        }

    def setup_proposal(self, *args, **kwargs) -> object:
        """
        Set up the proposal distribution for the CN algorithm.

        :param args: Positional arguments for setup.
        :param kwargs: Keyword arguments, expected to contain 'rwmh_scaling', 'adaptive', and 'period'.
        :return: The initialized proposal distribution object.
        :raises Warning: If required parameters are missing.
        """
        rwmh_scaling = kwargs.get('rwmh_scaling')
        adaptive = kwargs.get('adaptive', False)
        period = kwargs.get('period')

        if rwmh_scaling is None or period is None:
            warnings.warn("'rwmh_scaling' or 'period' is missing in CNAlgorithm's setup_proposal.")

        # Assuming CrankNicolson is defined elsewhere in your code
        return CrankNicolson(scaling=rwmh_scaling, adaptive=adaptive, period=period)


class DREAMZAlgorithm(ProposalAlgorithm):
    """
    Implementation of the DREAM(ZS) algorithm.
    """

    def get_parameters(self, rwmh_scaling: float, rwmh_cov: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Retrieve the parameters specific to the DREAM(ZS) algorithm.

        :param rwmh_scaling: Scaling factor for the RWMH algorithm.
        :param rwmh_cov: Covariance matrix for the RWMH proposal distribution.
        :return: A dictionary containing the relevant parameters.
        """
        return {
            'rwmh_scaling': 1.0,  # Generic value for DREAM(ZS)
            'rwmh_cov': np.eye(1),  # Generic value for DREAM(ZS)
            'subsampling_rate': self._check_subsampling_rate()
        }

    def setup_proposal(self, *args, **kwargs) -> object:
        """
        Set up the proposal distribution for the DREAM(ZS) algorithm.

        :param args: Positional arguments for setup.
        :param kwargs: Keyword arguments, expected to contain 'dim'.
        :return: The initialized proposal distribution object.
        :raises Warning: If required parameters are missing.
        """
        dim = kwargs.get('dim')

        if dim is None:
            warnings.warn("'dim' is missing in DREAMZAlgorithm's setup_proposal.")

        return DREAMZ(M0=10 * dim, Z_method = 'lhs', adaptive=True, period=100)  # Generic example values


class MH_cuqiAlgorithm(ProposalAlgorithm):
    """
    Implementation of the Metropolis-Hastings (MH) algorithm with CUQI-specific settings.
    """

    def get_parameters(self, proposal_sd: Union[float, Tuple[float, float]], 
                        scale: Optional[Union[float, Tuple[float, float]]] = None) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        Retrieve the parameters specific to the MH_cuqi algorithm.

        :param sd_noise: Standard deviation of noise.
        :param proposal_sd: Standard deviation of the proposal distribution.
        :param num_data: Number of data points.
        :param scale: Optional scaling factor.
        :return: A dictionary containing the relevant parameters.
        """
        pbounds = {
            'proposal_sd': proposal_sd if isinstance(proposal_sd, tuple) else None,
            'scale': proposal_sd if isinstance(scale, tuple) else None,
        }

        pbounds = {k: v for k, v in pbounds.items() if v is not None}
        return pbounds

    def setup_proposal(self, *args, **kwargs) -> object:
        """
        Set up the proposal distribution for the MH_cuqi algorithm.

        :param args: Positional arguments for setup.
        :param kwargs: Keyword arguments, expected to contain 'sd_noise', 'proposal_sd', and 'scale'.
        :return: The initialized proposal distribution object.
        :raises Warning: If required parameters are missing.
        """

        scale = kwargs.get('scale')

        return MH(kwargs.get('posterior'), x0=kwargs.get('x_init'), scale=scale) if kwargs.get('adapt') else MH(kwargs.get('posterior'), x0=kwargs.get('x_init'))


class NUTSAlgorithm(ProposalAlgorithm):
    """
    Implementation of the No-U-Turn Sampler (NUTS) algorithm.
    """

    def get_parameters(self,  proposal_sd: Union[float, Tuple[float, float]], scale: Optional[Union[float, Tuple[float, float]]] = None) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        Retrieve the parameters specific to the NUTS algorithm.

        :return: A dictionary containing the relevant parameters.
        """
        pbounds = {
            'proposal_sd': proposal_sd if isinstance(proposal_sd, tuple) else None,
            'scale':1.0,  
        }

        pbounds = {k: v for k, v in pbounds.items() if v is not None}
        return pbounds

    def setup_proposal(self, *args, **kwargs) -> object:
        """
        Set up the proposal distribution for the NUTS algorithm.

        :param args: Positional arguments for setup.
        :param kwargs: Keyword arguments for setup.
        :return: The initialized proposal distribution object.
        """
        return NUTS(kwargs.get('posterior'), x0=kwargs.get('x_init'))


class pCNAlgorithm(ProposalAlgorithm):
    """
    Implementation of the preconditioned Crank-Nicolson (pCN) algorithm.
    """

    def get_parameters(self, proposal_sd: Union[float, Tuple[float, float]],scale: Optional[Union[float, Tuple[float, float]]] = None) -> Dict[str, Union[float, Tuple[float, float]]]:

        """
        Retrieve the parameters specific to the pCN algorithm.

        :return: A dictionary containing the relevant parameters.
        """
        pbounds = {
            'proposal_sd': proposal_sd if isinstance(proposal_sd, tuple) else None,
            'scale': 1.0, # 
        }

        pbounds = {k: v for k, v in pbounds.items() if v is not None}
        return pbounds

    def setup_proposal(self, *args, **kwargs) -> object:
        """
        Set up the proposal distribution for the pCN algorithm.

        :param args: Positional arguments for setup.
        :param kwargs: Keyword arguments for setup.
        :return: The initialized proposal distribution object.
        """
        return pCN(kwargs.get('posterior'), x0=kwargs.get('x_init'))


class ProposalAlgorithmFactory:
    """
    Factory class to create instances of proposal algorithms based on the specified algorithm name.
    """

    @staticmethod
    def create_algorithm(algorithm: str, levels: int=1, subsampling_rate: Optional[Union[int, Tuple[int, int]]] = None) -> ProposalAlgorithm:
        """
        Create an instance of the appropriate ProposalAlgorithm subclass.

        :param algorithm: The name of the algorithm to create.
        :param levels: Number of levels in the model.
        :param subsampling_rate: Optional subsampling rate, required if levels > 1.
        :return: An instance of a subclass of ProposalAlgorithm.
        :raises ValueError: If an unrecognized algorithm name is provided.
        """
        if algorithm == 'MH_tiny':
            return MH_tinyAlgorithm(levels, subsampling_rate)
        elif algorithm == 'MH_cuqi':
            return MH_cuqiAlgorithm(levels, subsampling_rate)
        elif algorithm == 'AM':
            return AMAlgorithm(levels, subsampling_rate)
        elif algorithm == 'CN':
            return CNAlgorithm(levels, subsampling_rate)
        elif algorithm == 'DREAMZ':
            return DREAMZAlgorithm(levels, subsampling_rate)
        elif algorithm == 'NUTS':
            return NUTSAlgorithm(levels, subsampling_rate)
        elif algorithm == 'pCN':
            return pCNAlgorithm(levels, subsampling_rate)
        else:
            raise ValueError(f"Unrecognized algorithm '{algorithm}' specified.")





class MCMC:
    """
    A class to perform Markov Chain Monte Carlo (MCMC) sampling with various algorithms.

    This class handles the setup and execution of MCMC simulations, including the creation of 
    output directories, running diagnostics, and saving summary statistics.
    The library used is TinyDA
    """

    def __init__(self, 
                 my_posterior: List[Any], 
                 algorithm: str = "MH_tiny", 
                 num_params: int = 1, 
                 num_observations: int = 0, 
                 levels: int = 1, 
                 subsampling_rate: Union[int, List[int]] = 1):
        """
        Initialize the MCMC class with the specified parameters and algorithm.

        :param my_posterior: A list of posterior distributions to sample from.
        :param algorithm: The MCMC algorithm to use (default is 'MH_tiny').
        :param num_params: The number of parameters to estimate.
        :param num_observations: The number of observations (data points).
        :param levels: The number of levels in the model.
        :param subsampling_rate: The subsampling rate for posterior samples.
        """
        self.my_posterior = my_posterior
        self.algorithm = algorithm
        self.num_params = num_params
        self.num_observations = num_observations
        self.levels = levels
        self.subsampling_rate = subsampling_rate
        self.algorithm_instance = ProposalAlgorithmFactory.create_algorithm(algorithm, levels, subsampling_rate)

    def __call__(self, 
                 iterations: int, 
                 burn_in: int, 
                 number_chains: int = 1, 
                 diagnostic: bool = True, 
                 rwmh_covariance: float = None, 
                 rmwh_scaling: float = 0.1, 
                 period: int = 100, 
                 t0: int = 0, 
                 rwmh_adaptive: bool = False, 
                 force_sequential: bool = False, 
                 **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute the MCMC sampling process.

        :param iterations: The number of iterations for the MCMC run.
        :param burn_in: The number of burn-in iterations to discard.
        :param number_chains: The number of MCMC chains to run.
        :param diagnostic: Whether to perform diagnostics on the MCMC run.
        :param rwmh_covariance: The covariance matrix for the Random Walk Metropolis-Hastings algorithm.
        :param rmwh_scaling: The scaling factor for the RWMH algorithm.
        :param period: The period for adaptive algorithms like Adaptive Metropolis.
        :param t0: The initial time for adaptive algorithms.
        :param rwmh_adaptive: Whether the algorithm is adaptive.
        :param force_sequential: Whether to enforce sequential processing.
        :param kwargs: Additional keyword arguments for the proposal setup.
        :return: A tuple containing the estimated parameters and a dictionary of summary statistics.
        """
        # Create a unique folder for saving outputs
        folder_name = self._create_output_folder(number_chains, rwmh_covariance, rmwh_scaling, self.algorithm, rwmh_adaptive)

        # Obtain the Maximum A Posteriori (MAP) estimate for the initial parameters
        MAP = get_MAP(self.my_posterior[-1])

        # Setup the proposal distribution using the algorithm instance
        my_proposal = self.algorithm_instance.setup_proposal(
            rwmh_scaling=rmwh_scaling,
            rwmh_cov=rwmh_covariance,
            adaptive=rwmh_adaptive,
            period=period,
            t0=t0,
            dim=self.num_observations,
            num_params=self.num_params
        )

        # Ensure subsampling_rate is a list of integers if needed
        subsampling_rate = self._prepare_subsampling_rate()

        # Perform MCMC sampling
        my_chains = sample(
            self.my_posterior,
            my_proposal,
            iterations=iterations,
            n_chains=number_chains,
            subsampling_rate=subsampling_rate,
            force_sequential=force_sequential,
            initial_parameters=MAP
        )

        # Convert chains into inference data and perform diagnostics
        idata = to_inference_data(my_chains, level=(len(self.my_posterior) - 1), burnin=burn_in) if len(self.my_posterior) > 2 else to_inference_data(my_chains, burnin=burn_in)
        summary = self._perform_diagnostics(idata, folder_name, diagnostic)

        # Extract and return the estimated values and summary statistics
        estimates = np.array(summary['mean'])
        return estimates, {
            'expected_param': summary['mean'],
            'std_dev': summary['sd'],
            'ess_bulk': summary['ess_bulk'],
            'ess_tail': summary['ess_tail'],
            'r_hat': summary['r_hat']
        }

    def _prepare_subsampling_rate(self) -> Union[int, List[int]]:
        """
        Ensure subsampling_rate is a list of integers, adjusted for the posterior's length.

        :return: The prepared subsampling rate.
        """
        if isinstance(self.subsampling_rate, int):
            subsampling_rate = [self.subsampling_rate] * (len(self.my_posterior) - 1)
            if len(self.my_posterior) == 2:
                return subsampling_rate[0]
            return subsampling_rate
        return self.subsampling_rate

    def _create_output_folder(self, 
                              n: int, 
                              rwmh_cov: np.ndarray, 
                              rmwh_scaling: float, 
                              algo: str, 
                              rwmh_adaptive: bool) -> str:
        """
        Create a unique folder name for saving outputs based on key MCMC parameters and prepare the folder.

        :param n: Number of MCMC samples.
        :param rwmh_cov: Covariance matrix for the Random Walk Metropolis-Hastings algorithm.
        :param rmwh_scaling: Scaling factor for the Metropolis-Hastings algorithm.
        :param algo: The algorithm used ('MH', 'AM', 'CN', 'DREAMZ').
        :param rwmh_adaptive: Indicates whether the algorithm is adaptive.
        :return: The path to the created folder.
        """
        if np.isscalar(rwmh_cov):
            rwmh_cov_save = np.array([[rwmh_cov]])
        else:
            rwmh_cov_save = rwmh_cov

        # Convert key parameters to strings that are safe to use in a file path
        rwmh_cov_str = "None" if rwmh_cov is None else np.array_str(rwmh_cov_save, precision=2).replace("\n", "")
        rwmh_adaptive_str = "adaptive" if rwmh_adaptive else "non_adaptive"

        # Create a folder name based on key parameters
        folder_name = (
            f"MCMC_output_n{n}_cov{rwmh_cov_str}_scaling{rmwh_scaling}_algo{algo}_{rwmh_adaptive_str}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        folder_path = create_folder_name(folder_name)
        folder_path=Clean.create_output_directory("output", folder_path)
        return folder_path

    def _perform_diagnostics(self, idata: az.InferenceData, 
                             folder_name: str, 
                             diagnostic: bool) -> Dict[str, Any]:
        """
        Perform and save MCMC diagnostic plots and summary statistics.

        :param idata: Inference data object containing MCMC samples.
        :param folder_name: The path to the folder where the diagnostics will be saved.
        :param diagnostic: Whether to perform diagnostics or just summarize.
        :return: Summary statistics of the MCMC samples.
        """
        summary = az.summary(idata)
        if diagnostic:
            # Plot and save diagnostic plots
            plot_paths = {
                "trace": "trace_plot.png",
                "autocorrelation": "autocorrelation_plot.png",
                "ess": "ess_plot.png",
                "ess_local": "ess_local_plot.png",
                "rank": "rank_plot.png",
                "ess_per_second": "ess_per_second_plot.png"  # ESS per second plot
            }
            
            az.plot_trace(idata)
            plt.savefig(os.path.join(folder_name, plot_paths["trace"]))
            plt.close()

            az.plot_autocorr(idata)
            plt.savefig(os.path.join(folder_name, plot_paths["autocorrelation"]))
            plt.close()

            az.plot_ess(idata)
            plt.savefig(os.path.join(folder_name, plot_paths["ess"]))
            plt.close()

            az.plot_ess(idata, kind='local')
            plt.savefig(os.path.join(folder_name, plot_paths["ess_local"]))
            plt.close()

            az.plot_rank(idata)
            plt.savefig(os.path.join(folder_name, plot_paths["rank"]))
            plt.close()

            # Plot and save ESS per second
            az.plot_ess(idata, kind="quantile", relative=True)
            plt.savefig(os.path.join(folder_name, plot_paths["ess_per_second"]))
            plt.close()

            # Save summary statistics
            self._save_summary_to_folder(summary, folder_name)

        return summary

    def _save_summary_to_folder(self, summary: Dict[str, Any], folder_name: str):
        """
        Save the summary statistics to a text file in the output folder.

        :param summary: The summary statistics to save.
        :param folder_name: The path to the folder where the summary will be saved.
        """

        with open(os.path.join(folder_name, "summary_statistics.txt"), "w") as f:
            f.write("MCMC Summary Statistics:\n")
            f.write(f"Estimated Parameters (mean):\n{summary['mean']}\n")
            f.write(f"Standard Deviation (sd):\n{summary['sd']}\n")
            f.write(f"Effective Sample Size (ESS):\n{summary['ess_bulk']}\n")
            f.write(f"ESS Tail:\n{summary['ess_tail']}\n")
            f.write(f"R-hat:\n{summary['r_hat']}\n")





class MCMC_cuqi:
    def __init__(
        self, 
        y: Any, 
        x: Any, 
        observation: np.ndarray, 
        algo: str = "MH_cuqi", 
        m: int = 1, 
        scale: float = 0.3, 
        adapt: bool = False, 
        parallel: bool = False,
    ):
        self.y = y
        self.x = x
        self.observation = observation.flatten()
        self.algo = algo
        self.m = m
        self.scale = scale
        self.adapt = adapt
        self.parallel = parallel


        # Initialize the appropriate algorithm instance
        self.algorithm_instance = ProposalAlgorithmFactory.create_algorithm(
            algo
        )

        adapt_str = "adaptive" if adapt else "non_adaptive"
        base_name = f"MCMC_cuqi_algo{algo}_{adapt_str}_scale{scale}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.folder_name =   Clean.create_output_directory("output",base_name)

    def __call__(
        self, 
        initial_point:np.ndarray,
        N: int, 
        burn_in: int, 
        n: int = 1, 
        diagnostic: bool = True,
        plot_result:bool=True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Initialize arrays for storing results
        estimates = np.empty((self.m, 0))
        chains = np.empty((0, self.m, N - burn_in))
        post = np.empty((self.m, 0))

        # Create the posterior distribution using CUQI
        posterior = JointDistribution(self.x, self.y)(y=self.observation)
        x_init=initial_point
        # Perform MCMC sampling
        if self.parallel:
            ray.init(ignore_reinit_error=True, logging_level=30)
            futures = [self.chain_creation_parallel.remote(self,N, burn_in, posterior, x_init) for _ in range(n)]
            results = ray.get(futures)
            ray.shutdown()
        else:
            results = [self.chain_creation(N, burn_in,  posterior, x_init) for _ in range(n)]

        for result in results:
                estimates = np.column_stack((estimates, result[0])) 
                chains = np.concatenate((chains, result[1]), axis=0)
                post = np.concatenate((post, result[2]), axis=1)

        # Generate and save trace plots for each parameter
        self.create_trace_plot(chains)

        # Perform diagnostic analysis if requested
        if diagnostic:
            self.perform_diagnostics(chains, post, estimates, plot_result)

        # Calculate and return statistics
        mean, std_dev = self.calculate_statistics(estimates)
        return estimates, {'expected_param': mean, 'std_dev': std_dev, 'ess': None, 'r_hat': None}  # ESS and r_hat will be updated in diagnostics

    @ray.remote
    def chain_creation_parallel(
        self,
        N: int, 
        burn_in: int,  
        posterior: Any, 
        x_init: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.chain_creation(N, burn_in,  posterior, x_init)

    def chain_creation(
        self,
        N: int, 
        burn_in: int, 
        posterior: Any, 
        x_init: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Setup the proposal distribution using the algorithm instance
        if not hasattr(self, 'proposal'):
            self.proposal = self.algorithm_instance.setup_proposal(posterior=posterior, x_init=x_init, scale=self.scale, adapt=self.adapt)    

        # Sample from the posterior distribution
        samples =  self.proposal.sample_adapt(N - burn_in, burn_in) if self.adapt else  self.proposal.sample(N - burn_in, burn_in)

        estimates = samples.mean()[:, np.newaxis]
        chains = np.expand_dims(samples.samples, axis=0)
        post = samples.samples

        print(f"Mean values: {estimates.mean(axis=1)}")

        return estimates, chains, post

    @staticmethod
    def preprocess_input(x: np.ndarray, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if mean.ndim == 1 and mean.size != x.size:
            mean_reshaped = mean.reshape(1, -1)
        else:
            mean_reshaped = mean
        return x, mean_reshaped

   

    def setup_proposal(self, *args, **kwargs) -> object:
        """
        Set up the proposal distribution for the MH_cuqi algorithm.

        :param args: Positional arguments for setup.
        :param kwargs: Keyword arguments, expected to contain 'sd_noise', 'proposal_sd', and 'scale'.
        :return: The initialized proposal distribution object.
        :raises Warning: If required parameters are missing.
        """

        scale = kwargs.get('scale')

        return MH(kwargs.get('posterior'), x0=kwargs.get('x_init'), scale=scale) if kwargs.get('adapt') else MH(kwargs.get('posterior'), x0=kwargs.get('x_init'))




    def create_trace_plot(self, chains: np.ndarray) -> None:
        """
        Create and save trace plots for MCMC chains.

        Args:
            chains (np.ndarray): A 3D array containing MCMC chains with dimensions (n_chains, n_params, n_samples).

        Returns:
            None
        """
        for l in range(chains.shape[1]):
            plt.figure(figsize=(10, 4))
            for i in range(chains.shape[0]):
                plt.plot(chains[i, l, :])
            plt.xlabel('Sample')
            plt.ylabel('Value')
            plt.title(f'Trace Plot for variable {l}')
            plt.legend([f'Chain {i+1}' for i in range(chains.shape[0])])
            plt.savefig(os.path.join(self.folder_name, f"trace_plot_var_{l}.png"))
            plt.close()

    @staticmethod
    @njit
    def calculate_statistics(estimates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the mean and standard deviation of the estimated parameters.

        Args:
            estimates (np.ndarray): A 2D array of estimated parameters with dimensions (n_params, n_samples).

        Returns:
            tuple: A tuple containing the mean and standard deviation of the estimates.
        """
        mean = estimates.sum(axis=1) / estimates.shape[1]
        std_dev = np.zeros(estimates.shape[0])
        
        for i in range(estimates.shape[0]):
            variance = 0.0
            for j in range(estimates.shape[1]):
                variance += (estimates[i, j] - mean[i]) ** 2
            variance /= estimates.shape[1]
            std_dev[i] = variance ** 0.5

        return mean, std_dev

    def save_diagnostics(self, estimates: np.ndarray, ess: float, r_hat: float) -> None:
        """
        Save diagnostic statistics to a text file.

        Args:
            estimates (np.ndarray): A 2D array of estimated parameters with dimensions (n_params, n_samples).
            ess (float): The effective sample size.
            r_hat (float): The R-hat statistic for convergence diagnostics.

        Returns:
            None
        """

        with open(os.path.join(self.folder_name, "diagnostic_stats.txt"), "w") as f:
            f.write("MCMC Diagnostics:\n")
            f.write(f"Estimated Parameters (mean): {np.mean(estimates, axis=1)}\n")
            f.write(f"Standard Deviation: {np.std(estimates, axis=1)}\n")
            f.write(f"Effective Sample Size (ESS): {ess}\n")
            if r_hat is not None:
                f.write(f"R-hat: {r_hat}\n")
            f.write("\n")

    def perform_diagnostics(self, chains: np.ndarray, post: np.ndarray, estimates: np.ndarray,plot_result:bool=True) -> None:
        # Calculate diagnostic statistics (ESS, autocovariance, R-hat)
        autocov = az.autocov(chains[:, 0, :])
        ess = az.ess(chains[:, 0, :])
        r_hat = az.rhat(chains[:, 0, :]) if chains.shape[0] > 1 else None  # Only compute R-hat if more than one chain
        print(f"ESS values = {ess}")

        # Save diagnostics
        self.save_diagnostics(estimates, ess, r_hat)

        # Generate and save distribution plots
        if plot_result:
            for num in range(post.shape[0]):
                plt.figure()
                num_bins = 20
                bin_edges = np.linspace(np.min(post[num, :]), np.max(post[num, :]), num_bins + 1)
                hist, _ = np.histogram(post[num, :], bins=bin_edges)
                hist = hist / post.shape[1]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', label=f'Var {num + 1}')
                plt.xlabel('Value')
                plt.ylabel('Probability')
                plt.title('Distribution')
                plt.legend()
                plt.savefig(os.path.join(self.folder_name, f"distribution_var_{num+1}.png"))
                plt.close()

            # Generate and save autocovariance plot for the first chain
            plt.figure()
            plt.plot(autocov[0, :])
            plt.title('Autocovariance first chain')
            plt.xlabel('Lag')
            plt.ylabel('Autocovariance')
            plt.savefig(os.path.join(self.folder_name, "autocovariance_first_chain.png"))
            plt.close()




