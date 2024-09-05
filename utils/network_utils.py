import gc
import logging
import os
import re
import platform
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pprint import pprint

import numpy as np
from numpy import newaxis as _  # easier reading

from keras.models import load_model
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf

from cuqi.distribution import Gaussian
import optuna
import tinyDA as tda

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from utils.bayesian_utils import *
from utils.helper_functions import *
from utils.functions_to_ray import *


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





# "SUPPORT" CLASSES
# management and safety

# Define the types of networks as an enumeration for type safety and clarity.
class NetworkType(Enum):
    LF = 'LF'            # Low-Fidelity Network
    MF = 'MF'            # Mid-Fidelity Network
    HF = 'HF'            # High-Fidelity Network
    HFLIN = 'HFLIN'      # High-Fidelity Linear Network
    SINGLE = 'SINGLE'    # Single-Fidelity Network
    HFPER = 'HFPER'      # High-Fidelity Periodic Network
    STEP = 'STEP'        # Step-Based Network
    INTER = 'INTER'      # Intermediate Network
    LSTM = 'LSTM'        # LSTM Network
    LSTM_SUPPORT = 'LSTM_SUPPORT'  # LSTM Support Network
    LSTM_SUPPORT2 = 'LSTM_SUPPORT2'  # LSTM Support Network (alternative version)

# Define a configuration data class for networks to encapsulate all configuration details.
@dataclass(slots=True)
class NetworkConfig:
    """
    NetworkConfig holds the configuration details required for creating a network.
    
    Attributes:
        network_type: Type of the network to be built (e.g., LF, MF, LSTM).
        names: Optional list of names used for MultiFidelity networks.
        network_parameters: Optional dictionary of parameters for the network.
        dataset_train: Training data for the network.
        output_train: Training output data.
        dataset_validation: Optional validation data.
        output_validation: Optional validation output data.
        epochs_number: Number of epochs for training.
        batch_size: Batch size for training.
        train: Flag indicating whether to train the network.
        do_HPO: Flag indicating whether to perform hyperparameter optimization.
        verbose: Flag indicating verbosity of the output.
        device: Device to be used for training (e.g., '/CPU:0', '/GPU:0').
        num_trials: int number of iterations of HPO 
    """
    network_type: str
    names: Optional[List[str]] = None
    network_parameters: Optional[dict] = None
    dataset_train: Optional[Union[np.ndarray, List[np.ndarray]]] = None
    output_train: Optional[Union[np.ndarray, List[np.ndarray]]] = None
    dataset_validation: Optional[Union[np.ndarray, List[np.ndarray]]] = None
    output_validation: Optional[Union[np.ndarray, List[np.ndarray]]] = None
    epochs_number: Union[int, List[int]] = 1000
    batch_size: Union[int, List[int]] = 10
    train: bool = True
    do_HPO: bool = False
    verbose: bool = False
    device: str = '/CPU:0'
    num_trials: int=15


# Factory class to build different types of networks based on the provided type.
class NetworkFactory:
    """
    NetworkFactory is responsible for creating instances of various network types 
    based on the provided configuration. It uses the network type to determine which 
    specific network class to instantiate.
    """

    @staticmethod
    def build_network(config: NetworkConfig, getModel: Callable) -> 'INetwork':
        """
        Build and return a network instance based on the provided configuration.

        Args:
            config (NetworkConfig): Configuration object with network details.
            detModel (Callable): An object that is expected to be a class instance with a get_model method.

        Returns:
            INetwork: An instance of a network class.
        """
        
        
        network_type_enum = NetworkFactory._get_network_type(config.network_type)
        
        if network_type_enum in {
            NetworkType.LF, NetworkType.MF, NetworkType.HF, 
            NetworkType.HFLIN, NetworkType.SINGLE, NetworkType.HFPER
        }:
            return NetworkFactory._build_neural_network(config, getModel)
        
        elif network_type_enum == NetworkType.STEP:
            return NetworkFactory._build_multifidelity_network(config, getModel)
        
        elif network_type_enum == NetworkType.INTER:
            return NetworkFactory._build_intermediate_network(config, getModel)
        
        elif network_type_enum in {
            NetworkType.LSTM, NetworkType.LSTM_SUPPORT, NetworkType.LSTM_SUPPORT2
        }:
            return NetworkFactory._build_lstm_network(config, getModel)
        
        else:
            raise ValueError(f"Invalid network type: {config.network_type}")

    @staticmethod
    def _get_network_type(network_type: str) -> NetworkType:
        """
        Convert the network type string to a NetworkType enum for safer comparison.

        Args:
            network_type (str): The type of network as a string.

        Returns:
            NetworkType: The corresponding NetworkType enum.

        Raises:
            ValueError: If the network type is invalid or not recognized.
        """
        try:
            return NetworkType[network_type.upper()]
        except KeyError:
            match = re.match(r'(\d+)STEP', network_type, re.IGNORECASE)
            if match:
                return NetworkType.STEP
            raise ValueError(f"Invalid network type: {network_type}")

    # Splitting for maintainability and possible extension
    @staticmethod
    def _build_neural_network(config: NetworkConfig, getModel:Callable) -> 'Neural_Network':
        """
        Build and return a Neural_Network instance.

        Args:
            config (NetworkConfig): Configuration object with network details.
            getModel (Callable): function to generate NN architecture
        Returns:
            Neural_Network: An instance of Neural_Network.
        """
        return Neural_Network(
            name=config.network_type, 
            params=config.network_parameters, 
            data_train=config.dataset_train, 
            output_train=config.output_train,
            data_val=config.dataset_validation, 
            output_val=config.output_validation, 
            N=config.epochs_number, 
            n=config.batch_size, 
            train=config.train, 
            do_HPO=config.do_HPO, 
            verbose=config.verbose, 
            device=config.device, 
            num_trials=config.num_trials,
            getModel=getModel
        )

    @staticmethod
    def _build_multifidelity_network(config: NetworkConfig, getModel:Callable) -> 'MultiFidelity':
        """
        Build and return a MultiFidelity network instance.

        Args:
            config (NetworkConfig): Configuration object with network details.
            getModel (Callable): function to generate NN architecture

        Returns:
            MultiFidelity: An instance of MultiFidelity.

        Raises:
            ValueError: If the training data and output data are not provided as lists.
        """
        if not (isinstance(config.dataset_train, list) and isinstance(config.output_train, list)):
            raise ValueError("For MultiFidelity network, data_train and output_train must be lists of numpy arrays.")
        
        return MultiFidelity(
            names=config.names, 
            params=config.network_parameters, 
            data_train=config.dataset_train, 
            output_train=config.output_train,
            data_val=config.dataset_validation, 
            output_val=config.output_validation, 
            N=config.epochs_number, 
            n=config.batch_size, 
            train=config.train,
            do_HPO=config.do_HPO, 
            verbose=config.verbose, 
            device=config.device, 
            num_trials=config.num_trials,
            getModel=getModel,

        )

    @staticmethod
    def _build_lstm_network(config: NetworkConfig, getModel:Callable) -> 'LSTM_network':
        """
        Build and return an LSTM_network instance.

        Args:
            config (NetworkConfig): Configuration object with network details.
            getModel (Callable): function to generate NN architecture

        Returns:
            LSTM_network: An instance of LSTM_network.
        """
        return LSTM_network(
            name=config.network_type, 
            params=config.network_parameters, 
            data_train=config.dataset_train, 
            output_train=config.output_train,
            data_val=config.dataset_validation, 
            output_val=config.output_validation, 
            N=config.epochs_number, 
            train=config.train, 
            do_HPO=config.do_HPO, 
            verbose=config.verbose, 
            device=config.device,
            num_trials=config.num_trials,
            getModel=getModel
        )

    @staticmethod
    def _build_intermediate_network(config: NetworkConfig, getModel:Callable) -> 'Intermediate':
        """
        Build and return an Intermediate network instance.

        Args:
            config (NetworkConfig): Configuration object with network details.
            getModel (Callable): function to generate NN architecture

        Returns:
            Intermediate: An instance of Intermediate.
        """
        return Intermediate(
            name=config.network_type, 
            params=config.network_parameters, 
            data_train=config.dataset_train, 
            output_train=config.output_train,
            data_val=config.dataset_validation, 
            output_val=config.output_validation, 
            N=config.epochs_number, 
            n=config.batch_size, 
            train=config.train, 
            do_HPO=config.do_HPO, 
            device=config.device,
            num_trials=config.num_trials
        )
    



# NETWORK CREATION CLASSES 


@dataclass(slots=True)
class INetwork(ABC):
    """
    Abstract base class for network-related operations.
    
    This class defines common interfaces and properties for various network types, 
    ensuring a consistent API across different network implementations. It includes 
    methods for training, hyperparameter optimization, predictions, and parameter inversion.
    
    Attributes:
        inputs: Stores the input variables.
        transformations: List of transformations to be applied to the input data.
        _data_train: Training data for the model.
        _output_train: Output data corresponding to the training data.
        _n: Batch size for training.
        _N: Number of epochs for training.
        _params: Dictionary of hyperparameters for the model.
    """
    
    inputs: Optional[Any] = None
    transformations: List[Any] = field(default_factory=list)
    _data_train: Optional[np.ndarray] = None
    _output_train: Optional[np.ndarray] = None
    _n: Optional[int] = None
    _N: Optional[int] = None
    _params: Optional[dict] = None


    # Property and Setter Methods
    @property
    def data_train(self) -> Optional[np.ndarray]:
        """Get or set the training data for the model."""
        return self._data_train

    @data_train.setter
    def data_train(self, data_train: np.ndarray) -> None:
        self._data_train = data_train

    @property
    def output_train(self) -> Optional[np.ndarray]:
        """Get or set the training output data for the model."""
        return self._output_train

    @output_train.setter
    def output_train(self, output_train: np.ndarray) -> None:
        self._output_train = output_train

    @property
    def params(self) -> Optional[dict]:
        """Get or set the model's hyperparameters."""
        return self._params

    @params.setter
    def params(self, value: dict) -> None:
        self._params = value
        self.print_params()

    @property
    def N(self) -> Optional[int]:
        """Get or set the number of epochs for training."""
        return self._N

    @N.setter
    def N(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError('Number of epochs must be a positive integer!')
        self._N = value

    @property
    def n(self) -> Optional[int]:
        """Get or set the batch size for training."""
        return self._n

    @n.setter
    def n(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError('Batch size must be a positive integer!')
        self._n = value

    # General Utility Methods
    def variable_input(self, input_discr: Any) -> None:
        """
        Set the input variable when solving the inverse problem.

        Args:
            input_discr (Any): The input discriminator.
        """
        self.inputs = input_discr

    def is_instance_MF(self) -> bool:
        """
        Check if the current instance is a MultiFidelity network.

        Returns:
            bool: True if the instance is of type MultiFidelity, False otherwise.
        """
        return isinstance(self, MultiFidelity)

    def print_params(self) -> None:
        """Prints the model's hyperparameters in a readable format."""
        print("Model Hyperparameters:")
        for key, val in self._params.items():
            print(f"{key}: {val}")

    def _select_device(self):
        """
        This method selects a device based on system properties.
        """
        if self._gpu_available():
            return "gpu"  # or any identifier for GPU you prefer
        else:
            return "cpu"  # or any identifier for CPU you prefer

    def _gpu_available(self):
        """
        Checks for GPU availability.
        """
        # Check for NVIDIA GPU by trying to run 'nvidia-smi'
        try:
            if platform.system() == "Windows":
                # Use 'where' command on Windows
                result = subprocess.run(["where", "nvidia-smi"], capture_output=True, text=True)
            else:
                # Use 'which' command on Unix-based systems
                result = subprocess.run(["which", "nvidia-smi"], capture_output=True, text=True)
                
            # If 'nvidia-smi' is found, it's likely that an NVIDIA GPU is available
            if result.returncode == 0:
                return True
            else:
                return False
        except Exception as e:
            print(f"Error checking GPU availability: {e}")
            return False
    
    # Prediction and Evaluation Methods
    def _input_wrapper_prediction(self, x_test: np.ndarray) -> np.ndarray:
        """
        Prepare input data for prediction by applying transformations and handling inputs.

        Args:
            x_test (np.ndarray): Test data for prediction.
            multi_input (bool): Flag indicating if the model expects multiple inputs.

        Returns:
            np.ndarray: Prepared input data for the model.
        """
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)

        x_test = x_test.T

        if self.transformations:
            x_final = reduce(lambda acc, transf: np.hstack([acc, transf(acc)]), self.transformations, x_test)
        else:
            x_final = x_test

        if self.inputs is not None:
            x_final = np.tile(x_final, (self.inputs.shape[0], 1))
        else:
            warnings.warn("Inputs are not set.", UserWarning)
            return np.array([])

        return x_final
    
    @decorators.prediction_cache_decorator()
    def _wrapper_prediction(self, x_test: np.ndarray) -> np.ndarray:
        """
        Wrapper for making predictions with processed test inputs.

        Args:
            x_test (np.ndarray): Test input data.
            multi_input (bool): Flag to indicate if multiple inputs are used.

        Returns:
            np.ndarray: Predicted output data.
        """
        x_final = self._input_wrapper_prediction(x_test)
        concatenated_input = Inversion_helper.concatenate_inputs(self.inputs, x_final)
        return self.prediction(concatenated_input).flatten()

    def performance(self, data_test: np.ndarray, output_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the performance of the model on test data.

        Args:
            data_test (np.ndarray): Test input data.
            output_test (np.ndarray): Expected output data.

        Returns:
            Tuple[float, float]: Test Mean Squared Error (MSE) and R^2 score.
        """
        pred = self.prediction(data_test)

        if len(output_test.shape) < len(pred.shape):
            output_test = output_test[:, np.newaxis]

        test_mse, r2 = Inversion_helper.calculate_metrics(output_test, pred)
        print(f"Test MSE: {test_mse}")
        print(f"R^2: {r2}")

        return test_mse, r2

    def summary(self) -> None:
        """Print the summary of the model architecture and free up memory."""
        self.model.summary()
        gc.collect()

    def save(self, file_path: str) -> None:
        """
        Save the trained model to a file.

        Args:
            file_path (str): The path where the model will be saved.
        """
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path: str) -> None:
        """
        Load a trained model from a file.

        Args:
            file_path (str): The path from where the model will be loaded.
        """
        self.model = load_model(
            file_path, 
            custom_objects={
                'sinusoidal_activation': sinusoidal_activation,
                'FourierLayer': FourierLayer, 
                'custom_activation': custom_activation
            }
        )
        self.input_shape = self.model.inputs[0][-1]
        self._output_shape = self.model.outputs[0][-1]

        print(f"Model loaded from {file_path}")

    # Abstract Methods
    @abstractmethod
    def prediction(self, *args, **kwargs) -> np.ndarray:
        """Abstract method for making predictions using the network."""
        pass

    @abstractmethod
    def HPO(self) -> None:
        """Abstract method for hyperparameter optimization."""
        pass

    @abstractmethod
    def training(self) -> None:
        """Abstract method for training the network."""
        pass

    # Advanced Inversion Methods
    @decorators.compute_time
    def param_inverse(self, 
                      mean_prior: np.ndarray, 
                      x_data: np.ndarray, 
                      max_parameter: float = None, 
                      cov_prior: Optional[np.ndarray] = None,
                      cov_noise: float = 0.1, 
                      cov_likelihood: Optional[np.ndarray] = None, 
                      observation: Optional[np.ndarray] = None,
                      x_real: Optional[np.ndarray] = None, 
                      number_chains: int = 1, 
                      iterations: int = 1000, 
                      burn_in: int = 500, 
                      levels: int = 1,
                      diagnostic: bool = True, 
                      plot_result: bool = True,
                      rwmh_covariance: Optional[float] = None, 
                      rmwh_scaling: float = 0.1, 
                      rwmh_adaptive: bool = True,
                      subsampling_rate: Union[int, List[int]] = 1, 
                      proposal_algorithm: str = "MH", 
                      force_sequential: bool = False, 
                      transformation: List[Any] = []) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Perform parameter inversion using MCMC sampling.

        Args:
            mean_prior (np.ndarray): Mean of the prior distribution.
            x_data (np.ndarray): Input data for the model.
            max_parameter (float): Maximum parameter value for estimation, used for plotting.
            cov_prior (Optional[np.ndarray]): Covariance of the prior distribution.
            cov_noise (float): Covariance of the noise in the observations.
            cov_likelihood (Optional[np.ndarray]): Covariance matrix for the likelihood function.
            observation (Optional[np.ndarray]): Observed data to match against the model.
            x_real (Optional[np.ndarray]): True parameter values for error computation.
            number_chains (int): Number of MCMC chains to run.
            iterations (int): Total number of MCMC iterations.
            burn_in (int): Number of burn-in iterations.
            levels (int): Number of model levels (for multi-level modeling).
            diagnostic (bool): Flag indicating whether to generate diagnostic plots.
            plot_result (bool): Flag to plot results after estimation.
            rwmh_covariance (Optional[float]): Covariance for the RWMH proposal distribution.
            rmwh_scaling (float): Scaling factor for the RWMH proposal.
            rwmh_adaptive (bool): Flag for enabling adaptive RWMH proposals.
            subsampling_rate (Union[int, List[int]]): Rate or rates of subsampling the posterior.
            proposal_algorithm (str): Algorithm to use for MCMC sampling ('MH', 'AM', 'CN', 'DREAMZ').
            force_sequential (bool): Flag to enforce sequential processing of the MCMC algorithm.
            transformation (List[Any]): List of transformations to apply to the input data.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[dict]]: Estimated parameters, relative error, and detailed sampling results.
        """

        # Check for the dimensionality of the observation data
        dim = x_real.shape[0] if x_real is not None else (observation.shape[0] if observation is not None else None)
        if dim is None:
            warnings.warn("No observation nor data given", UserWarning)
            return np.array([]), np.array([]), np.array([])

        if iterations <= burn_in:
            warnings.warn("Number of steps insufficient, smaller or equal to burn-in", UserWarning)
            return np.array([]), np.array([]), np.array([])

        self.transformations = transformation
        self.inputs = x_data

        # Set up the prior distribution
        my_prior = Inversion_helper.setup_prior(mean_prior, cov_prior)

        # If no observation is provided, simulate it
        if observation is None and x_real is not None:
            observation = Inversion_helper.simulate_observations(x_real, cov_noise, self._wrapper_prediction)
        else:
            observation = [Inversion_helper.perturbation(y, cov_noise).flatten() for y in observation] if isinstance(observation, list) else Inversion_helper.perturbation(observation, cov_noise).flatten()

        # Handle multi-level models
        if levels > 1:
            if not hasattr(self, 'model_list'):
                warnings.warn("Single-level case considered", UserWarning)
            elif levels > len(self.model_list):
                raise ValueError("Number of levels exceeds available models")

            for l in range(levels):
                self.model_list[l]._set_level(x_data=x_data, prev_steps=self.model_list[0:l], level=l+1)

            my_loglike = [Inversion_helper.setup_likelihood(observation[l], cov_likelihood, cov_noise, dim) for l in range(levels)]
            my_posterior = [tda.Posterior(my_prior, my_loglike[i], self.model_list[i]._wrapper_prediction) for i in range(levels)]
        else:
            my_loglike = Inversion_helper.setup_likelihood(observation, cov_likelihood, cov_noise, dim)
            my_posterior = [tda.Posterior(my_prior, my_loglike, self._wrapper_prediction)]

        rwmh_covariance = rwmh_covariance or 1.0

        # Run MCMC
        MC_MC = MCMC(my_posterior, proposal_algorithm, x_real.shape[0], dim, levels, subsampling_rate)
        estimates, chains_params = MC_MC(iterations=iterations, burn_in=burn_in, number_chains=number_chains, diagnostic=diagnostic, rwmh_covariance=rwmh_covariance, rmwh_scaling=rmwh_scaling, rwmh_adaptive=rwmh_adaptive, force_sequential=force_sequential)

        # Plot diagnostics if requested
        if plot_result and x_real is not None:
            self._plot_diagnostics(estimates, x_real, max_parameter)
        if x_real is not None:
            error = np.sum(Inversion_helper.relative_error(estimates, x_real))
        else:
            error = None
        return estimates, error, chains_params
    

    @decorators.compute_time
    def inverse_cuqi(self, 
                     mean_prior: np.ndarray, 
                     x_data: np.ndarray, 
                     max_parameter: float, 
                     x_real: Optional[np.ndarray] = None, 
                     observation: Optional[np.ndarray] = None,
                     iterations: int = 1000, 
                     burn_in: int = 500, 
                     cov_prior: float = 0.5, 
                     sd_noise: float = 0.1, 
                     adapt: bool = False, 
                     scale: float = 0.3,
                     proposal_sd: float = 0.3, 
                     diagnostic: bool = True,
                     plot_result: bool = True,
                     number_chains: int = 1,
                     proposal_algorithm: str = "MH", 
                     transformation: List[Any] = [], 
                     parallel: bool = False) -> Union[np.ndarray, float]:
        """
        Solve an inverse problem using a Bayesian framework with MCMC sampling.

        Args:
            mean_prior (np.ndarray): Mean of the prior distribution.
            x_data (np.ndarray): Input data for the model.
            max_parameter (float): Maximum parameter value for estimation, used for plotting.
            x_real (Optional[np.ndarray]): True parameter values for error computation.
            observation (Optional[np.ndarray]): Observed data to match against the model.
            iterations (int): Total number of MCMC iterations.
            burn_in (int): Number of burn-in iterations.
            cov_prior (float): Covariance of the prior distribution.
            sd_noise (float): Standard deviation of the noise to be added to observations.
            adapt (bool): Flag for enabling adaptive scaling in the MCMC proposal.
            scale (float): Scale factor for the proposal distribution.
            proposal_sd (float): Standard deviation of the proposal distribution.
            diagnostic (bool): Flag to enable diagnostic plots.
            plot_result (bool): Flag to plot results after estimation.
            number_chains (int): Number of MCMC chains to run.
            proposal_algorithm (str): MCMC algorithm to use ('MH', 'AM', 'CN', 'DREAMZ').
            transformation (List[Any]): List of transformations to apply to the input data.
            parallel (bool): Flag to enable parallel processing.

        Returns:
            Union[np.ndarray, float]: The estimated parameters and the error relative to the true parameters.
        """
        if iterations <= burn_in:
            warnings.warn("Number of steps insufficient, smaller or equal to burn-in", UserWarning)
            return

        if observation is not None:
            dim_obs, range_geometry = Inversion_helper.check_observation_shape(observation)
            if dim_obs is None:
                return
        else:
            warnings.warn("No observation nor data given", UserWarning)
            return

        self.inputs = x_data
        self.transformations = transformation

        m = x_real.shape[0]
        A = Inversion_helper.initialize_model(self._wrapper_prediction, proposal_algorithm, range_geometry, m)

        x = Gaussian(mean=mean_prior, cov=cov_prior)
        y = Gaussian(A(x), sqrtcov=proposal_sd)

        observation = Inversion_helper.perturbation(observation, sd_noise)

        MC_MC = MCMC_cuqi(y, x, observation, proposal_algorithm, m, scale, adapt, parallel)
        estimates, chains_params = MC_MC(mean_prior,iterations, burn_in, number_chains, diagnostic, plot_result)
        estimates = np.mean(estimates, axis=1)

        error = np.sum(Inversion_helper.relative_error(estimates, x_real))
        print(f"Error wrt true parameters: {error}")

        if plot_result:
            self._plot_diagnostics(estimates, x_real, max_parameter)

        return estimates, error, chains_params
    


    def _plot_diagnostics(self, estimates: np.ndarray, x_real: np.ndarray, max_par: float) -> None:
        """
        Plot diagnostics to compare the estimates with the true values.

        Args:
            estimates (np.ndarray): Estimated parameters from MCMC.
            x_real (np.ndarray): True parameter values.
            max_par (float): Maximum parameter value for estimation, used for plotting.
        """
        Inversion_helper.plot_hist(estimates, x_real, max_par)




class Neural_Network(INetwork):
    
    def __init__(
        self, 
        name: str, 
        params: Optional[Dict[str, Any]] = None, 
        data_train: Optional[np.ndarray] = None, 
        output_train: Optional[np.ndarray] = None, 
        data_val: Optional[np.ndarray] = None, 
        output_val: Optional[np.ndarray] = None, 
        getModel:Callable=None,
        N: int = 1000, 
        n: int = 10, 
        train: bool = True, 
        do_HPO: bool = False, 
        transformations: Optional[list] = None, 
        verbose: bool = False,
        device: str = None, 
        num_trials:int=15
    ):
        """
        Initializes the Neural_Network instance.
        
        Args:
            name (str): Name of the network.
            params (Optional[dict]): Hyperparameters of the network.
            data_train (Optional[np.ndarray]): Training data.
            output_train (Optional[np.ndarray]): Training outputs.
            data_val (Optional[np.ndarray]): HPO data.
            output_val (Optional[np.ndarray]): HPO outputs data.
            getModel (Callable): function to obtain the Network structure
            N (int): Number of epochs for training.
            n (int): Batch size for training.
            train (bool): Flag to indicate if training should be performed.
            do_HPO (bool): Flag to indicate if hyperparameter optimization is to be performed.
            transformations (Optional[list]): List of transformations to apply to the data.
            verbose (bool): Flag to indicate verbosity of the output.
            device (str): Device for computation (e.g., '/CPU:0' or '/GPU:0').
            num_trials (int): Number of HPO iterations
        """

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
        self.device = device if device else self._select_device()

        if getModel is None:
            raise ModelNotFoundError("Missing Network Structure!")
        # Set input and output shapes based on training data dimensions
        self.input_shape = self._get_shape(self._data_train)
        self.output_shape = self._get_shape(self._output_train)
        self.getModel=getModel
        # Perform hyperparameter optimization if required or if no parameters are provided
        if do_HPO:
            if output_val is None or data_val is None:
                warning_message = "Not enough data given for HPO!"
                warnings.warn(warning_message, UserWarning)
            self._params = self.HPO(data_val, output_val, num_trials)
            print("New parameters identified during HPO:")
            pprint(self._params)

        # Initialize the model with the given or optimized parameters
        if self._params is not None:
            self.model = getModel(self._params, self.input_shape, self.name, self.output_shape)

        # Train the model if required and if no HPO was performed
        if train and output_train is not None:
            self.hist = self.training(
                data_train, 
                output_train, 
                epoch=self._N, 
                batch=self._n,  
            )
            # Plot training loss after training is complete
            self.plot_training_loss()

    def _select_device(self) -> str:
        """Automatically select GPU if available, else fall back to CPU."""
        physical_devices = tf.config.list_physical_devices('GPU')
        return '/GPU:0' if physical_devices else '/CPU:0'


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
        Plots the training loss over the epochs and saves the plot in a specified directory.
        """
        if self.hist is not None and 'loss' in self.hist.history:
            # Create the folder if it doesn't exist
            save_dir = Clean.create_output_directory('output', 'simulation_result')


            # Define the file name
            base_filename = f"loss_{self.name}_NN"
            filename = base_filename + ".png"
            file_path = os.path.join(save_dir, filename)

            # If file exists, append a number
            counter = 1
            while os.path.exists(file_path):
                filename = f"{base_filename}_{counter}.png"
                file_path = os.path.join(save_dir, filename)
                counter += 1

            # Plotting
            plt.plot(self.hist.history['loss'][100:], label='Training Loss')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            
            # Save the plot instead of showing it
            plt.savefig(file_path)
            plt.close()
        else:
            warnings.warn("No training history or 'loss' key found.", UserWarning)


    @decorators.compute_time
    def training(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        epoch: int, 
        batch: int, 
    ) -> Any:
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
        # Set random seed for reproducibility
        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism()

        # Enable mixed precision training for memory optimization
        if self.device.startswith('/GPU'):
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')

        #self.hist = self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=0, callbacks=[callbacks] if callbacks else None)
        train_dataset = self.create_dataset(x, y, batch)

        self.hist = self.model.fit(train_dataset, epochs=epoch, verbose=0)
                
        tf.keras.backend.clear_session()     
        gc.collect()  # Explicit garbage collection after training to free memory

        return self.hist

    def create_dataset(self, x, y, batch_size):
        """
        Helper function to create a tf.data.Dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=len(x)).batch(batch_size)
        return dataset
    
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
            with decorators.Suppressor():  # Suppress output if verbosity is off
                return self.model.predict(x_test)

    def HPO(self, data_val: np.ndarray, output_val: np.ndarray, num_trials:int=15) -> Dict[str, Any]:
        """
        Performs hyperparameter optimization using Bayesian optimization.

        Args:
            data_val (np.ndarray): Training data.
            output_val (np.ndarray): Training outputs.
            num_trials (int): number of HPO iterations

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """
        # Set random seed for reproducibility
        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism()
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
            with tf.device(self.device if self.device else '/GPU:0'):  # Default to GPU if not specified and an appropriate GPU is present                
                loss = CrossValidations.kCrossVal_parallel(self._N, data_val, output_val,                        # Perform k-fold cross-validation to evaluate the model
                                          params, self.name, self._get_shape(data_val), 
                                          self._get_shape(output_val),getModel=self.getModel
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
        study.optimize(objective, n_trials=num_trials, n_jobs=-1) # Bayesian optimization

        # Return the best hyperparameters found
        return study.best_params
        

    def _set_level(self, x_data: np.ndarray, prev_steps: List, level: int = 1) -> None:            
        """
        Sets the level of the model and stores the input data and previous steps. Usefull in a multilevel scenario

        Args:
            x_data (np.ndarray): The input data for the model.
            prev_steps (List): A list of models that have been used in previous steps.
            level (int): The current level of the model (default is 1).
        """
        self.level = level
        self.inputs = x_data 
        self.prev_steps = prev_steps   # Stores the model list up to this point

    def _input_wrapper_prediction(self, x_test: np.ndarray) -> np.ndarray:     
        """
        Wraps the input for prediction, potentially combining it with previous model predictions.

        Args:
            x_test (np.ndarray): The test data for prediction.
            multi_input (bool): Flag indicating if the model expects multiple inputs.

        Returns:
            np.ndarray: The final input array for prediction.
        """
        x_final = super()._input_wrapper_prediction(x_test)

        if self.level > 1:
            for l in range(0, self.level - 1):
                x_final = np.hstack((x_final, self.prev_steps[l].prediction(Inversion_helper.concatenate_inputs(self.inputs, x_final))))
        
        return x_final

    
@dataclass(slots=True)
class MultiFidelity(INetwork):
    """
    MultiFidelity network that allows for the sequential training and prediction of models
    at different levels of fidelity.
    """
    names: List[str]=field(default_factory=list)
    params: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    data_train: Optional[List[np.ndarray]] = field(default_factory=list)
    output_train: Optional[List[np.ndarray]] = field(default_factory=list)
    data_val: Optional[List[np.ndarray]] = field(default_factory=list)
    output_val: Optional[List[np.ndarray]] = field(default_factory=list)
    getModel:Callable=None
    N: Optional[List[int]] = field(default_factory=lambda: [1000])
    n: Optional[List[int]] = field(default_factory=lambda: [10])
    train: bool = True
    do_HPO: bool = False
    verbose: bool = False
    device: str = None
    input_shape:int = 1
    output_shape:int = 1
    model_list: List[Neural_Network] = field(default_factory=list)
    num_trials:int=15

    def __post_init__(self):
        """
        Post-initialization to set up MultiFidelity network.
        """
        self.device = self.device if self.device else self._select_device()
        self._initialize_networks()

    def _initialize_networks(self):
        """
        Initialize and train each model in the MultiFidelity network.
        """


        # Validate the provided training data and output data
        if not self.data_train or not self.output_train or len(self.data_train) != len(self.output_train):
            raise ValueError('The training and output data are either incoherent or insufficient.')

        # Set input and output shapes based on the first dataset
        if self.data_train and len(self.data_train[0].shape) > 1:
            self.input_shape = self.data_train[0].shape[1]
        if self.output_train and len(self.output_train[0].shape) > 1:
            self.output_shape = self.output_train[0].shape[1]

 #       K.clear_session()

        if self.params is None:
            self.params = [None] * len(self.names)
        
        # Set up training data for the first model
        data_train_support = self.data_train[0] if self.data_train else None
        data_val_support = self.data_val[0] if self.data_val else None
        
        for index, name in enumerate(self.names):
            config=NetworkConfig(network_type=name,
                names=self.names,
                network_parameters=self.params[index],
                dataset_train=data_train_support,
                output_train=self.output_train[index],
                dataset_validation=data_val_support,
                output_validation=self.output_val[index] if self.output_val else None,
                epochs_number=self.N[index] if len(self.N) > index else 1000,
                batch_size=self.n[index] if len(self.n) > index else 10,
                train=self.train,
                do_HPO=self.do_HPO,
                verbose=self.verbose,
                device=self.device,
                num_trials=self.num_trials)
            model = NetworkFactory.build_network(
                    config, self.getModel
            )
            self.model_list.append(model)

            # Update training data for the next model
            if (index + 1) < len(self.data_train) and self.train:
                data_train_support = self.data_train[index + 1]
                data_val_support = self.data_val[index + 1] if self.data_val else None

                # Incorporate predictions from previous models into the training data
                for l in range(index + 1):
                    data_train_support = np.c_[data_train_support, self.model_list[l].prediction(data_train_support)]
                    if data_val_support is not None:
                        data_val_support = np.c_[data_val_support, self.model_list[l].prediction(data_val_support)]


    def plot_training_loss(self) -> None:
        """
        Plot training loss for all models in the MultiFidelity network.
        """
        for model in self.model_list:
            model.plot_training_loss()

    @property
    def params(self) -> List[Dict[str, Any]]:
        """Get the hyperparameters of each network."""
        return [model.params for model in self.model_list]

    @params.setter
    def params(self, value: List[Dict[str, Any]]) -> None:
        """Sets the hyperparameters, ensuring they are provided as a list."""
        if not isinstance(value, list) or len(value) != len(self.model_list):
            raise ValueError("Params must be a list with the same length as the number of models.")
        
        for i, model in enumerate(self.model_list):
            model.params = value[i]

    @property
    def N(self) -> List[int]:
        """Get the number of epochs for each network."""
        return [model.N for model in self.model_list]

    @N.setter
    def N(self, value: List[int]) -> None:
        """Sets the number of epochs for each network."""
        if not isinstance(value, list) or len(value) != len(self.model_list):
            raise ValueError("N must be a list with the same length as the number of models.")
        
        for i, model in enumerate(self.model_list):
            model.N = value[i]

    @property
    def n(self) -> List[int]:
        """Get the batch sizes for each network."""
        return [model.n for model in self.model_list]

    @n.setter
    def n(self, value: List[int]) -> None:
        """Sets the batch sizes for each network."""
        if not isinstance(value, list) or len(value) != len(self.model_list):
            raise ValueError("n must be a list with the same length as the number of models.")
        
        for i, model in enumerate(self.model_list):
            model.n = value[i]

    def performance(self, data_test: np.ndarray, output_test: np.ndarray, position: Optional[int] = None) -> Tuple[float, float]:
        """
        Evaluate the performance of the model at a specific position in the model list.
        """
        if position is None:
            position = len(self.model_list)
        elif not isinstance(position, int) or position > len(self.model_list):
            raise ValueError('Invalid model position specified.')

        data = data_test.copy()

        for i in range(position - 1):
            pred = self.model_list[i].prediction(data)
            data = np.c_[data, pred.reshape(-1, 1)] if len(pred.shape) <= 1 else np.c_[data, pred]

        pred = self.model_list[position - 1].prediction(data)
        output_test = output_test[:, np.newaxis] if len(output_test.shape) < len(pred.shape) else output_test

        test_mse, r2 = Inversion_helper.calculate_metrics(output_test, pred)
        print(f"Test MSE: {test_mse}")
        print(f"R^2: {r2}")

        return test_mse, r2
    
    def summary(self) -> None:
        """Print the summary of all models in the MultiFidelity network."""
        for model in self.model_list:
            model.summary()
        gc.collect()

    def prediction(self, data_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the multi-fidelity model.
        """
        outputs = data_test.copy()

        if len(outputs.shape) == 1:
            outputs = outputs.reshape(-1, 1)

        for index in range(len(self.names)):
            outputs = np.c_[outputs, self.model_list[index].prediction(outputs)]
        # outputs = np.zeros((data_test.shape[0], self.output_shape))
        # outputs[:, 0] = data_test.reshape(-1) if data_test.ndim == 1 else data_test[:, 0]

        # for index in range(len(self.names)):
        #     outputs[:, index + 1] = self.model_list[index].prediction(outputs[:, :index + 1])
        return outputs[:, -self.output_shape:]  # Return only the final model's prediction

    def save(self, file_paths: List[str]) -> None: 
        """ 
        Save all trained models to files.
        """
        for i, model in enumerate(self.model_list):
            model.save(file_paths[i])

    def load(self, file_paths: List[str]) -> None: 
        """ 
        Load trained models from files.
        """
        self.model_list = []
        for file_path in file_paths:
            model = load_model(file_path, custom_objects={'FourierLayer': FourierLayer, 'custom_activation': custom_activation})
            self.model_list.append(model)

        self.input_shape = self.model_list[0].input_shape[-1]
        self.output_shape = self.model_list[-1].output_shape[-1]

    def training(self, device: str = None):
        """
        Train all models in the MultiFidelity network.
        """
        if not self.params:
            raise ValueError("Params list is empty")
        
        if not self.data_train or not self.output_train:
            raise ValueError("Not enough data provided for training")
        
        data_train_support = self.data_train[0]

        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism()

        for index, name in enumerate(self.names):
            
            config=NetworkConfig(
                network_type=name,
                names=self.names,
                network_parameters=self.params[index],
                dataset_train=data_train_support,
                output_train=self.output_train[index],
                dataset_validation=None,
                output_validation=None,
                epochs_number=self.N[index],
                batch_size=self.n[index],
                train=True,
                do_HPO=False,
                verbose=self.verbose,
                device=device or self.device,
                num_trials=self.num_trials
            )

            model=NetworkFactory.build_network(config,self.getModel)
            self.model_list.append(model)

            if (index + 1) < len(self.data_train):
                data_train_support = self.data_train[index + 1]

                for l in range(index + 1):
                    data_train_support = np.c_[data_train_support, self.model_list[l].prediction(data_train_support)]
    
    def HPO(self, device: str = None,
            data_val: Optional[List[np.ndarray]] = None, output_val: Optional[List[np.ndarray]] = None)-> None:
        """
        Perform Hyperparameter Optimization (HPO) for all models in the MultiFidelity network.
        """
        if not self.params:
            raise ValueError("Params list is empty")
        
        if not self.data_train or not self.output_train:
            raise ValueError("Not enough data provided for training")
        
        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism()

        data_train_support = self.data_train[0]
        data_val_support = data_val[0] if data_val else None

        for index, name in enumerate(self.names):
            
            config=NetworkConfig(
                network_type=name,
                names=self.names,
                network_parameters=self.params[index],
                dataset_train=data_train_support,
                output_train=self.output_train[index],
                dataset_validation=data_val_support,
                output_validation=output_val[index] if output_val else None,
                epochs_number=self.N[index],
                batch_size=self.n[index],
                do_HPO=True,
                verbose=self.verbose,
                device=device or self.device,

            )

            model = NetworkFactory.build_network(config,self.getModel)

            self.model_list.append(model)

            if (index + 1) < len(self.data_train):
                data_train_support = self.data_train[index + 1]
                data_val_support = data_val[index + 1] if data_val else None

                for l in range(index + 1):
                    data_train_support = np.c_[data_train_support, self.model_list[l].prediction(data_train_support)]
                    if data_val_support is not None:
                        data_val_support = np.c_[data_val_support, self.model_list[l].prediction(data_val_support)]



    def get_output(self) -> np.ndarray:
        """
        Get the final output from the multi-fidelity model.
        """
        return self.model_list[-1].prediction(self.data_train[-1])


@dataclass
class LSTM_network(INetwork):
    name: str = 'LSTM'
    params: Optional[Dict[str, Any]] = None
    data_train: Optional[np.ndarray] = None
    output_train: Optional[np.ndarray] = None
    data_val: Optional[np.ndarray] = None
    output_val: Optional[np.ndarray] = None
    getModel:Callable=None
    N: int = 1000
    dim_input: int = 0
    dim_output: int = 0
    train: bool = True
    do_HPO: bool = False
    transformations: List[Any] = field(default_factory=list)
    verbose: bool = False
    device: str = None
    hist: Optional[Any] = field(default=None)
    input_shape: int = None
    output_shape: int = None
    num_trials: int = 15
    model: Model = field(default=None)
    input_support: Any = field(default=None)
    _forward_low_fidelity: Callable = field(default=None)

    def __post_init__(self):
        self.device = self.device if self.device else self._select_device()
        self.input_shape = self.dim_input
        self.output_shape = self.dim_output

        # Determine input and output shapes based on the provided training data
        if self.data_train is not None:
            self.input_shape = self.data_train.shape[-1] if len(self.data_train.shape) > 1 else self.input_shape
        if self.output_train is not None:
            self.output_shape = self.output_train.shape[-1] if len(self.output_train.shape) > 1 else self.output_shape

        # Disable training if data is missing
        if self.data_train is None or self.output_train is None:
            self.train = False

        # Perform hyperparameter optimization if required
        if self.do_HPO:
            if self.data_val is None or self.output_val is None:
                warnings.warn("Not enough data provided for HPO!", UserWarning)
            self.params = self.HPO(self.data_val, self.output_val)
            print("New parameters identified during HPO:")
            pprint(self.params)

        # Initialize the model if parameters are available
        if self.params:
            self.model = self.getModel(self.params, self.input_shape, self.name, self.output_shape)

        # Train the model if training is enabled
        if self.train and self.output_train is not None:
            self.hist = self.training(
                seq_length=self.params.get('sequence_length', 50),
                seq_freq=self.params.get('sequence_freq', 5),
                epoch=self.N
            )
            self.plot_training_loss()

    def set_parameters(self, params: Optional[Dict[str, Any]]) -> None:
        self.params = params

    @decorators.compute_time
    def training(self, seq_length: int, seq_freq: int, epoch: int) -> tf.keras.callbacks.History:

        input_train_seq, output_train_seq = self._sliding_windows(     
            self.data_train, self.output_train, seq_length, seq_freq)

        # Early stopping callback
        callback = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=self.params['patience'], restore_best_weights=True)

        # Enable mixed precision training for GPU
        if self.device.startswith('/GPU'):
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')

        # Set random seed for reproducibility
        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism()

        # Train the model
        with tf.device(self.device):
            self.hist = self.model.fit(
                input_train_seq, output_train_seq, epochs=epoch, 
                verbose=self.verbose, callbacks=[callback]
            )
        #tf.keras.backend.clear_session()
        return self.hist
        

    def plot_training_loss(self) -> None:
        """
        Plots the training loss over the epochs and saves the plot in a specified directory.
        """
        if self.hist is not None and 'loss' in self.hist.history:
            # Create the folder if it doesn't exist

            save_dir = Clean.create_output_directory('output', 'simulation_result')


            # Define the file name
            base_filename = f"loss_{self.name}"
            filename = base_filename + ".png"
            file_path = os.path.join(save_dir, filename)

            # If file exists, append a number
            counter = 1
            while os.path.exists(file_path):
                filename = f"{base_filename}_{counter}.png"
                file_path = os.path.join(save_dir, filename)
                counter += 1

            # Plotting
            plt.plot(self.hist.history['loss'][100:], label='Training Loss')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            
            # Save the plot instead of showing it
            plt.savefig(file_path)
            plt.close()
        else:
            warnings.warn("No training history or 'loss' key found.", UserWarning)

    def prediction(self, x_test: np.ndarray) -> np.ndarray:
        if self.verbose:
            y_pred = self.model.predict(x_test)
        else:
            with decorators.Suppressor():
                y_pred = self.model.predict(x_test)
        return y_pred

    @staticmethod
    def _sliding_windows(data_input: np.ndarray, data_output: np.ndarray, seq_length: int, freq: int = 1) -> Tuple[np.ndarray,np.ndarray]:
        x, y = [], []
        for i in range(data_input.shape[0]):
            for j in range(0, data_input.shape[1] - seq_length, freq):
                _x = data_input[i, j:(j + seq_length), :]
                _y = data_output[i, j:(j + seq_length), :]
                x.append(_x)
                y.append(_y)
        return np.array(x), np.array(y)

    def HPO(self, data_val: np.ndarray, output_val: np.ndarray, num_trials=15) -> Dict[str, Any]:
        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism()

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
            with tf.device(self.device):
                loss = CrossValidations.kCrossVal_parallel(
                    Nepo=self.N, x=data_val, y=output_val, 
                    params=params, name=self.name, input_shape=self.input_shape, 
                    output_shape=self.output_shape, p=num_trials, n_jobs=-1, getModel=self.getModel
                )

            trial.report(loss, step=trial.number)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return loss

        # Optimize using Optuna
        current_directory = os.getcwd()
        storage_path = os.path.join(current_directory, 'optuna_study.db')
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42), storage=f'sqlite:///{storage_path}')
        study.optimize(objective, n_trials=10, n_jobs=-1)
        return study.best_params

    def param_inverse(self, 
                      mean_prior: np.ndarray, 
                      x_data: np.ndarray,
                      max_parameter: float, 
                      cov_prior: Optional[np.ndarray] = None, 
                      cov_noise: float = 0.1, 
                      cov_likelihood: Optional[np.ndarray] = None, 
                      observation: Optional[np.ndarray] = None, 
                      x_real: Optional[np.ndarray] = None, 
                      number_chains: int = 1, 
                      iterations: int = 1000, 
                      burn_in: int = 500, 
                      levels: int = 1, 
                      diagnostic: bool = True, 
                      plot_result=True,
                      subsampling_rate=1,
                      rwmh_covariance: Optional[np.ndarray] = None, 
                      rmwh_scaling: float = 0.1, 
                      rwmh_adaptive: bool = True, 
                      proposal_algorithm: str = "MH", 
                      transformation: List[Any] = [], 
                      forward_low_fidelity: Optional[Callable] = None, 
                      force_sequential: bool = False, 
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        
        if forward_low_fidelity is None or not callable(forward_low_fidelity):
            raise KeyError("Provide forward_low_fidelity related to the class of the problem. It must be a callable function")

        self.input_support = next(iter(kwargs.values()))   # kwargs["input_support"]
        self._forward_low_fidelity = forward_low_fidelity

        return super().param_inverse(
            mean_prior=mean_prior, 
            x_data=x_data, 
            max_parameter=max_parameter,
            cov_prior=cov_prior, 
            cov_noise=cov_noise, 
            cov_likelihood=cov_likelihood, 
            observation=observation, 
            x_real=x_real, 
            number_chains=number_chains, 
            iterations=iterations, 
            burn_in=burn_in, 
            levels=levels, 
            plot_result=plot_result,
            subsampling_rate=subsampling_rate,
            diagnostic=diagnostic, 
            rwmh_covariance=rwmh_covariance, 
            rmwh_scaling=rmwh_scaling, 
            rwmh_adaptive=rwmh_adaptive, 
            proposal_algorithm=proposal_algorithm, 
            transformation=transformation,
            force_sequential=force_sequential
        )
    
    def inverse_cuqi(self, 
                     mean_prior: np.ndarray, 
                     x_data: np.ndarray, 
                     max_parameter: float, 
                     x_real: Optional[np.ndarray] = None, 
                     observation: Optional[np.ndarray] = None,
                     iterations: int = 1000, 
                     burn_in: int = 500, 
                     cov_prior: float = 0.5, 
                     sd_noise: float = 0.1, 
                     adapt: bool = False, 
                     scale: float = 0.3,
                     proposal_sd: float = 0.3, 
                     diagnostic: bool = True,
                     plot_result: bool = True,
                     number_chains: int = 1,
                     proposal_algorithm: str = "MH", 
                     transformation: List[Any] = [], 
                     parallel: bool = False,
                     forward_low_fidelity: Optional[Callable] = None,
                     **kwargs) -> Union[np.ndarray, float]:
        
        if forward_low_fidelity is None or not callable(forward_low_fidelity):
            raise KeyError("Provide forward_low_fidelity related to the class of the problem. It must be a callable function")

        self.input_support = next(iter(kwargs.values()))   # kwargs["input_support"]
        self._forward_low_fidelity = forward_low_fidelity

        return super().param_inverse(
                     mean_prior=mean_prior, 
                     x_data=x_data, 
                     max_parameter=max_parameter, 
                     x_real=x_real, 
                     observation=observation,
                     iterations=iterations, 
                     burn_in=burn_in, 
                     cov_prior=cov_prior, 
                     sd_noise=sd_noise, 
                     adapt=adapt, 
                     scale=scale,
                     proposal_sd=proposal_sd, 
                     diagnostic=diagnostic,
                     plot_result=plot_result,
                     number_chains=number_chains,
                     proposal_algorithm=proposal_algorithm, 
                     transformation=transformation, 
                     parallel=parallel)


    def _input_wrapper_prediction(self, x_test: np.ndarray) -> np.ndarray:
        x_final = super()._input_wrapper_prediction(x_test)
        prediction_input = self._forward_low_fidelity(x_final, self.inputs, self.input_support)
        return prediction_input
        
    
class Intermediate(INetwork):

    def __init__(self, 
                 name: str = "Inter", 
                 params: Optional[dict] = None, 
                 data_train: Optional[np.ndarray] = None, 
                 output_train: Optional[np.ndarray] = None, 
                 N: int = 1000, 
                 n: int = 10, 
                 getModel:Callable=None,
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
        self.getModel=getModel
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

    @decorators.compute_time
    def training(self, x: np.ndarray, y: np.ndarray, epoch: int, batch: int, device: str = '/CPU:0') -> Any:
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

        # Set random seed for reproducibility
        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism()

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
            with decorators.Suppressor():
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
        # Set random seed for reproducibility
        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism()

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
                loss = CrossValidations.kCrossVal_parallel(self._N, data_train, output_train, params, self.name, self.input_shape, self.output_shape, self.getModel)            
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
    

    def _input_wrapper_prediction(self, x_test: np.ndarray) -> np.ndarray:    # check
        x_final = super()._input_wrapper_prediction(x_test)

        if self.level >1 :

            for l in range(0,self.level-1):
                x_final = np.concatenate((x_final, self.prev_steps[l]._wrapper_prediction(x_final)), axis=1)
        
        return x_final

