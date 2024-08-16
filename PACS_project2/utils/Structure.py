import sys
import os
import numpy as np
from numpy import newaxis as _
import copy
import pickle
import time
from keras.models import load_model
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
from Helpers import *
from BIP_functions import *
from cuqi.distribution import Uniform, Gaussian,JointDistribution
from cuqi.sampler import MH
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Continuous2D, Discrete
import re
import tinyDA as tda
import optuna
from keras.src.callbacks.tensorboard import TensorBoard
import dask
from dask.distributed import Client
import multiprocessing
import joblib
from joblib import parallel_backend
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


# Define the types of networks as an enumeration for type safety and clarity.
class NetworkType(Enum):
    LF = "LF"
    MF = "MF"
    HF = "HF"
    HFLIN = "Hflin"
    HFPER = "Hfper"
    INTER = "Inter"
    LSTM = "LSTM"
    LSTM_SUPPORT="LSTM_support"
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
                      verbose: bool = False,
                      device: str = '/CPU:0',
                      profiler:TensorBoard=None) -> 'INetwork':
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
        - device (str): denotes GPU or CPU 
        - profiler (TensorBoard): allows to use the profiler to study the keras network performance

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
                return MultiFidelity(names, params, data_train, output_train, N, n, do_HPO, verbose, device=device, profiler=profiler)
            else:
                raise ValueError(f"Invalid network type: {network_type}")
            
        # Map the enum to the respective network class constructors.
        if network_type_enum in {NetworkType.LF, NetworkType.MF, NetworkType.HF, NetworkType.HFLIN, NetworkType.HFPER}:
            return Neural_Network(network_type, params, data_train, output_train, N, n, train, do_HPO, verbose,device=device, profiler=profiler)
        
        elif  network_type_enum ==  NetworkType["STEP"]: #network_type_enum == NetworkType.step:
            # Ensure data_train and output_train are lists for MultiFidelity networks.
            if not (isinstance(data_train, list) and isinstance(output_train, list)):
                raise ValueError("For MultiFidelity network, data_train and output_train must be lists of numpy arrays.")
            
            return MultiFidelity(names, params, data_train, output_train, N, n, do_HPO, verbose, device=device, profiler=profiler)
        
        elif network_type_enum == NetworkType.INTER:
            return Intermediate(name=network_type, params=params, data_train=data_train, output_train=output_train, N=N, n=n, train=train, do_HPO=do_HPO, device=device)
        
        elif network_type_enum == NetworkType.LSTM or network_type_enum==NetworkType.LSTM_SUPPORT:
            return LSTM_network(name=network_type,params=params, data_train=data_train, output_train=output_train, N=N, train=train, do_HPO=do_HPO, verbose=verbose, device=device)

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

    def set_train_sets(self, data_train,  output_train )->None:
        self.data_train = data_train
        self.output_train = output_train


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
            x_final = np.tile(x_final, (self.inputs.shape[0], 1))          
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
        # Process the test inputs using the internal wrapper
        x_final = self._input_wrapper_prediction(x_test, multi_input)
        
        # Determine the number of dimensions
        if x_final.ndim == 2:
            # 2D Case: self.inputs (n, 1) and x_final (n, dim-1)
            concatenated_input = np.concatenate((self.inputs, x_final), axis=1)
        elif x_final.ndim == 3:                                                               # commente 13-8-24, for reactionPOD
            # 3D Case: self.inputs (n, 1) and x_final (1, n, dim-1)
            inputs_expanded = np.expand_dims(self.inputs, axis=0)  # Shape (1, n, 1)
            concatenated_input = np.concatenate((inputs_expanded, x_final), axis=-1)
        else:
            raise ValueError("Unsupported number of dimensions for x_final")
        # elif x_final.ndim==3:
        #     concatenated_input=x_final
        # else:
        #      raise ValueError("Unsupported number of dimensions for x_final")

        # Perform the prediction and flatten the result
        return self.prediction(concatenated_input).flatten()



    @compute_time
    def param_inverse(self, mean_prior: np.ndarray, 
                      x_data: np.ndarray, 
                      max_par:float, 
                      cov_prior: Optional[np.ndarray] = None, 
                      cov_noise: float = 0.1, 
                      cov_likelihood: Optional[np.ndarray] = None,
                      y_obs: Optional[np.ndarray] = None, 
                      x_real: Optional[np.ndarray] = None, 
                      number_chains: int = 1, 
                      N: int = 1000, 
                      burn_in: int = 500, 
                      levels: int = 1, 
                      diagnostic: bool = True, 
                      rwmh_cov: Optional[np.ndarray] = None, 
                      rmwh_scaling: float = 0.1, 
                      rwmh_adaptive: bool = True, 
                      algo: str = "MH",
                      force_sequential: bool = False,
                      transformation: List[Any] = []) -> Tuple[np.ndarray, np.ndarray, List[dict]]:

        """
        Perform parameter inversion using MCMC sampling.

        Parameters:
        - mean_prior: Mean of the prior distribution.
        - x_data: Input data.
        - max_par: maximal value of the parameter to be estimated. Used only for graphical purposes
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
        - force_sequential: if True impose a sequential approach to the MCMC algorithm

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
            if not hasattr(self, 'model_list'):
                warnings.warn("1 level case considered", UserWarning)
            elif levels > len(self.model_list):
                warnings.warn("Number of levels is exceeding the number of models", UserWarning)
            elif levels==2:
                self.model_list[0].inputs=x_data
                self.model_list[1].inputs=x_data
                my_loglike = [tda.GaussianLogLike(y_obs, cov_likelihood), tda.GaussianLogLike(y_obs, cov_likelihood)]
                my_posterior = [tda.Posterior(my_prior, my_loglike[0], self.model_list[0].wrapper_prediction), tda.Posterior(my_prior, my_loglike[1], self.wrapper_prediction)]
            else:
                for l in range(levels):
                    self.model_list[i]._set_level(x_data=x_data,prev_steps=self.model_list[0:l],level=l+1)    

                my_loglike = [tda.GaussianLogLike(y_obs, cov_likelihood) for _ in range(levels)]
                my_posterior = [tda.Posterior(my_prior, my_loglike[i], self.model_list[i].wrapper_prediction) for i in range(levels)]
        else:
            my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
            my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)

        if levels>1: 
            for i in range(levels):
                self.model_list[i].inputs=x_data

        if rwmh_cov is None:
            rwmh_cov = np.eye(len(x_real))

        estimates, param_results = MCMC(my_posterior=my_posterior, N=N, burnin=burn_in, n=number_chains, diagnostic=diagnostic, rwmh_cov=rwmh_cov, rmwh_scaling=rmwh_scaling, rwmh_adaptive=rwmh_adaptive, algo=algo, force_sequential=force_sequential,dim=dim)

        if diagnostic:
            plot_hist(estimates, x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real),max_par)
        
        error = np.abs(estimates - x_real) / np.abs(x_real + 1e-10)
        
        return estimates, error, param_results    
   

    @compute_time
    def inverse_cuqi(self,mean_prior: np.ndarray, 
                    x_data: np.ndarray,
                    max_par:float,
                    x_real: Optional[np.ndarray] = None, 
                    y_obs: Optional[np.ndarray] = None, 
                    N: int = 1000, 
                    burn_in: int = 500, 
                    cov_prior: float = 0.5, 
                    sd_noise: float = 0.1,
                    adapt: bool = False, 
                    scale: float = 0.3, 
                    proposal_sd: float = 0.3, 
                    x_init: Optional[Union[int, float, np.ndarray]] = None, #####
                    diagnostic: bool = True, 
                    number_chains: int = 1, 
                    algo: str = "MH", 
                    transformation: List = [], 
                    parallel: bool=False) -> Union[np.ndarray, float]:
        """
        Perform Bayesian inference using the CUQI framework.

        Parameters:
        - mean_prior: np.ndarray : Prior mean
        - x_data: np.ndarray : Input data
        - max_par float: maximal value of the parameter to be estimated. Used only for graphical purposes
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
        - parallel: bool: True to parallelize computation of Markov Chains
        
        Returns:
        - estimates: np.ndarray : Estimated parameters
        - error: float : Error with respect to true parameters
        """

        # Set inputs and transformations
        self.inputs = x_data
        self.transformations = transformation

        # Check if observations are provided
        if y_obs is not None:
            dim = y_obs.shape[0] # dim is the number of observation points

        else:
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)
            return

        if len(y_obs.shape)==1 or y_obs.shape[1]==1:
            dim_obs=1
            range_geometry=Continuous1D(dim)
        elif y_obs.shape[1]==2:
            dim_obs=2
            range_geometry=Continuous2D(dim)
        else:
            warning_message = "impossible for Cuqipy to manage a problem with 3 or more equations"
            warnings.warn(warning_message, UserWarning)
            return

        # Check if the number of steps is greater than burn-in period
        if N <= burn_in:
            warning_message = "Number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning)
            return
        
        m = x_real.shape[0]
        # Select algorithm and initialize CuqiModel and Gaussian objects
        if algo == "NUTS":
            fun = Function(self.wrapper_prediction)
            A = CuqiModel(forward=self.wrapper_prediction, jacobian=fun.compute_jacobian, range_geometry=range_geometry, domain_geometry=Discrete(m))
        else:
            A = CuqiModel(forward=self.wrapper_prediction, range_geometry=range_geometry, domain_geometry=Discrete(m))

        x = Gaussian(mean=mean_prior, cov=cov_prior)

        y = Gaussian(A(x), sqrtcov=proposal_sd)

        # Generate or perturb observations
        if y_obs is None:                                   # incoerente con prima riga che da errore... , poi che senso ha definire osservazione a caso... ragiona e se caso elimina
            y_obs = y(x=x_real).sample()
        else:
            y_obs = y_obs + np.random.normal(loc=0., scale=sd_noise, size=y_obs.shape)

        # Run MCMC to get estimates
        estimates,parameters = MCMC_cuqi(y, x, y_obs, N, m, burn_in, number_chains, diagnostic=diagnostic, algo=algo, adapt=adapt, scale=scale, parallel=parallel)
        estimates = np.mean(estimates, axis=1)

        # Calculate and print error
        error = np.linalg.norm(estimates - x_real)
        print(f"Error wrt true parameters: {error}")

        # Plot diagnostics if required
        if diagnostic:
            plot_hist(estimates, x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real),max_par)

        return estimates, error,parameters

    @staticmethod
    def save(self, file_path: str) -> None: 
        """ 
        Save the trained LSTM model to a file. 
        Args: 
            file_path (str): The path where the model will be saved. 
        """ 
        self.model.save(file_path) 
        print(f"Model saved to {file_path}") 

    def load(self, file_path: str) -> None: 
        """ 
        Load a trained LSTM model from a file. 
        Args: 
            file_path (str): The path from where the model will be loaded. 
        """ 
        self.model = load_model(file_path, custom_objects={'FourierLayer': FourierLayer, 'custom_activation':custom_activation}) 
        self.input_shape=self.model.inputs[0][-1]
        self.output_shape=self.model.outputs[0][-1]

        print(f"Model loaded from {file_path}")

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
        verbose: bool = False,
        device: str = '/CPU:0', 
        profiler: TensorBoard = None
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
            device (str): denotes GPU or CPU 
        """
        K.clear_session()
        self.name = name
        self.type="NN"
        self.params = params
        self.N = N
        self.n = n
        self.verbose = verbose
        self.hist = None
        self.data_train = data_train
        self.output_train = output_train
        self.transformations = transformations if transformations is not None else []
        self.inputs = None
        self.level=0
        # Set input and output shapes based on training data dimensions
        self.input_shape = self._get_shape(data_train)
        self.output_shape = self._get_shape(output_train)
        # Perform hyperparameter optimization if required or if no parameters are provided
        if do_HPO or params is None:
            if output_train is None or data_train is None:
                warning_message = "Not enough data given!"
                warnings.warn(warning_message, UserWarning)
            self.params = self.HPO(data_train, output_train, device=device)

        # Initialize the model
        self.model = getModel(self.params, self.input_shape, self.name, self.output_shape)

        # Train the model if required
        if train:
            self.hist = self.training(data_train, output_train, epoch=self.N, batch=self.n,device=device, callbacks=profiler) 
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
    def training(self, x: np.ndarray, y: np.ndarray, epoch: int, batch: int, device: str = '/CPU:0', callbacks: TensorBoard=None) -> Any:
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
        if callbacks is not None:
            with tf.device(device):

                self.hist = self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=0, callbacks=[callbacks])
        else:
            with tf.device(device):

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

    def HPO(self, data_train: np.ndarray, output_train: np.ndarray, device: str = '/CPU:0') -> Dict[str, Any]:
        """
        Performs hyperparameter optimization using Bayesian optimization.

        Args:
            data_train (np.ndarray): Training data.
            output_train (np.ndarray): Training outputs.

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """
        print("HPO name ", self.name )
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
                loss = kCrossVal_parallel(self.n, self.N, data_train, output_train, params, self.name, self.input_shape, self.output_shape)            
            return loss

        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        tf.get_logger().setLevel('ERROR')
        storage = 'sqlite:///C:/Users/Giacomo/Documents/GitHub/pacs_new/PACS_project2/optuna_study.db'
        study = optuna.create_study(direction="minimize",storage=storage)
        # Use n_jobs=2 since it is stable on your system
        study.optimize(objective, n_trials=10, n_jobs=-1)
        
        best_params = study.best_params
        return best_params
    # def HPO(self, data_train: np.ndarray, output_train: np.ndarray, device: str = '/CPU:0') -> Dict[str, Any]:
    #     """
    #     Performs hyperparameter optimization using Bayesian optimization.

    #     Args:
    #         data_train (np.ndarray): Training data.
    #         output_train (np.ndarray): Training outputs.

    #     Returns:
    #         Dict[str, Any]: The best hyperparameters found.
    #     """


    #     #client = Client(n_workers=multiprocessing.cpu_count(), threads_per_worker=1)

    #     def objective(trial):
    #         K.clear_session()
    #         params = {
    #             "nodes": trial.suggest_int("nodes", 4, 64, log=True),
    #             "l2weight": trial.suggest_float("l2weight", 1e-4, 1e-1, log=True),  # Adjusted to use suggest_float
    #             "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),  # Adjusted to use suggest_float
    #             "kernel_init": trial.suggest_categorical("kernel_init", ["uniform", "glorot_uniform"]),
    #             "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),

    #         }
    #         #with tf.device(device):
    #         loss = kCrossVal(self.n, self.N, data_train, output_train, params, self.name, self.input_shape, self.output_shape)
            
    #         return loss
    #     #logging.getLogger('tensorflow').setLevel(logging.ERROR)
    #     #tf.get_logger().setLevel('ERROR')
    #     study = optuna.create_study(direction="minimize")
    #     study.optimize(objective, n_trials=5, n_jobs=3)#, client=client)  # Parallelize trials
    #     best_params = study.best_params
    #     return best_params



    # def HPO(self, data_train: np.ndarray, output_train: np.ndarray, device: str = '/CPU:0') -> Dict[str, Any]:
    #     """
    #     Performs hyperparameter optimization using Bayesian optimization.

    #     Args:
    #         data_train (np.ndarray): Training data.
    #         output_train (np.ndarray): Training outputs.

    #     Returns:
    #         Dict[str, Any]: The best hyperparameters found.
    #     """
    #     def objective(trial):
    #         K.clear_session()
    #         params = {
    #             "nodes": trial.suggest_int("nodes", 4, 64, log=True),
    #             "l2weight": trial.suggest_float("l2weight", 1e-4, 1e-1, log=True),
    #             "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
    #             "kernel_init": trial.suggest_categorical("kernel_init", ["uniform", "glorot_uniform"]),
    #             "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),
    #         }
    #         with tf.device(device):
    #             model = getModel(params, self.input_shape, self.name, self.output_shape)
    #             #model.compile(loss="mean_squared_error", optimizer=params["opt"])
    #             score = kCrossVal(self.n, self.N, data_train, output_train, model, self.name, self.input_shape, self.output_shape)
    #         return -score  # Maximize the score

    #     logging.getLogger('tensorflow').setLevel(logging.ERROR)
    #     tf.get_logger().setLevel('ERROR')
    #     study = optuna.create_study(direction="maximize")
    #     study.optimize(objective, n_trials=5, n_jobs=-1)
    #     best_params = study.best_params
    #     return best_params

    # def HPO(self, data_train: np.ndarray, output_train: np.ndarray, device: str = '/CPU:0') -> Dict[str, Any]:
    #     def objective(trial):
    #         K.clear_session()
    #         params = {
    #             "nodes": trial.suggest_int("nodes", 4, 64, log=True),
    #             "l2weight": trial.suggest_float("l2weight", 1e-4, 1e-1, log=True),
    #             "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
    #             "kernel_init": trial.suggest_categorical("kernel_init", ["uniform", "glorot_uniform"]),
    #             "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),
    #         }
    #         with tf.device(device):
    #             loss = kCrossVal(self.n, self.N, data_train, output_train, params, self.name, self.input_shape, self.output_shape)
    #         return loss
        
    #     logging.getLogger('tensorflow').setLevel(logging.ERROR)
    #     tf.get_logger().setLevel('ERROR')
    #     study = optuna.create_study(direction="minimize")
    #     study.optimize(objective, n_trials=5, n_jobs=-1)
    #     # with parallel_backend('multiprocessing'):
    #     #     study.optimize(objective, n_trials=5, n_jobs=-1)
    #     best_params = study.best_params
    #     return best_params

# delete
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
    

    def _set_level(self, x_data:np.ndarray, prev_steps:List, level:int=1)-> None:
        self.level=level
        self.inputs=x_data 
        self.prev_steps=prev_steps   # it contains the model_list up to this point

    def _input_wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:
        x_final = super()._input_wrapper_prediction(x_test, multi_input)

        if self.level >1 :

            for l in range(0,self.level-1):
                x_final = np.concatenate((x_final, self.prev_steps[l]._wrapper_prediction(x_final, multi_input)), axis=1)
        
        return x_final
    
class MultiFidelity(INetwork):
        # REMARK: IN MULTIFIDELITY NN C'è LA POSSIBILTIà DI NON FARE TRAINING 
    def __init__(self, 
                 names: List[str], 
                 params: Optional[List[dict]] = None, 
                 data_train: Optional[List[np.ndarray]] = None, 
                 output_train: Optional[List[np.ndarray]] = None, 
                 N: Optional[List[int]] = None, 
                 n: Optional[List[int]] = None, 
                 do_HPO: bool = False, 
                 verbose: bool = False,
                 device: str ='/CPU:0', profiler: TensorBoard=None):
        """
        Initialize MultiFidelity network.

        Args:
            names (List[str]): Names of the networks.
            params (Optional[List[dict]]): Parameters for each network.
            data_train (Optional[List[np.ndarray]]): Training data for the networks.
            output_train (Optional[List[np.ndarray]]): Training outputs for each network.
            N (Optional[List[int]]): Number of epochs for each network.
            n (Optional[List[int]]): Batch sizes for each network.
            do_HPO (bool): Whether to perform hyperparameter optimization.
            verbose (bool): Whether to print verbose output.
            device (str): denotes GPU or CPU 
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

        #count = 1
        data_train_support=data_train[0]
        for index, name in enumerate(names):
            model = NetworkFactory.build_network(
                name,
                params=params[index],
                data_train=data_train_support, #[count - 1],
                output_train=output_train[index],#][count - 1],
                N=self.Ns[index],
                n=self.ns[index],#[count - 1],
                train=True,
                do_HPO=do_HPO,
                verbose=verbose,
                device=device,
                profiler=profiler
            )
            self.model_list.append(model)

            if (index+1)<len(data_train):

                data_train_support=data_train[index+1]

                for l in range(index+1):
                    data_train_support=np.c_[data_train_support, self.model_list[l].prediction(data_train_support)]
            # if (index + 1) == (self.steps - 1) * (count - 1) + 1:
            #     count += 1
            # for i in range(count - 1, len(data_train)):
            #     data_train[i] = np.c_[data_train[i], model.prediction(data_train[i])]


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
                 name:str='LSTM',
                 params: Optional[dict] = None, 
                 data_train: Optional[np.ndarray] = None, 
                 output_train: Optional[np.ndarray] = None, 
                 N: int = 1000, 
                 dim_input:int=0,
                 dim_output:int=0,
                 train: bool = True, 
                 do_HPO: bool = False, 
                 transformations: List[Any] = [], 
                 verbose: bool = False,
                 device: str = '/CPU:0'):
        """
        Initialize LSTM_network instance.
        Args:
            params (Optional[dict]): Hyperparameters for the network.
            data_train (Optional[np.ndarray]): Training data.
            output_train (Optional[np.ndarray]): Training outputs.
            N (int): Number of epochs for training.
            dim_input:int=0,
            dim_output:int=0,
            train (bool): Flag to indicate if training should be performed.
            do_HPO (bool): Flag to indicate if hyperparameter optimization is to be performed.
            transformations (List[Any]): List of transformations to apply to the data.
            verbose (bool): Flag to indicate verbosity of the output.
            device (str): denotes GPU or CPU 

        """
        self.params = params
        self.name = name 
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
        elif dim_input>1:
            self.input_shape = dim_input

        if output_train is not None and len(output_train.shape) > 1:
            self.output_shape = output_train.shape[-1]
        elif dim_output>1:
            self.output_shape = dim_output

        if data_train is None or output_train is None:
            train = False


        # Perform hyperparameter optimization if required       
        if self.params is None and train:
                warnings.warn("Not enough data given!", UserWarning)
            
        if do_HPO:
            self.params = self.HPO(data_train, output_train,device=device)
        # if do_HPO or params is None:
        #     if output_train is None or data_train is None:
        #         warnings.warn("Not enough data given!", UserWarning)
        #     self.params = self.HPO(data_train, output_train,device=device)
        
        # Initialize the model
        if self.params is not None:
            self.model = getModel(self.params,self.input_shape,self.name,self.output_shape)  # dim_input = n_POD + 2, dim_output = n_POD

            # Train the model if required
            if train: 
                self.hist = self.training(int(self.params['sequence_length']),int(self.params['sequence_freq']),epoch=self.N,device=device) 
                self.plot_training_loss()

        else: 
            print(f"class instance {self.name} created, load a keras model")

        # if no params, no dataset, no HPO, then you must load

    def set_parameters(self, params: Optional[dict] )->None:
        self.params=params


    @compute_time
    def training(self,seq_length,seq_freq,epoch, device: str = '/CPU:0'):
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
        # with tf.device(device):

        #     self.hist = self.model.fit(self.input_train_seq, self.output_train_seq, epochs=epoch, verbose = self.verbose, callbacks=[callback])
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

    # def save(self, file_path: str) -> None: 
    #     """ 
    #     Save the trained LSTM model to a file. 
    #     Args: 
    #         file_path (str): The path where the model will be saved. 
    #     """ 
    #     self.model.save(file_path) 
    #     print(f"Model saved to {file_path}") 
 
    # def load(self, file_path: str) -> None: 
    #     """ 
    #     Load a trained LSTM model from a file. 
    #     Args: 
    #         file_path (str): The path from where the model will be loaded. 
    #     """ 
    #     self.model = load_model(file_path, custom_objects={'FourierLayer': FourierLayer, 'custom_activation':custom_activation}) 
    #     self.input_shape=self.model.inputs[0][-1]
    #     self.output_shape=self.model.outputs[0][-1]

    #     print(f"Model loaded from {file_path}")

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

    def param_inverse(  self, 
                        mean_prior: np.ndarray, 
                        x_data: np.ndarray,
                        max_par:float, 
                        cov_prior: Optional[np.ndarray] = None, 
                        cov_noise: float = 0.1, 
                        cov_likelihood: Optional[np.ndarray] = None, 
                        y_obs: Optional[np.ndarray] = None, 
                        x_real: Optional[np.ndarray] = None, 
                        number_chains: int = 1, 
                        N: int = 1000, 
                        burn_in: int = 500, 
                        levels: int = 1, 
                        diagnostic: bool = True, 
                        rwmh_cov: Optional[np.ndarray] = None, 
                        rmwh_scaling: float = 0.1, 
                        rwmh_adaptive: bool = True, 
                        algo: str = "MH", 
                        transformation: List[Any] = [], 
                        forward_low_fidelity: Optional[Callable] = None, 
                        force_sequential: bool = False, 
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform parameter inversion using MCMC sampling, potentially utilizing a low fidelity forward model.

        Parameters:
        - mean_prior (np.ndarray): Mean of the prior distribution.
        - x_data_support (np.ndarray): Input data for the basis reduction.
        - x_data (np.ndarray): Input data.
        - cov_prior (Optional[np.ndarray]): Covariance of the prior distribution (optional).
        - cov_noise (float): Noise covariance.
        - cov_likelihood (Optional[np.ndarray]): Covariance of the likelihood (optional).
        - y_obs (Optional[np.ndarray]): Observed data (optional).
        - x_real (Optional[np.ndarray]): Real parameters (optional).
        - number_chains (int): Number of MCMC chains.
        - N (int): Number of MCMC iterations.
        - burn_in (int): Number of burn-in iterations.
        - levels (int): Number of model levels.
        - diagnostic (bool): Flag to enable diagnostic plots.
        - rwmh_cov (Optional[np.ndarray]): Covariance matrix for RWMH proposal (optional).
        - rmwh_scaling (float): Scaling factor for RWMH.
        - rwmh_adaptive (bool): Flag for adaptive RWMH.
        - algo (str): MCMC algorithm to use ("MH", "AM", "CN", "DREAMZ").
        - transformation (List[Any]): List of transformations to apply.
        - forward_low_fidelity (Optional[Callable]): Low fidelity forward model function (optional).
        - force_sequential (bool): True to avoid parallelization  
        - *args: Additional positional arguments used depending on the class with the data of the LSTM model

        Returns:
        - Tuple[np.ndarray, np.ndarray]: MCMC estimates of the parameters and relative error of the estimates.
        """

        # Check if forward_low_fidelity is provided and is a callable function
        if forward_low_fidelity is None or not callable(forward_low_fidelity):
            raise KeyError("Provide forward_low_fidelity related to the class of the problem. It must be a callable function")

        self.input_support = next(iter(kwargs.values()))
        # Store the forward_low_fidelity function for later use
        self._forward_low_fidelity = forward_low_fidelity
        # Call the parent class's param_inverse method with the provided parameters
        return super().param_inverse(
            mean_prior=mean_prior, 
            x_data=x_data, 
            max_par=max_par,
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
            transformation=transformation,
            force_sequential = force_sequential
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
        prediction_input = self._forward_low_fidelity(x_final, self.inputs, self.input_support)

        return prediction_input#self.prediction(prediction_input).flatten()
        
    
    def HPO(self, data_train: np.ndarray, output_train: np.ndarray, device: str = '/CPU:0') -> Dict[str, Any]:
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
                "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),
                "sequence_length": trial.suggest_int("sequence_length", 10, 100),
                "sequence_freq": trial.suggest_int("sequence_freq", 2, 10),
                "patience": trial.suggest_int("patience", 50, 100),
                "lay": trial.suggest_int("lay", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.05, 0.5, log=True)

            }
            
            with tf.device(device):

                loss = kCrossVal_parallel(N=self.data_train.shape[0], Nepo=self.N, x=data_train, y=output_train, 
                                        params=params, name=self.name, input_shape=self.input_shape, 
                                        output_shape=self.output_shape, p=5, n_jobs=-1)
            return loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20, n_jobs=-1)
        best_params = study.best_params
        return best_params
    
class Intermediate(INetwork):

    def __init__(self, 
                 name: str = "Inter", 
                 params: Optional[dict] = None, 
                 data_train: Optional[np.ndarray] = None, 
                 output_train: Optional[np.ndarray] = None, 
                 N: int = 1000, 
                 n: int = 10, 
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
        self.params = params
        self.N = N
        self.n = n
        self.verbose = verbose
        self.hist = None
        self.data_train = data_train
        self.output_train = output_train
        self.input_shape = 1
        self.output_shape = 1
        self.inputs = None
        self.transformations = transformations

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
            self.params = self.HPO(data_train, output_train,device=device)

        # Create the model
        self.model = getModel(self.params, self.input_shape, self.name, self.output_shape)

        if train:
            self.hist = self.training(data_train, output_train, epoch=self.N, batch=self.n,device=device) 
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

    @compute_time
    def training(self, x: np.ndarray, y: np.ndarray, epoch: int, batch: int, device: str = '/CPU:0', callbacks: TensorBoard=None) -> Any:
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
        if callbacks is not None:
            with tf.device(device):

                self.hist = self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=0, callbacks=[callbacks])
        else:
            with tf.device(device):

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
            with Suppressor():
                return self.model.predict(x_test)

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

    def objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Objective function for hyperparameter optimization.

        Args:
            params (dict): Hyperparameters.

        Returns:
            Dict[str, Any]: Result of cross-validation.
        """
        K.clear_session()
        loss = kCrossVal(self.n, self.N, self.data_train, self.output_train, params, self.name, self.input_shape, self.output_shape)
        return {"loss": loss, "params": params, "status": STATUS_OK}

    def HPO(self, data_train: np.ndarray, output_train: np.ndarray, device: str = '/CPU:0') -> Dict[str, Any]:
        """
        Performs hyperparameter optimization using Bayesian optimization.

        Args:
            data_train (np.ndarray): Training data.
            output_train (np.ndarray): Training outputs.

        Returns:
            Dict[str, Any]: The best hyperparameters found.
        """
        print("HPO name ", self.name )
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
                loss = kCrossVal_parallel(self.n, self.N, data_train, output_train, params, self.name, self.input_shape, self.output_shape)            
            return loss

        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        tf.get_logger().setLevel('ERROR')
        storage = 'sqlite:///C:/Users/Giacomo/Documents/GitHub/pacs_new/PACS_project2/optuna_study_LSTM.db'
        study = optuna.create_study(direction="minimize",storage=storage)
        # Use n_jobs=2 since it is stable on your system
        study.optimize(objective, n_trials=4, n_jobs=-1)
        
        best_params = study.best_params
        return best_params
    

    def _input_wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:    # check
        x_final = super()._input_wrapper_prediction(x_test, multi_input)

        if self.level >1 :

            for l in range(0,self.level-1):
                x_final = np.concatenate((x_final, self.prev_steps[l]._wrapper_prediction(x_final, multi_input)), axis=1)
        
        return x_final

# class Intermediate(INetwork):
#     def __init__(self, 
#                  params: Optional[dict] = None, 
#                  data_train: Optional[List[np.ndarray]] = None, 
#                  output_train: Optional[List[np.ndarray]] = None, 
#                  N: int = 1000, 
#                  n: int = 10, 
#                  train: bool = True, 
#                  do_HPO: bool = False, 
#                  verbose: bool = False):
#         """
#         Initialize Intermediate network.

#         Args:
#             params (Optional[dict]): Parameters for the network.
#             data_train (Optional[List[np.ndarray]]): Training data for the network.
#             output_train (Optional[List[np.ndarray]]): Training outputs for the network.
#             N (int): Number of epochs for training.
#             n (int): Batch size for training.
#             train (bool): Whether to train the model.
#             do_HPO (bool): Whether to perform hyperparameter optimization.
#             verbose (bool): Whether to print verbose output.
#         """
#         self.params = params
#         self.name = "Inter" 
#         self.N = N
#         self.n = n
#         self.verbose = verbose
#         self.hist = None
#         self.data_train = data_train
#         self.output_train = output_train
#         self.transformations = []
        
#         self.input_shape = 1
#         self.output_shape = 1

#         if len(data_train) != 2 or len(output_train) != 2:
#             raise ValueError('The data are incoherent or insufficient')
                
#         # Concatenate data for training
#         data_train = np.concatenate((data_train[1], data_train[0]), axis=0)
#         output_train = np.concatenate((output_train[1], output_train[0]), axis=0)
        
#         # Determine input and output shapes
#         if data_train is not None and len(data_train.shape) > 1:
#             self.input_shape = data_train.shape[1]

#         if output_train is not None and len(output_train.shape) > 1:
#             self.output_shape = output_train.shape[1]

#         if do_HPO or params is None:
#             if output_train is None or data_train is None:
#                 warnings.warn("Not enough data given!", UserWarning)
#             self.params = self.HPO(data_train, output_train)

#         # Create the model
#         self.model = getModel(self.params, self.input_shape, self.name, self.output_shape)

#         if train:
#             self.hist = self.model.fit(data_train, output_train, epochs=self.N, batch_size=self.n, verbose=self.verbose) 
#             self.plot_training_loss()

#     @compute_time
#     def training(self, x: np.ndarray, y: np.ndarray, epoch: int, batch: int) -> Any:
#         """
#         Train the model on the given data.

#         Args:
#             x (np.ndarray): Training data.
#             y (np.ndarray): Training labels.
#             epoch (int): Number of epochs.
#             batch (int): Batch size.

#         Returns:
#             Any: Training history.
#         """
#         self.hist = self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=self.verbose) 
#         return self.hist
    
#     def plot_training_loss(self) -> None:
#         """
#         Plot the training loss.
#         """
#         if self.hist is not None and 'loss' in self.hist.history:
#             plt.plot(self.hist.history['loss'][100:], label='Training Loss')
#             plt.title('Mean Squared Error (MSE) over Epochs')
#             plt.xlabel('Epochs')
#             plt.ylabel('MSE')
#             plt.legend()
#             plt.show()

#     def prediction(self, x_test: List[np.ndarray]) -> np.ndarray:
#         """
#         Make predictions using the model.

#         Args:
#             x_test (List[np.ndarray]): Test data.

#         Returns:
#             np.ndarray: Predictions.
#         """
#         if len(x_test) != 2:
#             raise ValueError("Not enough data given")
        
#         x_test = np.concatenate((x_test[1], x_test[0]), axis=0)

#         if self.verbose:
#             return self.model.predict(x_test)
#         else:
#             with Suppressor():
#                 return self.model.predict(x_test)

#     def performance(self, data_test: List[np.ndarray], output_test: List[np.ndarray]) -> Tuple[float, float]:
#         """
#         Evaluate the performance of the model.

#         Args:
#             data_test (List[np.ndarray]): Test data.
#             output_test (List[np.ndarray]): Test labels.

#         Returns:
#             Tuple[float, float]: Test MSE and R^2 score.
#         """
#         data_test = np.concatenate((data_test[1], data_test[0]), axis=0)
#         output_test = np.concatenate((output_test[1], output_test[0]), axis=0)
        
#         pred = self.prediction(data_test)

#         if len(output_test.shape) < len(pred.shape):
#             output_test = output_test[:, np.newaxis]

#         test_mse = np.mean(np.square(output_test - pred))
#         print(f"Test MSE: {test_mse}")

#         r2 = 1 - np.sum(np.square(output_test - pred)) / np.sum(np.square(output_test - np.mean(output_test)))
#         print(f"R^2: {r2}")
        
#         return test_mse, r2

#     def objective(self, par: dict) -> Dict[str, Any]:
#         """
#         Objective function for hyperparameter optimization.

#         Args:
#             par (dict): Hyperparameters.

#         Returns:
#             Dict[str, Any]: Result of cross-validation.
#         """
#         K.clear_session()
#         CVres = kCrossVal(self.n, self.N, self.data_train, self.output_train, par, self.name, self.input_shape)
#         return {"loss": CVres, "params": par, "status": STATUS_OK} 

#     def HPO(self, data_train: np.ndarray, output_train: np.ndarray) -> dict:
#         """
#         Hyperparameter Optimization (to be implemented).

#         Args:
#             data_train (np.ndarray): Training data.
#             output_train (np.ndarray): Training labels.

#         Returns:
#             dict: Best hyperparameters.
#         """
#         pass  # Implement hyperparameter optimization logic here



