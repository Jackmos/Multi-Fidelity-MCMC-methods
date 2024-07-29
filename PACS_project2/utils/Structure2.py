from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
from keras.models import save_model
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers import Layer
from tensorflow.keras.layers import (
    concatenate,
)  
from sklearn.utils import extmath

from keras.regularizers import l2, l1
from sklearn.model_selection import KFold
from keras.optimizers import Adam, Nadam, Adamax, RMSprop
import keras.backend as K
import tensorflow as tf
import keras as kr
import h5py
import sys
import os
import warnings
import copy
from cuqi.distribution import Uniform, Gaussian,JointDistribution
from cuqi.sampler import MH
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Discrete
import time
import tinyDA as tda
from scipy.stats import multivariate_normal, beta   
import arviz as az

from abc import ABCMeta, abstractstaticmethod, abstractmethod
from functools import reduce

import importlib
from pathlib import Path
import sys

# connect to specific definition
def load_context_functions(context_folder):
    try:
        sys.path.append('context_folder')
        return True
    except ImportError:
        print(f"folder path {context_folder} not found")
        return False
from module_utils import *




from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
from typing import Tuple
from typing import List, Optional, Union, Any
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

class Function:
    def __init__(self, func):
        """
        Initialize the class with a given function.

        Parameters:
        - func: The function for which we want to compute the Jacobian.
        """
        self.func = func

    def compute_jacobian(self, x):
        """
        Compute the Jacobian matrix for the given point x.

        Parameters:
        - x: The point at which to compute the Jacobian.

        Returns:
        - The Jacobian matrix.
        """
        x = np.atleast_1d(x).astype(float)
        n = len(x)
        m = len(self.func(x))

        jacobian = np.zeros((m, n))

        h = 1e-8
        for i in range(n):
            x_copy = x.copy()

            # Perturb the i-th element positively
            x_copy[i] += h
            func_pos = self.func(x_copy)

            # Perturb the i-th element negatively
            x_copy[i] -= 2 * h
            func_neg = self.func(x_copy)

            # Compute partial derivative using central difference
            jacobian[:, i] = (func_pos - func_neg) / (2 * h)

        return jacobian
    
def compute_time(function):
    def wrapper(*args, **kwargs):
        init = time.time()
        res = function(*args, **kwargs)
        end = time.time()
        timespam = end - init
        print(f"The function {function.__name__} took {timespam} seconds.")
        return res
    return wrapper



@contextmanager
def Suppressor() -> None:
    original_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout



class INetwork(metaclass=ABCMeta):

    @abstractmethod
    def prediction(self) -> None:
        pass

    @abstractmethod
    def performance(self) -> None:
        pass

    @abstractmethod
    def inverse_cuqi(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def save() -> None:
        pass

# def sliding_windows(data_input, data_output, seq_length, freq=1):
#     x = []
#     y = []

#     for i in range(data_input.shape[0]):
#         for j in range(0, data_input.shape[1] - seq_length, freq):
#             _x = data_input[i, j:(j + seq_length), :]
#             _y = data_output[i, j:(j + seq_length), :]
#             x.append(_x)
#             y.append(_y)

#     return np.array(x), np.array(y)



def sliding_windows(data_input: np.ndarray, data_output: np.ndarray, seq_length: int, freq: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sliding windows of sequences from the input and output data.

    Parameters:
    - data_input (np.ndarray): Input data array of shape (num_samples, num_timesteps, num_features).
    - data_output (np.ndarray): Output data array of shape (num_samples, num_timesteps, num_output_features).
    - seq_length (int): Length of each sequence window.
    - freq (int): Frequency of sliding. Defaults to 1.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Arrays of input and output windows.
    """
    # Extract the dimensions of the input data
    num_samples: int = data_input.shape[0]
    num_timesteps: int = data_input.shape[1]
    num_features: int = data_input.shape[2]

    # Extract the number of output features from the output data
    num_output_features: int = data_output.shape[2]

    # Calculate the number of windows per sample
    windows_per_sample: int = (num_timesteps - seq_length) // freq + 1

    # Calculate the total number of windows
    total_windows: int = num_samples * windows_per_sample

    # Preallocate the arrays for input and output windows
    x: np.ndarray = np.empty((total_windows, seq_length, num_features), dtype=data_input.dtype)
    y: np.ndarray = np.empty((total_windows, seq_length, num_output_features), dtype=data_output.dtype)

    idx: int = 0
    for i in range(num_samples):
        for j in range(0, num_timesteps - seq_length + 1, freq):
            x[idx] = data_input[i, j:j + seq_length]
            y[idx] = data_output[i, j:j + seq_length]
            idx += 1

    return x, y



# class Neural_Network(INetwork):
    
#     def __init__(self,name,params=None,data_train=None,output_train=None,N=1000,n=10,train=True,do_HPO=False,transformations=[],verbose=False):
#         K.clear_session()
#         self.name=name
#         self.params=params
#         self.N=N    # epochs
#         self.n=n    # batch size    
#         self.verbose=verbose
#         self.hist=None
#         self.data_train=data_train
#         # NON VALE LA PENA SALVARSI SOLO LE DIMENSIONI?
#         self.output_train=output_train
#         self.transformations=transformations
#         self.input_shape=1
#         self.output_shape=1
#         self.inputs=None

#         if data_train is not None and len(data_train.shape) > 1:
#             self.input_shape = data_train.shape[1]
      
#         if output_train is not None and len(output_train.shape) > 1:
#             self.output_shape = output_train.shape[1] 

#         if (do_HPO or params is None):
#             if (output_train is None or data_train is None):
#                 warning_message = "Not enough data given!"
#                 warnings.warn(warning_message, UserWarning)
#             self.params=self.HPO(data_train,output_train)
#         self.model = getModel(self.params,self.input_shape,self.name,self.output_shape)

#         if(train):
#             self.hist=self.training(data_train,output_train,epoch=self.N,batch=self.n) 
#             plt.plot(self.model.history.history['loss'][100:], label='Training Loss')
#             plt.title('Mean Squared Error (MSE) over Epochs')
#             plt.xlabel('Epochs')
#             plt.ylabel('MSE')
#             plt.legend()
#             plt.show()

#     def variable_input(self, input_discr):
#         # function to give a value at the input variable when solving the inverse problem from a Multilevel perspective
#         self.inputs = input_discr
          
#     @compute_time
#     def training(self,x,y,epoch,batch):
#         # DUBBIO : va definita variabile hist?
#         self.hist=self.model.fit(x,y,epochs=epoch,batch_size=batch,verbose=0) 
#         return self.hist

#     def prediction(self,x_test):
#         if(self.verbose):
#             y_pred=self.model.predict(x_test)#[:,0]             # <-------------  commentato il 6/03
#         else:
#             with Suppressor():

#                 y_pred = self.model.predict(x_test)#[:,0]       # <--------------- commentato il 6/03
#         return y_pred  
    
    # def wrapper_prediction(self,x_test):
    #     # ricontrolla
    #     # ---- caso shear cube ----
    #     # if (x_test.ndim==1 and self.data_train.shape[1]>1 and len(self.transformations)==0):   # e.g. Shear cube case
    #     #     x_test=x_test.reshape(-1,1).T
    #     # elif (x_test.ndim==1):
    #     #     x_test=x_test.reshape(-1,1)
    #     # -----------
    #     print("x_test")
    #     print(x_test.shape)
    #     if (x_test.ndim==1):
    #         x_test=x_test.reshape(-1,1)
    #         # ----- sostituito 
    #     x_test=x_test.T # VEDI
    #     #print(x_test.shape)
    #     if self.transformations:
    #         x_final = reduce(lambda acc, trasf: np.hstack([acc, trasf(acc)]), self.transformations, x_test)
    #     else:
    #         x_final = x_test
    #     print("x_final")
    #     print(x_final)    
    #     x_final=np.tile(x_final,(self.inputs.shape[0],1))
    #     #print(x_final.shape)
    #     #print(np.concatenate((self.inputs,x_final),axis=1).shape)
    #     rep=self.prediction(np.concatenate((self.inputs,x_final),axis=1)).flatten()
    #     #print(rep.shape)
    #     return rep


import warnings
import matplotlib.pyplot as plt
from keras import backend as K
#from some_module import getModel, compute_time, Suppressor  # Assuming these are imported from the relevant modules
from functools import wraps
import numpy as np
from numpy import newaxis as _
from hyperopt import fmin, tpe, hp, Trials
import numpy as np
from typing import Dict, Any
import optuna  # Import Optuna for efficient HPO
from joblib import Parallel, delayed  # For parallel processing
import time  # For benchmarking



def profile(func):     # NON NECESSARIO, C'è COMPUTE TIME
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper
# ............................................................................................
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
        verbose: bool = False
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
        """
        K.clear_session()
        self.name = name
        self.params = params
        self.N = N
        self.n = n
        self.verbose = verbose
        self.hist = None
        self.data_train = data_train
        self.output_train = output_train
        self.transformations = transformations if transformations is not None else []
        self.inputs = None

        # Set input and output shapes based on training data dimensions
        self.input_shape = self._get_shape(data_train)
        self.output_shape = self._get_shape(output_train)
        # Perform hyperparameter optimization if required or if no parameters are provided
        if do_HPO or params is None:
            if output_train is None or data_train is None:
                warning_message = "Not enough data given!"
                warnings.warn(warning_message, UserWarning)
            self.params = self.HPO(data_train, output_train)

        # Initialize the model
        self.model = getModel(self.params, self.input_shape, self.name, self.output_shape)

        # Train the model if required
        if train:
            self.hist = self.training(data_train, output_train, epoch=self.N, batch=self.n) 
            self._plot_training_loss()

    def _get_shape(self, data: Optional[np.ndarray]) -> int:
        """
        Gets the shape of the data.

        Args:
            data (Optional[np.ndarray]): The data to get the shape of.

        Returns:
            int: The shape of the data.
        """
        return data.shape[1] if data is not None and len(data.shape) > 1 else 1

    def _plot_training_loss(self) -> None:
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

    def variable_input(self, input_discr: Any) -> None:
        """
        Sets the input variable when solving the inverse problem from a Multilevel perspective.
        
        Args:
            input_discr (Any): The input discriminator.
        """
        self.inputs = input_discr

    @compute_time
    @profile
    def training(self, x: np.ndarray, y: np.ndarray, epoch: int, batch: int) -> Any:
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
        self.hist = self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=0)
        return self.hist

   # @profile
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



    def wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:
        """
        Wrapper for making predictions with transformations and multi-input support.

        Args:
            x_test (np.ndarray): Test data.
            multi_input (bool): Flag to indicate if multi-input is used.

        Returns:
            np.ndarray: Prediction results.
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
            return self.prediction(np.concatenate((self.inputs, x_final), axis=1)).flatten()
        else:
            warning_message = "Inputs are not set."
            warnings.warn(warning_message, UserWarning)
            return np.array([])
    
    #@compute_time
    def param_inverse(self, mean_prior, x_data, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, levels=1,diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
            # I should insert the 
        self.transformations=transformation
        self.inputs=x_data

        # rivedere la questione dimensioni (x_real!=y_obs)
        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    

        if (N<=burn_in):
            warning_message = "number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning) 

        if(cov_prior is None):
            cov_prior=mean_prior*0.2

        if(cov_likelihood is None):
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
        # prior
        # if(mean_prior.shape[0]==1):
        #     my_prior=beta(1.,1.)
        # else:
        #     my_prior = multivariate_normal(mean_prior, cov_prior) 
        my_prior = multivariate_normal(mean_prior, cov_prior) 
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape) # check   
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape) 
            print(y_obs.shape)
            #if(len(y_obs.shape)==1):   # if introduced when working on LV 3 outputs, I think the condition was added with the benchmark cases. CHeck
            y_obs=y_obs.flatten() #



        # likelihood
        if levels>1:
            my_loglike=[]
            my_posterior=[]
            if(levels>len(self.model_list)):
                warning_message = "number of levels is exceeding the number of models"
                warnings.warn(warning_message, UserWarning)
            else:
                # caso l> 2, ancora da implementare per gestione input reti     
                #for i in range(levels): 
                #    self.model_list[i].variable_input(x_data)  
                #    my_loglike = [my_loglike, tda.GaussianLogLike(y_obs, cov_likelihood)]
                #    my_posterior = [my_posterior, tda.Posterior(my_prior, my_loglike[-1], self.model_list[i].wrapper_prediction)]   # attention, you should put ML model at the end
                ## SEZIONE TEMPORANEA
                for i in range(levels):
                    self.model_list[i].variable_input(x_data) 
                    my_loglike = [my_loglike, tda.GaussianLogLike(y_obs, cov_likelihood)]
                my_posterior = [tda.Posterior(my_prior, my_loglike[-1], self.model_list[0].wrapper_prediction),tda.Posterior(my_prior, my_loglike[-1], self.wrapper_prediction)] 
                print(len(my_posterior))
                ## SEZIONE TEMPORANEA
        
        else:
            my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
            my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
           # print(my_posterior)
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)

        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        error=np.abs(estimates-x_real)/np.abs(x_real+1e-10)
        return estimates, error
    
    # def save(self,discr="_",place=""):
    #     print("saving the model ...")
    #     folder=os.path.join(place,f"{self.name}_model{discr}.h5")
    #     save_model(self.model,folder)

    
    # def performance(self,data_test,output_test):

    #     data=copy.copy(data_test)
    #     pred=self.prediction(data)#[:,0]pred=self.wrapper_prediction(data_test)#[:,0]
        
    #     if len(output_test.shape)<len(pred.shape):
    #         output_test=output_test[:,np.newaxis]
    #     test_mse = np.mean(np.square(output_test - pred))
    #     print(f"Test MSE: {test_mse}")

    #     r2= 1 - np.sum(np.square(output_test - pred)) / np.sum(
    #         np.square(output_test - np.mean(output_test))
    #     )
    #     print(f"R^2: {r2}")
        
    #     return (test_mse,r2)

    def save(self, discr: str = "_", place: str = "") -> None:
        """
        Save the model to a specified directory.

        Args:
            discr (str): Discriminator to add to the model filename.
            place (str): Directory where the model should be saved.
        """
        print("Saving the model ...")
        folder = os.path.join(place, f"{self.name}_model{discr}.h5")
        save_model(self.model, folder)

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
    
    @compute_time
    def inverse(self, mean_prior, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, cov_search=False,transformations=[], algo="MH"):
        
        self.transformations=transformations
        
        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    
        if (N<=burn_in):
            warning_message = "number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning)        
        
        if(cov_prior is None):
            cov_prior=mean_prior*0.2
        if(cov_likelihood is None):
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
            
            
        my_prior = multivariate_normal(mean_prior, cov_prior) # modo per settare uniforme?
        
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=0.5,size=y_obs.shape)
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)    # 
        
        my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
        my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
    
    
    # CUQI CASE
    @compute_time
    def inverse_cuqi(self, mean_prior, x_real=None,y_obs=None, N=1000, burn_in=500, cov_prior=0.5,sd_noise=0.1,proposal_sd=0.3,adapt=False,scale=0.3,x_init=None,diagnostic=True, number_chains=1, algo="MH"):

        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    

        if (N<=burn_in):
            warning_message = "number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning)        
              
        if (x_init is None):
            x_init=np.random.rand(dim)
        elif isinstance(x_init, (int, float)):
            x_init=x_init*np.ones(dim)

        if algo=="NUTS":
            fun=Function(self.wrapper_prediction)
            A=CuqiModel(forward=self.wrapper_prediction,jacobian=fun.compute_jacobian, range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
            x=Gaussian(mean=mean_prior,cov=cov_prior)
        else:    
            A=CuqiModel(forward=self.wrapper_prediction,range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
            x=Uniform(np.zeros(dim),np.ones(dim)*mean_prior*2)  # prior  # mettere if con diverse prior?
        y=Gaussian(mean=A(x),cov=proposal_sd)
        
        if(y_obs is None):
            y_obs=y(x=x_real).sample()
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=sd_noise,size=y_obs.shape)

        # posterior=JointDistribution(y,x)(y=y_obs)
        
        # sampler=MH(posterior,x0=x_init)   # rendere variabile per altri sampler
        # samples=sampler.sample_adapt(N-burn_in,burn_in)
        
        # if(diagnostic is True):
        #     samples.plot_trace()
        #     mean=samples.mean()
        #     print(f"Mean values= {mean}")
        #     ESS=samples.compute_ess()
        #     print(f"ESS= {ESS}")
        #     samples.plot_autocorrelation()
        #     plot_hist(estimates,x_real, self.wrapper_prediction(mean), self.wrapper_prediction(x_real))
        
        estimates= MCMC_cuqi(y,x,y_obs,N,burn_in,number_chains,diagnostic=diagnostic,algo=algo, adapt=adapt, scale=scale)
        estimates=np.mean(estimates,axis=1)
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
    


    # def objective(self,par): 
    #     #print(self.name)
    #     K.clear_session()
    #     CVres = kCrossVal(self.n,self.N,self.data_train,self.output_train,par,self.name,self.input_shape)  # ATTENZIONE!! NEL CASO LF, data_train E output_train dovrebbero essere HF (vedere codice Nlf di esempio), correggere
    #     return {"loss": CVres, "params": par, "status": STATUS_OK}   

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
    # def HPO(self,data_train,output_train):

    #     MAX_EVAL = 5

    #     bayes_trials = Trials()
    #     opt_list = ["Adam", "Adamax"]
    #     kernel_list = ["uniform", "glorot_uniform"]
    #     aux_dic = {"opt": opt_list, "kernel_init": kernel_list}
    #     space = {
    #     "nodes": hp.qloguniform("nodes", np.log(4), np.log(64), 2),
    #     "l2weight": hp.loguniform("l2weight", np.log(0.0001), np.log(1)),
    #     "lr": hp.loguniform("lr", np.log(0.0001), np.log(0.1)),
    #     "kernel_init": hp.choice("kernel_init", kernel_list),
    #     "opt": hp.choice("opt", opt_list),
    #     }

    #     best_params = fmin(fn = self.objective,
    #                     space = space,
    #                     algo = tpe.suggest,
    #                     max_evals = MAX_EVAL,
    #                     trials = bayes_trials)
    #     transfBestparam(best_params, aux_dic)
    #     #self.params=best_params
        
    #     print("Best parameters from the HPO:")
    #     print(best_params)

    #     return best_params


    # def HPO(self, data_train: np.ndarray, output_train: np.ndarray) -> Dict[str, Any]:
    #     """
    #     Performs hyperparameter optimization using Bayesian optimization.

    #     Args:
    #         data_train (np.ndarray): Training data.
    #         output_train (np.ndarray): Training outputs.

    #     Returns:
    #         Dict[str, Any]: The best hyperparameters found.
    #     """
    #     MAX_EVAL = 1
    #     bayes_trials = Trials()
    #     opt_list = ["Adam", "Adamax"]
    #     kernel_list = ["uniform", "glorot_uniform"]
    #     aux_dic = {"opt": opt_list, "kernel_init": kernel_list}

    #     space = {
    #         "nodes": hp.qloguniform("nodes", np.log(4), np.log(64), 2),
    #         "l2weight": hp.loguniform("l2weight", np.log(0.0001), np.log(0.1)),  # Adjusted range
    #         "lr": hp.loguniform("lr", np.log(0.0001), np.log(0.1)),
    #         "kernel_init": hp.choice("kernel_init", kernel_list),
    #         "opt": hp.choice("opt", opt_list),
    #     }

    #     best_params = fmin(
    #         fn=self.objective,
    #         space=space,
    #         algo=tpe.suggest,
    #         max_evals=MAX_EVAL,
    #         trials=bayes_trials
    #     )
        
    #     transfBestparam(best_params, aux_dic)

    #     print("Best parameters from the HPO:")
    #     print(best_params)
        
    #     return best_params

    def HPO(self, data_train: np.ndarray, output_train: np.ndarray) -> Dict[str, Any]:
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
                "l2weight": trial.suggest_float("l2weight", 1e-4, 1e-1, log=True),  # Adjusted to use suggest_float
                "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),  # Adjusted to use suggest_float
                "kernel_init": trial.suggest_categorical("kernel_init", ["uniform", "glorot_uniform"]),
                "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),
                # "nodes": trial.suggest_int("nodes", 4, 64),
                # "l2weight": trial.suggest_loguniform("l2weight", 1e-4, 1e-1),
                # "lr": trial.suggest_loguniform("lr", 1e-4, 1e-1),
                # "kernel_init": trial.suggest_categorical("kernel_init", ["uniform", "glorot_uniform"]),
                # "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),
            }
            loss = kCrossVal(self.n, self.N, data_train, output_train, params, self.name, self.input_shape, self.output_shape)
            return loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2, n_jobs=-1)  # Parallelize trials
        best_params = study.best_params
        return best_params




# import optuna  # Import Optuna for efficient HPO
# from joblib import Parallel, delayed  # For parallel processing
# import time  # For benchmarking

# # Enable XLA JIT compilation
# tf.config.optimizer.set_jit(True)

# class Neural_Network(INetwork):
    
#     def __init__(
#         self, 
#         name: str, 
#         params: Optional[Dict] = None, 
#         data_train: Optional[np.ndarray] = None, 
#         output_train: Optional[np.ndarray] = None, 
#         N: int = 1000, 
#         n: int = 10, 
#         train: bool = True, 
#         do_HPO: bool = False, 
#         transformations: Optional[list] = None, 
#         verbose: bool = False
#     ):
#         """
#         Initializes the Neural_Network instance.
        
#         Args:
#             name (str): Name of the network.
#             params (Optional[dict]): Hyperparameters of the network.
#             data_train (Optional[np.ndarray]): Training data.
#             output_train (Optional[np.ndarray]): Training outputs.
#             N (int): Number of epochs for training.
#             n (int): Batch size for training.
#             train (bool): Flag to indicate if training should be performed.
#             do_HPO (bool): Flag to indicate if hyperparameter optimization is to be performed.
#             transformations (Optional[list]): List of transformations to apply to the data.
#             verbose (bool): Flag to indicate verbosity of the output.
#         """
#         K.clear_session()
#         self.name = name
#         self.params = params
#         self.data_train = data_train
#         self.output_train = output_train
#         self.N = N
#         self.n = n
#         self.train_flag = train #
#         self.do_HPO = do_HPO  #
#         self.transformations = transformations if transformations else []
#         self.verbose = verbose
#         self.inputs = None
#         self.hist = None

#         # Set input and output shapes based on training data dimensions
#         self.input_shape = self._get_shape(data_train)
#         self.output_shape = self._get_shape(output_train)
#         # Perform hyperparameter optimization if required or if no parameters are provided        
#         if self.do_HPO or self.params is None:
#             if not (self.data_train and self.output_train):
#                 warnings.warn("Not enough data given!", UserWarning)
#             self.params = self.HPO(data_train, output_train)

#         # Initialize the model
#         self.model = getModel(self.params, self.input_shape, self.name, self.output_shape)
#         # Train the model if required        
#         if self.train_flag:
#             self.hist = self.training(data_train, output_train, self.N, self.n)
#             self._plot_training_loss()

#     def _get_shape(self, data: Optional[np.ndarray]) -> int:
#         """
#         Gets the shape of the data.

#         Args:
#             data (Optional[np.ndarray]): The data to get the shape of.

#         Returns:
#             int: The shape of the data.
#         """
#         return data.shape[1] if data is not None and len(data.shape) > 1 else 1

#     def _plot_training_loss(self) -> None:
#         """
#         Plot the training loss over epochs.
#         """
#         if self.hist is not None and 'loss' in self.hist.history:
#             plt.plot(self.hist.history['loss'][100:], label='Training Loss')
#             plt.title('Mean Squared Error (MSE) over Epochs')
#             plt.xlabel('Epochs')
#             plt.ylabel('MSE')
#             plt.legend()
#             plt.show()

#     def variable_input(self, input_discr: Any) -> None:
#         """
#         Sets the input variable when solving the inverse problem from a Multilevel perspective.
        
#         Args:
#             input_discr (Any): The input discriminator.
#         """
#         self.inputs = input_discr

#     @compute_time
#     @profile
#     def training(self, x: np.ndarray, y: np.ndarray, epoch: int, batch: int) -> Any:
#         """
#         Trains the model on the given data.
        
#         Args:
#             x (np.ndarray): Training data.
#             y (np.ndarray): Training outputs.
#             epoch (int): Number of epochs for training.
#             batch (int): Batch size for training.

#         Returns:
#             Any: The training history.
#         """
#         # Early stopping callback
#        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)

#         # Enable JIT compilation for this specific function
#        # @tf.function(jit_compile=True)
#         def train_step():
#             return self.model.fit(x, y, epochs=epoch, batch_size=batch, verbose=0)#, callbacks=[early_stopping])
        
#         return train_step()



#     def HPO(self, data_train: np.ndarray, output_train: np.ndarray) -> Dict[str, Any]:
#         """
#         Performs hyperparameter optimization using Bayesian optimization.

#         Args:
#             data_train (np.ndarray): Training data.
#             output_train (np.ndarray): Training outputs.

#         Returns:
#             Dict[str, Any]: The best hyperparameters found.
#         """
#         def objective(trial):
#             K.clear_session()
#             params = {
#                 "nodes": trial.suggest_int("nodes", 4, 64),
#                 "l2weight": trial.suggest_loguniform("l2weight", 1e-4, 1e-1),
#                 "lr": trial.suggest_loguniform("lr", 1e-4, 1e-1),
#                 "kernel_init": trial.suggest_categorical("kernel_init", ["uniform", "glorot_uniform"]),
#                 "opt": trial.suggest_categorical("opt", ["Adam", "Adamax"]),
#             }
#             loss = kCrossVal(self.n, self.N, data_train, output_train, params, self.name, self.input_shape, self.output_shape)
#             return loss

#         study = optuna.create_study(direction="minimize")
#         study.optimize(objective, n_trials=100, n_jobs=-1)  # Parallelize trials
#         best_params = study.best_params
#         return best_params

#     def objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Objective function to minimize during hyperparameter optimization.

#         Args:
#             params (Dict[str, Any]): Hyperparameters to evaluate.

#         Returns:
#             Dict[str, Any]: The loss value and parameters.
#         """
#         K.clear_session()
#         loss = kCrossVal(self.n, self.N, self.data_train, self.output_train, params, self.name, self.input_shape, self.output_shape)
#         return {"loss": loss, "params": params, "status": STATUS_OK}

#     def prediction(self, x_test: np.ndarray) -> np.ndarray:
#         """
#         Makes predictions on the test data.

#         Args:
#             x_test (np.ndarray): Test data.

#         Returns:
#             np.ndarray: Predicted values.
#         """
#         if self.verbose:
#             return self.model.predict(x_test)
#         else:
#             with Suppressor():
#                 return self.model.predict(x_test)

#     def wrapper_prediction(self, x_test: np.ndarray, multi_input: bool = False) -> np.ndarray:
#         """
#         Wrapper for making predictions with transformations and multi-input support.

#         Args:
#             x_test (np.ndarray): Test data.
#             multi_input (bool): Flag to indicate if multi-input is used.

#         Returns:
#             np.ndarray: Prediction results.
#         """
#         if x_test.ndim == 1:
#             x_test = x_test.reshape(-1, 1)
#         x_test = x_test.T
#         if self.transformations:
#             x_final = reduce(lambda acc, transf: np.hstack([acc, transf(acc)]), self.transformations, x_test)
#         else:
#             x_final = x_test

#         if self.inputs is not None:
#             x_final = np.tile(x_final, (self.inputs.shape[0], 1))
#             return self.prediction(np.concatenate((self.inputs, x_final), axis=1)).flatten()
#         else:
#             warnings.warn("Inputs are not set.", UserWarning)
#             return np.array([])





#     def param_inverse( # recheck
#         self, 
#         mean_prior, 
#         x_data, 
#         cov_prior=None, 
#         cov_noise=0.1, 
#         cov_likelihood=None, 
#         y_obs=None, 
#         x_real=None, 
#         number_chains=1, 
#         N=1000, 
#         burn_in=500, 
#         levels=1,
#         diagnostic=True,
#         rwmh_cov=None,
#         rmwh_scaling=0.1,
#         rwmh_adaptive=True, 
#         algo="MH", 
#         transformation=[]
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         self.transformations = transformation
#         self.inputs = x_data
#         dim = x_real.shape[0] if x_real is not None else y_obs.shape[0] if y_obs is not None else None
#         if dim is None:
#             warnings.warn("No observation nor data given", UserWarning)
#         if N <= burn_in:
#             warnings.warn("Number of steps insufficient, smaller or equal than burn-in", UserWarning)
#         cov_prior = cov_prior if cov_prior is not None else mean_prior * 0.2
#         cov_likelihood = cov_likelihood if cov_likelihood is not None else cov_noise**2 * np.eye(dim)

#         my_prior = multivariate_normal(mean_prior, cov_prior)
#         if y_obs is None:
#             y_obs = self.wrapper_prediction(x_real) + np.random.normal(0., cov_noise, size=x_real.shape)
#         else:
#             y_obs = y_obs + np.random.normal(0., cov_noise, size=y_obs.shape)
#             y_obs = y_obs.flatten()
        
#         my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
#         my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
        
#         if levels > 1:
#             my_loglike = [my_loglike for _ in range(levels)]
#             my_posterior = [tda.Posterior(my_prior, loglike, model.wrapper_prediction) for loglike, model in zip(my_loglike, self.model_list[:levels])]

#         rwmh_cov = rwmh_cov if rwmh_cov is not None else np.eye(len(x_real))
#         estimates = MCMC(my_posterior, N, burn_in, number_chains, diagnostic=diagnostic, rwmh_cov=rwmh_cov, rmwh_scaling=rmwh_scaling, rwmh_adaptive=rwmh_adaptive, algo=algo, dim=dim)

#         if diagnostic:
#             plot_hist(estimates, x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
#         error = np.abs(estimates - x_real) / np.abs(x_real + 1e-10)
#         return estimates, error

#     def save(self, discr: str = "_", place: str = "") -> None:
#         """
#         Save the model to a specified directory.

#         Args:
#             discr (str): Discriminator to add to the model filename.
#             place (str): Directory where the model should be saved.
#         """
#         folder = os.path.join(place, f"{self.name}_model{discr}.h5")
#         save_model(self.model, folder)

#     def performance(self, data_test: np.ndarray, output_test: np.ndarray) -> Tuple[float, float]:
#         """
#         Evaluate the performance of the model on test data.

#         Args:
#             data_test (np.ndarray): Test data.
#             output_test (np.ndarray): Expected output data.

#         Returns:
#             Tuple[float, float]: Test Mean Squared Error (MSE) and R^2 score.
#         """
#         pred = self.prediction(data_test)
#         if len(output_test.shape) < len(pred.shape):
#             output_test = output_test[:, None]
#         test_mse = np.mean(np.square(output_test - pred))
#         r2 = 1 - np.sum(np.square(output_test - pred)) / np.sum(np.square(output_test - np.mean(output_test)))
#         return test_mse, r2    
        
   
    
#     # CUQI CASE
#     @compute_time
#     def inverse_cuqi(self, mean_prior, x_real=None,y_obs=None, N=1000, burn_in=500, cov_prior=0.5,sd_noise=0.1,proposal_sd=0.3,adapt=False,scale=0.3,x_init=None,diagnostic=True, number_chains=1, algo="MH"):

#         if x_real is not None:
#             dim = x_real.shape[0]
#         elif y_obs is not None:
#             dim=y_obs.shape[0]
#         else: 
#             warning_message = "No observation nor data given"
#             warnings.warn(warning_message, UserWarning)    

#         if (N<=burn_in):
#             warning_message = "number of steps insufficient, smaller or equal than burn-in"
#             warnings.warn(warning_message, UserWarning)        
              
#         if (x_init is None):
#             x_init=np.random.rand(dim)
#         elif isinstance(x_init, (int, float)):
#             x_init=x_init*np.ones(dim)

#         if algo=="NUTS":
#             fun=Function(self.wrapper_prediction)
#             A=CuqiModel(forward=self.wrapper_prediction,jacobian=fun.compute_jacobian, range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
#             x=Gaussian(mean=mean_prior,cov=cov_prior)
#         else:    
#             A=CuqiModel(forward=self.wrapper_prediction,range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
#             x=Uniform(np.zeros(dim),np.ones(dim)*mean_prior*2)  # prior  # mettere if con diverse prior?
#         y=Gaussian(mean=A(x),cov=proposal_sd)
        
#         if(y_obs is None):
#             y_obs=y(x=x_real).sample()
#         else:
#             y_obs=y_obs+np.random.normal(loc=0., scale=sd_noise,size=y_obs.shape)

#         # posterior=JointDistribution(y,x)(y=y_obs)
        
#         # sampler=MH(posterior,x0=x_init)   # rendere variabile per altri sampler
#         # samples=sampler.sample_adapt(N-burn_in,burn_in)
        
#         # if(diagnostic is True):
#         #     samples.plot_trace()
#         #     mean=samples.mean()
#         #     print(f"Mean values= {mean}")
#         #     ESS=samples.compute_ess()
#         #     print(f"ESS= {ESS}")
#         #     samples.plot_autocorrelation()
#         #     plot_hist(estimates,x_real, self.wrapper_prediction(mean), self.wrapper_prediction(x_real))
        
#         estimates= MCMC_cuqi(y,x,y_obs,N,burn_in,number_chains,diagnostic=diagnostic,algo=algo, adapt=adapt, scale=scale)
#         estimates=np.mean(estimates,axis=1)
#         if diagnostic is True:
#             plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
#         return estimates
    


# #     # def objective(self,par): 
# #     #     #print(self.name)
# #     #     K.clear_session()
# #     #     CVres = kCrossVal(self.n,self.N,self.data_train,self.output_train,par,self.name,self.input_shape)  # ATTENZIONE!! NEL CASO LF, data_train E output_train dovrebbero essere HF (vedere codice Nlf di esempio), correggere
# #     #     return {"loss": CVres, "params": par, "status": STATUS_OK}   

# #     def objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
# #         """
# #         Objective function to minimize during hyperparameter optimization.

# #         Args:
# #             params (Dict[str, Any]): Hyperparameters to evaluate.

# #         Returns:
# #             Dict[str, Any]: The loss value and parameters.
# #         """
# #         K.clear_session()
# #         loss = kCrossVal(self.n, self.N, self.data_train, self.output_train, params, self.name, self.input_shape, self.output_shape)
# #         return {"loss": loss, "params": params, "status": STATUS_OK}
# #     # def HPO(self,data_train,output_train):

# #     #     MAX_EVAL = 5

# #     #     bayes_trials = Trials()
# #     #     opt_list = ["Adam", "Adamax"]
# #     #     kernel_list = ["uniform", "glorot_uniform"]
# #     #     aux_dic = {"opt": opt_list, "kernel_init": kernel_list}
# #     #     space = {
# #     #     "nodes": hp.qloguniform("nodes", np.log(4), np.log(64), 2),
# #     #     "l2weight": hp.loguniform("l2weight", np.log(0.0001), np.log(1)),
# #     #     "lr": hp.loguniform("lr", np.log(0.0001), np.log(0.1)),
# #     #     "kernel_init": hp.choice("kernel_init", kernel_list),
# #     #     "opt": hp.choice("opt", opt_list),
# #     #     }

# #     #     best_params = fmin(fn = self.objective,
# #     #                     space = space,
# #     #                     algo = tpe.suggest,
# #     #                     max_evals = MAX_EVAL,
# #     #                     trials = bayes_trials)
# #     #     transfBestparam(best_params, aux_dic)
# #     #     #self.params=best_params
        
# #     #     print("Best parameters from the HPO:")
# #     #     print(best_params)

# #     #     return best_params   
 
class MultiFidelity(INetwork):
        
    def __init__(self, 
                 names: List[str], 
                 params: Optional[List[dict]] = None, 
                 data_train: Optional[List[np.ndarray]] = None, 
                 output_train: Optional[List[np.ndarray]] = None, 
                 N: Optional[List[int]] = None, 
                 n: Optional[List[int]] = None, 
                 do_HPO: bool = False, 
                 verbose: bool = False):
        """
        Initialize MultiFidelity network.

        Args:
            names (List[str]): Names of the networks.
            params (Optional[List[dict]]): Parameters for each network.
            data_train (Optional[List[np.ndarray]]): Training data for each network.
            output_train (Optional[List[np.ndarray]]): Training outputs for each network.
            N (Optional[List[int]]): Number of epochs for each network.
            n (Optional[List[int]]): Batch sizes for each network.
            do_HPO (bool): Whether to perform hyperparameter optimization.
            verbose (bool): Whether to print verbose output.
        """
        self.names = names
        self.Ns = N
        self.ns = n
        self.data_train = data_train
        self.steps = int((len(names) - 1) / (len(data_train) - 1)) + 1
        self.model_list = []
        self.outputs = np.empty((0, 0))
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
        if len(params) < len(self.names):
            params += [None] * (len(self.names) - len(params))

        count = 1
        for index, name in enumerate(names):
            model = NetworkFactory.build_network(
                name,
                params=params[index],
                data_train=data_train[count - 1],
                output_train=output_train[count - 1],
                N=self.Ns[index],
                n=self.ns[count - 1],
                train=True,
                do_HPO=do_HPO,
                verbose=verbose
            )
            self.model_list.append(model)
            
            if (index + 1) == (self.steps - 1) * (count - 1) + 1:
                count += 1

            for i in range(count - 1, len(data_train)):
                data_train[i] = np.c_[data_train[i], model.prediction(data_train[i])]

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
            data = np.c_[data, self.model_list[i].prediction(data).reshape(-1, 1)]

        pred = self.model_list[position - 1].prediction(data)

        if len(output_test.shape) < len(pred.shape):
            output_test = output_test[:, np.newaxis]

        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2 = 1 - np.sum(np.square(output_test - pred)) / np.sum(np.square(output_test - np.mean(output_test)))
        print(f"R^2: {r2}")

        return test_mse, r2
   
#     def __init__(self,names,params=None,data_train=None,output_train=None,N=None,n=None,do_HPO=False,verbose=False):
#         # names: list of strings, names of the networks (if the discretizations layer that we consider are 2, len(names)==nb_steps)
#         # params: list of dictionaries
#         # N: list of epochs
#         # n: list of batch_sizes
#         # data_train: list of inputs
#         # output_train: list of outputs 
#         self.names=names
#         self.Ns=N
#         self.ns=n
#         self.data_train=data_train
#         self.steps=int((len(names)-1)/(len(data_train)-1))+1
#         self.model_list = []
#         self.outputs = np.empty((0,0))
#         self.transformations=[]
#         self.input_shape=1
#         self.output_shape=1

#         if len(data_train)!=len(output_train) or data_train is None or output_train is None or len(data_train)==0 or len(output_train)==0:
#             raise ValueError('The data are incoherent or insufficient')

#         if data_train is not None and len(data_train[0].shape) > 1:
#             self.input_shape = data_train[0].shape[1] 

#         if output_train is not None and len(output_train[0].shape) > 1:
#             self.output_shape = output_train[0].shape[1] 

#         K.clear_session()
#         if len(params)<len(self.names):
#             diff = len(self.names) - len(params)
#             params += [None] * diff
                        
#         count=1             
#         for index, name in enumerate(names):  # number networks
#             # ATTENTION to the fact that the "number of dataset" is denoted by the number of discretizations considered 
        
#             #print(data_train[count-1].shape)
#             model=NetworkFactory.build_network(name,params=params[index],data_train=data_train[count-1],output_train=output_train[count-1],N=self.Ns[index],n=self.ns[count-1],train=True,do_HPO=do_HPO,verbose=verbose)
#             self.model_list.append(model)
#             # self.outputs.append(model.prediction(self.outputs)) 
#             if (index+1)==(self.steps-1)*(count-1)+1:
#                 count=count+1
#            # print(data_train[count:].shape)
#             data_train[count-1:] = [np.c_[matrix, model.prediction(matrix)] for matrix in data_train[count-1:]]
#             #for i in range(len(data_train)):
#             #    print(data_train[i].shape)

#             # data_train[index+1]=np.c_[data_train_HF,model.prediction(data_train_HF)]     # caso con LF di seguito non è mai capitato finora    LINEA VERA
#             # debugging purpose
#             # indici_piu_vicini = trova_indici_piu_vicini(data_train_LF[:,0], data_train_HF[:,0])
#             # print(indici_piu_vicini)
#             # data_train_HF=np.c_[data_train_HF,output_train_LF[indici_piu_vicini]] 
            
#             #data_train_HF=np.c_[data_train_HF,output_train_HF]     # caso con LF di seguito non è mai capitato finora
    
#     def prediction(self,data_test):
#         self.outputs = data_test

#         if(len(self.outputs.shape)==1):
#             self.outputs=self.outputs.reshape(-1,1)

#         for index, _ in enumerate(self.names):
#             #print(index)
#             #print(self.outputs)
#             self.outputs=np.c_[self.outputs,self.model_list[index].prediction(self.outputs)]
#         #print(self.output_shape)
#         #print("output")
#         #print(self.outputs[:,-self.output_shape:])
#         return self.outputs[:,-self.output_shape:]


#     def performance(self,data_test,output_test,position=None):
        
#         data=copy.copy(data_test)
#         if(position is None):
#             position=len(self.model_list)
#         elif(not isinstance(position, int) or position>len(self.model_list)):
#             raise ValueError('the required NN is not existent')
        
#         # for i in range(position-1):
#         #     data = np.concatenate(
#         #             (data,  self.model_list[i].wrapper_prediction(data).reshape(-1,1)),axis=1
#         #         ) 

#         for i in range(position-1):
#             data = np.concatenate(
#                     (data,  self.model_list[i].prediction(data).reshape(-1,1)),axis=1
#                 ) 

        
#         pred=self.model_list[position-1].prediction(data)#[:,0]
  
#         #print(pred.shape)
#         if len(output_test.shape)<len(pred.shape):
#             output_test=output_test[:,_]
#         test_mse = np.mean(np.square(output_test - pred))
#         print(f"Test MSE: {test_mse}")

#         r2= 1 - np.sum(np.square(output_test - pred)) / np.sum(
#             np.square(output_test - np.mean(output_test))
#         )
#         print(f"R^2: {r2}")
        
#         return (test_mse,r2)
    
    # @compute_time
    # def param_inverse(self, mean_prior, x_data, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, levels=1,diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
    #         # I should insert the 
    #     self.transformations=transformation
    #     self.inputs=x_data

    #     # rivedere la questione dimensioni (x_real!=y_obs)
    #     if x_real is not None:
    #         dim = x_real.shape[0]
    #     elif y_obs is not None:
    #         dim=y_obs.shape[0]
    #     else: 
    #         warning_message = "No observation nor data given"
    #         warnings.warn(warning_message, UserWarning)    

    #     if (N<=burn_in):
    #         warning_message = "number of steps insufficient, smaller or equal than burn-in"
    #         warnings.warn(warning_message, UserWarning) 

    #     if(cov_prior is None):
    #         cov_prior=mean_prior*0.2

    #     if(cov_likelihood is None):
    #         cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
    #     # prior
    #     # if(mean_prior.shape[0]==1):
    #     #     my_prior=beta(1.,1.)
    #     # else:
    #     #     my_prior = multivariate_normal(mean_prior, cov_prior) 
    #     my_prior = multivariate_normal(mean_prior, cov_prior) 
    #     if(y_obs is None):
    #         y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape) # check   
    #     else:
    #         y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape) 
    #         print(y_obs.shape)
    #         #if(len(y_obs.shape)==1):   # if introduced when working on LV 3 outputs, I think the condition was added with the benchmark cases. CHeck
    #         y_obs=y_obs.flatten() #



    #     # likelihood
    #     if levels>1:
    #         my_loglike=[]
    #         my_posterior=[]
    #         if(levels>len(self.model_list)):
    #             warning_message = "number of levels is exceeding the number of models"
    #             warnings.warn(warning_message, UserWarning)
    #         else:
    #             # caso l> 2, ancora da implementare per gestione input reti     
    #             #for i in range(levels): 
    #             #    self.model_list[i].variable_input(x_data)  
    #             #    my_loglike = [my_loglike, tda.GaussianLogLike(y_obs, cov_likelihood)]
    #             #    my_posterior = [my_posterior, tda.Posterior(my_prior, my_loglike[-1], self.model_list[i].wrapper_prediction)]   # attention, you should put ML model at the end
    #             ## SEZIONE TEMPORANEA
    #             for i in range(levels):
    #                 self.model_list[i].variable_input(x_data) 
    #                 my_loglike = [my_loglike, tda.GaussianLogLike(y_obs, cov_likelihood)]
    #             my_posterior = [tda.Posterior(my_prior, my_loglike[-1], self.model_list[0].wrapper_prediction),tda.Posterior(my_prior, my_loglike[-1], self.wrapper_prediction)] 
    #             print(len(my_posterior))
    #             ## SEZIONE TEMPORANEA
        
    #     else:
    #         my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
    #         my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
    #        # print(my_posterior)
    #     print(f"real values are {x_real}")
        
    #     if(rwmh_cov is None):
    #         rwmh_cov = np.eye(len(x_real))
            
    #     estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)

    #     if diagnostic is True:
    #         plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
    #     error=np.abs(estimates-x_real)/np.abs(x_real+1e-10)
    #     return estimates, error
    @compute_time
    def param_inverse(self, mean_prior: np.ndarray, x_data: np.ndarray, cov_prior: Optional[np.ndarray] = None, 
                      cov_noise: float = 0.1, cov_likelihood: Optional[np.ndarray] = None, y_obs: Optional[np.ndarray] = None, 
                      x_real: Optional[np.ndarray] = None, number_chains: int = 1, N: int = 1000, burn_in: int = 500, 
                      levels: int = 1, diagnostic: bool = True, rwmh_cov: Optional[np.ndarray] = None, rmwh_scaling: float = 0.1, 
                      rwmh_adaptive: bool = True, algo: str = "MH", transformation: List[Any] = []) -> Tuple[np.ndarray, np.ndarray]:

        """
        Perform parameter inversion using MCMC sampling.

        Parameters:
        - mean_prior: Mean of the prior distribution.
        - x_data: Input data.
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
            if levels > len(self.model_list):
                warnings.warn("Number of levels is exceeding the number of models", UserWarning)
            else:
                my_loglike = [tda.GaussianLogLike(y_obs, cov_likelihood) for _ in range(levels)]
                my_posterior = [tda.Posterior(my_prior, my_loglike[i], self.model_list[i].wrapper_prediction) for i in range(levels)]
        else:
            my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
            my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)

        if rwmh_cov is None:
            rwmh_cov = np.eye(len(x_real))

        estimates = MCMC(my_posterior, N, burn_in, number_chains, diagnostic, rwmh_cov, rmwh_scaling, rwmh_adaptive, algo, dim)

        if diagnostic:
            plot_hist(estimates, x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        error = np.abs(estimates - x_real) / np.abs(x_real + 1e-10)
        return estimates, error    


    def wrapper_prediction(self,x_test,multi_input=False):
        # ricontrolla, se usata solo per param inv si potra mettere private
    
        #print(x_test)
        if x_test.ndim==1:
            x_test=x_test.reshape(-1,1)
        #print(x_test.shape)
        # if len(x_test)>1:
        #     x_test=x_test[0]

        x_test=x_test.T # VEDI
        if self.transformations:
            x_final = reduce(lambda acc, trasf: np.hstack([acc, trasf(acc)]), self.transformations, x_test)
        else:
            x_final = x_test
            # mettere messaggio di errore quando inputs non riempiton 

        x_final=np.tile(x_final,(self.inputs.shape[0],1))

        
        rep=self.prediction(np.concatenate((self.inputs,x_final),axis=1)).flatten()#.flatten()

        return rep


    @compute_time
    def inverse(self, mean_prior, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, levels=1,diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
        # transformations is a list of lambda functions
        self.transformations=transformation

        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    
        if (N<=burn_in):
            warning_message = "number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning)        
        
        if(cov_prior is None):
            cov_prior=mean_prior*0.2
        if(cov_likelihood is None):
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
        if(mean_prior.shape[0]==1):
            my_prior=beta(1.,1.)
        else:
            my_prior = multivariate_normal(mean_prior, cov_prior) # modo per settare uniforme?
        
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)    # 

        if levels>1:
            my_loglike=[]
            my_posterior=[]
            if(levels>=len(self.model_list)):
                warning_message = "number of levels is exceeding the number of models"
                warnings.warn(warning_message, UserWarning)
            else:    
                for i in range(levels):    
                    my_loglike = [my_loglike, tda.GaussianLogLike(y_obs, cov_likelihood)]
                    my_posterior = [my_posterior, tda.Posterior(my_prior, my_loglike[-1], self.model_list[i].wrapper_prediction)]   # attention, you should put ML model at the end
        else:
            my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
            my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
    
    @compute_time
    def inverse_ML(self, mean_prior, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
        # transformations is a list of lambda functions
        self.transformations=transformation

        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    
        if (N<=burn_in):
            warning_message = "number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning)        
        
        if(cov_prior is None):
            cov_prior=mean_prior*0.2
        if(cov_likelihood is None):
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
        if(mean_prior.shape[0]==1):
            my_prior=beta(1.,1.)
        else:
            my_prior = multivariate_normal(mean_prior, cov_prior) # modo per settare uniforme?
        
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)    # 
        
        my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
        my_posterior=[]

        for i in range(len(self.model_list)):
            my_posterior.append(tda.Posterior(my_prior, my_loglike, self.model_list[i].wrapper_prediction))
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates



    # # CUQIPY
    # @compute_time
    # def inverse_cuqi(self, mean_prior, x_real=None,y_obs=None, N=1000, burn_in=500, cov_prior=0.5, sd_noise=0.1,adapt=False,scale=0.3,proposal_sd=0.3,x_init=None,diagnostic=True, number_chains=1, algo="MH"):

    #     if x_real is not None:
    #         dim = x_real.shape[0]
    #     elif y_obs is not None:
    #         dim=y_obs.shape[0]
    #     else: 
    #         warning_message = "No observation nor data given"
    #         warnings.warn(warning_message, UserWarning)    

    #     if (N<=burn_in):
    #         warning_message = "number of steps insufficient, smaller or equal than burn-in"
    #         warnings.warn(warning_message, UserWarning)        
              
    #     if (x_init is None):
    #         x_init=np.random.rand(dim)
    #     elif isinstance(x_init, (int, float)):
    #         x_init=x_init*np.ones(dim)

    #     if algo=="NUTS":
    #         fun=Function(self.wrapper_prediction)
    #         A=CuqiModel(forward=self.wrapper_prediction,jacobian=fun.compute_jacobian, range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
    #         x=Gaussian(mean=mean_prior,cov=cov_prior)
    #     else:    
    #         A=CuqiModel(forward=self.wrapper_prediction,range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
    #         x=Uniform(np.zeros(dim),np.ones(dim)*mean_prior*2)  # prior  # mettere if con diverse prior?
    #     y=Gaussian(mean=A(x),cov=proposal_sd)
        
    #     if(y_obs is None):
    #         y_obs=y(x=x_real).sample()
    #     else:
    #         y_obs=y_obs+np.random.normal(loc=0., scale=sd_noise,size=y_obs.shape)

        


    #     # posterior=JointDistribution(y,x)(y=y_obs)
        
    #     # sampler=MH(posterior,x0=x_init)   # rendere variabile per altri sampler
    #     # samples=sampler.sample_adapt(N-burn_in,burn_in)
        
    #     # if(diagnostic is True):
    #     #     samples.plot_trace()
    #     #     mean=samples.mean()
    #     #     print(f"Mean values= {mean}")
    #     #     ESS=samples.compute_ess()
    #     #     print(f"ESS= {ESS}")
    #     #     samples.plot_autocorrelation()
    #     #     plot_hist(estimates,x_real, self.wrapper_prediction(mean), self.wrapper_prediction(x_real))
        
    #     estimates= MCMC_cuqi(y,x,y_obs,N,burn_in,number_chains,diagnostic=diagnostic,algo=algo,adapt=adapt,scale=scale)
    #     estimates=np.mean(estimates,axis=1)
    #     if diagnostic is True:
    #         plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
    #     return estimates
    # @compute_time
    # def inverse_cuqi(self, mean_prior, x_real=None, y_obs=None, N=1000, burn_in=500, cov_prior=0.5, sd_noise=0.1, adapt=False, scale=0.3, proposal_sd=0.3, x_init=None, diagnostic=True, number_chains=1, algo="MH"):

    #     if x_real is not None:
    #         dim = x_real.shape[0]
    #     elif y_obs is not None:
    #         dim = y_obs.shape[0]
    #     else:
    #         warning_message = "No observation nor data given"
    #         warnings.warn(warning_message, UserWarning)
    #         return

    #     if N <= burn_in:
    #         warning_message = "Number of steps insufficient, smaller or equal than burn-in"
    #         warnings.warn(warning_message, UserWarning)
    #         return

    #     if x_init is None:
    #         x_init = np.random.rand(dim)
    #     elif isinstance(x_init, (int, float)):
    #         x_init = x_init * np.ones(dim)

    #     if algo == "NUTS":
    #         fun = Function(self.wrapper_prediction)
    #         A = CuqiModel(forward=self.wrapper_prediction, jacobian=fun.compute_jacobian, range_geometry=Continuous1D(dim), domain_geometry=Continuous1D(dim))
    #         x = Gaussian(mean=mean_prior, cov=cov_prior)
    #     else:
    #         A = CuqiModel(forward=self.wrapper_prediction, range_geometry=Continuous1D(dim), domain_geometry=Continuous1D(dim))
    #         x = Uniform(np.zeros(dim), np.ones(dim) * mean_prior * 2)

    #     y = Gaussian(mean=A(x), cov=proposal_sd)

    #     if y_obs is None:
    #         y_obs = y(x=x_real).sample()
    #     else:
    #         y_obs = y_obs + np.random.normal(loc=0., scale=sd_noise, size=y_obs.shape)

    #     estimates = MCMC_cuqi(y, x, y_obs, N, burn_in, number_chains, diagnostic=diagnostic, algo=algo, adapt=adapt, scale=scale)
    #     estimates = np.mean(estimates, axis=1)
    #     if diagnostic:
    #         self.plot_hist(estimates, x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
    #     return estimates         
    def inverse_cuqi(self, mean_prior, x_data,x_real=None, y_obs=None, N=1000, burn_in=500, cov_prior=0.5, sd_noise=0.1,
                     adapt=False, scale=0.3, proposal_sd=0.3, x_init=None, diagnostic=True, number_chains=1, algo="MH",transformation= []):
        self.inputs=x_data
        self.transformations = transformation

        if y_obs is not None:
            dim = y_obs.shape[0]
        else:
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)
            return

        if N <= burn_in:
            warning_message = "Number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning)
            return

        if x_init is None:
            x_init = np.random.rand(dim)
        elif isinstance(x_init, (int, float)):
            x_init = x_init * np.ones(dim)
        m=x_real.shape[0]
        if algo == "NUTS":
            fun = Function(self.wrapper_prediction)
            A = CuqiModel(forward=self.wrapper_prediction, jacobian=fun.compute_jacobian, range_geometry=Continuous1D(dim), domain_geometry=Continuous1D(dim))
            x = Gaussian(mean=mean_prior, cov=cov_prior)
        else:
            A = CuqiModel(forward=self.wrapper_prediction, range_geometry=Continuous1D(dim), domain_geometry=Continuous1D(m))
#A = CuqiModel(forward=self.wrapper_prediction, range_geometry=Continuous1D(dim), domain_geometry=Continuous1D(dim))
            x = Gaussian(mean=mean_prior, cov=cov_prior)# Uniform(np.zeros(m), np.ones(m) * mean_prior * 2)

        y = Gaussian(mean=A(x), cov=proposal_sd)

        if y_obs is None:
            y_obs = y(x=x_real).sample()
        else:
            y_obs = y_obs + np.random.normal(loc=0., scale=sd_noise, size=y_obs.shape)

        estimates = MCMC_cuqi(y, x, y_obs, N, burn_in, number_chains, diagnostic=diagnostic, algo=algo, adapt=adapt, scale=scale)
        estimates = np.mean(estimates, axis=1)
        if diagnostic:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))

        return estimates   
            
    def save(self, discr: str = "_", place: str = "") -> None:
        """
        Save all models in the model list to the specified directory.

        Args:
            discr (str): Discriminator to add to the model filename.
            place (str): Directory where the models should be saved.
        """
        print("Saving ...")
        for index, model in enumerate(self.model_list):
            folder = os.path.join(place, f"{self.names[index]}_model_{discr}.h5")
            save_model(model.model, folder)
    
    def get_output(self) -> np.ndarray:
        """
        Get the last output from the outputs.

        Returns:
            np.ndarray: The last output.
        """
        return self.outputs[-1]
    
class LSTM(INetwork):

    def __init__(self, params=None, data_train=None,output_train=None,N=1000,train=True,do_HPO=False,transformations=[],verbose=False):
        self.params=params
        self.name="LSTM" 
        self.N=N
        self.verbose=verbose
        self.hist=None
        self.data_train=data_train
        self.output_train=output_train
        self.transformations=transformations

        self.input_shape=1
        self.output_shape=1
        self.inputs=None


        if data_train is not None and len(data_train.shape) > 1:
            self.input_shape = data_train.shape[-1]
      
        if output_train is not None and len(output_train.shape) > 1:
            self.output_shape = output_train.shape[-1] 

        if (do_HPO or params is None):
            if (output_train is None or data_train is None):
                warning_message = "Not enough data given!"
                warnings.warn(warning_message, UserWarning)
            self.params=self.HPO(data_train,output_train)
# da includere in train?
        self.sequence_length = int(params['sequence_length'])
        self.sequence_freq = int(params['sequence_freq'])
        self.input_train_seq, self.output_train_seq = sliding_windows(data_train, output_train, self.sequence_length, self.sequence_freq)

        self.model = getModel(self.params,self.input_shape,self.name,self.output_shape)  # dim_input = n_POD + 2, dim_output = n_POD

        if train: 
            #self.hist=self.training(data_train,output_train,epoch=self.N) 
            self.hist=self.training(self.input_train_seq,self.output_train_seq,epoch=self.N) 

            plt.plot(self.model.history.history['loss'][100:], label='Training Loss')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()    
        else:
            name = './models/MF_POD_model'
            self.model = tf.keras.models.load_model(name)       

    @compute_time
    def training(self,x,y,epoch):
        # DUBBIO : va definita variabile hist?

        callback = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=self.params['patience'], restore_best_weights=True)
        tf.keras.utils.set_random_seed(29)
        tf.config.experimental.enable_op_determinism() # for reproducibility
        self.hist = self.model.fit(x, y, epochs=epoch, verbose = self.verbose, callbacks=[callback])

        return self.hist



    def prediction(self,x_test):
        if(self.verbose):
            y_pred=self.model.predict(x_test)#[:,0]             # <-------------  commentato il 6/03
        else:
            with Suppressor():

                y_pred = self.model.predict(x_test)#[:,0]       # <--------------- commentato il 6/03
        return y_pred  

    def performance(self,data_test,output_test,position=None):
        
        data=copy.copy(data_test)
        if(position is None):
            position=len(self.model_list)
        elif(not isinstance(position, int) or position>len(self.model_list)):
            raise ValueError('the required NN is not existent')
        
        for i in range(position-1):
            data = np.concatenate(
                    (data,  self.model_list[i].wrapper_prediction(data).reshape(-1,1)),axis=1
                ) 
        
        pred=self.model_list[position-1].prediction(data)#[:,0]
  
        #print(pred.shape)
        if len(output_test.shape)<len(pred.shape):
            output_test=output_test[:,_]
        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2= 1 - np.sum(np.square(output_test - pred)) / np.sum(
            np.square(output_test - np.mean(output_test))
        )
        print(f"R^2: {r2}")
        
        return (test_mse,r2)
    
    def wrapper_prediction(self,x_test,multi_input=False):
        # ricontrolla, se usata solo per param inv si potra mettere private
    

        if (x_test.ndim==1):
            x_test=x_test.reshape(-1,1)
        #print(x_test.shape)
        x_test=x_test.T # VEDI
        if self.transformations:
            x_final = reduce(lambda acc, trasf: np.hstack([acc, trasf(acc)]), self.transformations, x_test)
        else:
            x_final = x_test
            # mettere messaggio di errore quando inputs non riempiton 

 #       tempi=
        x_final=np.tile(x_final,(self.inputs.shape[1],1))

##########################à
# componente spaziale 
        def u_LF(x, t, re):
            A0 = np.exp(re/8.0)
            return x/(t+1)/(1.0+np.exp(re*(x)/(4*t+4))*((t+1)/A0)**0.5)
        

        #L=1.
        #x = np.linspace(0, L, self.nh)
        
        re_min, re_max = 80, 500
        u_lf = np.zeros((1, self.nt, self.nh))
#         u_hf = np.zeros((1, 110, nh))
        n_POD=16
        for n in range(self.inputs.shape[1]):       
            for i in range(self.x.shape[0]):
                u_lf[0, n, i] = u_LF(self.x[i], self.inputs[0,n,0], denormalization(x_final[0],re_max,re_min))
#                 u_hf[0, n, i] = u_HF(x[i], self.inputs[0,n,0], denormalization(x_final[0],re_max,re_min))

        u_lf_pod=np.reshape(u_lf,(self.nt,self.nh))
        ulf_train=u_lf_pod@self.basis
        ulf_train=np.reshape(ulf_train,(1,self.nt,n_POD))
#         u_hf=np.reshape(u_hf, (110,nh))
#         basis, S = compute_randomized_SVD(u_hf, n_POD, nh, 1)
#         new_inputs=np.reshape(u_lf,(110,nh))@ basis

#         new_inputs=np.reshape(new_inputs,(1,110,16))
# ##########################
#         new_inputs=np.concatenate((self.inputs[0,:, :1], x_final, new_inputs[0,:, :]), axis=1)
        t_grid_lstm, re_grid_lstm = np.meshgrid(self.inputs[0,:,:1], x_final[0])
        new_inputs = np.concatenate((t_grid_lstm[:,:,_], re_grid_lstm[:,:,_], ulf_train), axis = 2)

#        new_inputs=new_inputs.reshape(1,new_inputs.shape[0],new_inputs.shape[1])

# # ATTENZIONE  sarebbe da rivedere. Qui stai adattando a dimensioni e ordine del caso Bruger, sarebbe da generalizzare mettendo parametri sempre in fondo o quanto meno specificare
#         # new_inputs = np.concatenate((self.inputs[:, :1], x_final, self.inputs[:, 1:]), axis=1)
#         # new_inputs=new_inputs.reshape(1,new_inputs.shape[0],new_inputs.shape[1])
        rep=self.prediction(new_inputs).flatten()   

        return rep  

    @compute_time
    def param_inverse(self, basis,datax,mean_prior, nh,nt,x_data, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, levels=1,diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
            # I should insert the 
        self.basis=basis    
        self.transformations=transformation
        self.inputs=x_data
        self.nh=nh
        self.nt=nt
        self.x=datax
        # rivedere la questione dimensioni (x_real!=y_obs)
        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    

        if (N<=burn_in):
            warning_message = "number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning) 

        if(cov_prior is None):
            cov_prior=mean_prior*0.2

        if(cov_likelihood is None):
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
        # prior
        # if(mean_prior.shape[0]==1):
        #     my_prior=beta(1.,1.)
        # else:
        #     my_prior = multivariate_normal(mean_prior, cov_prior) 
        my_prior = multivariate_normal(mean_prior, cov_prior) 
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape) # check   
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape) 
            #print(y_obs.shape)
            #if(len(y_obs.shape)==1):   # if introduced when working on LV 3 outputs, I think the condition was added with the benchmark cases. CHeck
            y_obs=y_obs.flatten() #



        # likelihood
        if levels>1:
            my_loglike=[]
            my_posterior=[]
            if(levels>len(self.model_list)):
                warning_message = "number of levels is exceeding the number of models"
                warnings.warn(warning_message, UserWarning)
            else:
                # caso l> 2, ancora da implementare per gestione input reti     
                #for i in range(levels): 
                #    self.model_list[i].variable_input(x_data)  
                #    my_loglike = [my_loglike, tda.GaussianLogLike(y_obs, cov_likelihood)]
                #    my_posterior = [my_posterior, tda.Posterior(my_prior, my_loglike[-1], self.model_list[i].wrapper_prediction)]   # attention, you should put ML model at the end
                ## SEZIONE TEMPORANEA
                for i in range(levels):
                    self.model_list[i].variable_input(x_data) 
                    my_loglike = [my_loglike, tda.GaussianLogLike(y_obs, cov_likelihood)]
                my_posterior = [tda.Posterior(my_prior, my_loglike[-1], self.model_list[0].wrapper_prediction),tda.Posterior(my_prior, my_loglike[-1], self.wrapper_prediction)] 
                print(len(my_posterior))
                ## SEZIONE TEMPORANEA
        
        else:
            my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
            my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
           # print(my_posterior)
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)

        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        error=np.abs(estimates-x_real)/np.abs(x_real+1e-10)
        return estimates, error
    
    @compute_time
    def inverse_cuqi(self, mean_prior, x_real=None,y_obs=None, N=1000, burn_in=500, cov_prior=0.5, sd_noise=0.1,adapt=False,scale=0.3,proposal_sd=0.3,x_init=None,diagnostic=True, number_chains=1, algo="MH"):
        return
    
    def save(self,place=""):
        print("saving the model ...")
        folder=os.path.join(place,f"{self.name}_model.h5")
        save_model(self.model,folder)
    

class Intermediate(INetwork):
    
    def __init__(self,params=None,data_train=None,output_train=None,N=None,n=None,train=True,do_HPO=False,verbose=False):
        # names: list of strings, names of the networks (if the discretizations layer that we consider are 2, len(names)==nb_steps)
        # params: list of dictionaries
        # N: list of epochs
        # n: list of batch_sizes
        # data_train: list of inputs
        # output_train: list of outputs
        self.params=params
        self.names="Inter" 
        self.N=N
        self.n=n
        self.verbose=verbose
        self.hist=None
        self.data_train=data_train
        self.output_train=output_train
        self.transformations=[]
        
        self.input_shape=1
        self.output_shape=1


        if len(data_train)!=2 or len(output_train)!=2:
            raise ValueError('The data are incoherent or insufficient')
                
        # in this way, the dataset should be only made of 1 matrix (np.array)         
        data_train=np.concatenate((data_train[1],data_train[0]),axis=0)
        output_train=np.concatenate((output_train[1],output_train[0]),axis=0)
        
        # number of inputs of the matrix (columns)  (DUBBIO: non si confonde con lista del caso multifidelity?)
        if data_train is not None and len(data_train.shape) > 1:
            self.input_shape=data_train.shape[1]

        if output_train is not None and len(output_train.shape) > 1:
            self.output_shape=output_train.shape[1]

        if (do_HPO or params is None):
            if (output_train is None or data_train is None):
                warning_message = "Not enough data given!"
                warnings.warn(warning_message, UserWarning)
            self.params=self.HPO(data_train,output_train)
        self.model = getModel(self.params,self.input_shape,self.name,self.output_shape)

        if(train):
        
            self.hist=self.model.fit(data_train,output_train,epochs=self.N,batch_size=self.n,verbose=0) 
            plt.plot(self.model.history.history['loss'][100:], label='Training Loss')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()

    @compute_time
    def training(self,x,y,epoch,batch):
        # DUBBIO : va definita variabile hist?
        self.hist=self.model.fit(x,y,epochs=epoch,batch_size=batch,verbose=0) 
        return self.hist
    
    def prediction(self,x_test):
        if len(x_test)!=2:
            raise ValueError("Not enough data given")
        x_test=np.concatenate((x_test[1],x_test[0]),axis=0)

        if(self.verbose):
            y_pred=self.model.predict(x_test)#[:,0]
        else:
            with Suppressor():
                y_pred = self.model.predict(x_test)
                #print(y_pred)
                #y_pred = self.model.predict(x_test)[:,0]
        return y_pred  
    
    def wrapper_prediction(self,x_test):
           # ricontrolla
        
        if (x_test.ndim==1 and self.data_train.shape[1]>1 and len(self.transformations)==0):   # e.g. Shear cube case
            x_test=x_test.reshape(-1,1).T
        elif (x_test.ndim==1):
            x_test=x_test.reshape(-1,1)
        #print(x_test.shape)
        if self.transformations:
            x_final = reduce(lambda acc, trasf: np.hstack([acc, trasf(acc)]), self.transformations, x_test)
        else:
            x_final = x_test
        return self.prediction(x_final) 
    
    def save(self,discr="_",place=""):
        print("saving the model ...")
        folder=os.path.join(place,f"Inter_model{discr}.h5")
        save_model(self.model,folder)

    def performance(self,data_test,output_test):
        
        data_test=np.concatenate((data_test[1],data_test[0]),axis=0)
        output_test=np.concatenate((output_test[1],output_test[0]),axis=0)
        
        pred=self.prediction(data_test)#[:,0]
        #print(pred.shape)
        if len(output_test.shape)<len(pred.shape):
            output_test=output_test[:,_]
        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2= 1 - np.sum(np.square(output_test - pred)) / np.sum(
            np.square(output_test - np.mean(output_test))
        )
        print(f"R^2: {r2}")
        
        return (test_mse,r2)
    

        
    @compute_time
    def param_inverse(self, mean_prior, x_data, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, levels=1,diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
            # I should insert the 
        self.transformations=transformation
        self.inputs=x_data

        # rivedere la questione dimensioni (x_real!=y_obs)
        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    

        if (N<=burn_in):
            warning_message = "number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning) 

        if(cov_prior is None):
            cov_prior=mean_prior*0.2

        if(cov_likelihood is None):
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
        # prior
        # if(mean_prior.shape[0]==1):
        #     my_prior=beta(1.,1.)
        # else:
        #     my_prior = multivariate_normal(mean_prior, cov_prior) 
        my_prior = multivariate_normal(mean_prior, cov_prior) 
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape) # check   
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape) 
            print(y_obs.shape)
            #if(len(y_obs.shape)==1):   # if introduced when working on LV 3 outputs, I think the condition was added with the benchmark cases. CHeck
            y_obs=y_obs.flatten() #



        # likelihood
        if levels>1:
            my_loglike=[]
            my_posterior=[]
            if(levels>len(self.model_list)):
                warning_message = "number of levels is exceeding the number of models"
                warnings.warn(warning_message, UserWarning)
            else:
                # caso l> 2, ancora da implementare per gestione input reti     
                #for i in range(levels): 
                #    self.model_list[i].variable_input(x_data)  
                #    my_loglike = [my_loglike, tda.GaussianLogLike(y_obs, cov_likelihood)]
                #    my_posterior = [my_posterior, tda.Posterior(my_prior, my_loglike[-1], self.model_list[i].wrapper_prediction)]   # attention, you should put ML model at the end
                ## SEZIONE TEMPORANEA
                for i in range(levels):
                    self.model_list[i].variable_input(x_data) 
                    my_loglike = [my_loglike, tda.GaussianLogLike(y_obs, cov_likelihood)]
                my_posterior = [tda.Posterior(my_prior, my_loglike[-1], self.model_list[0].wrapper_prediction),tda.Posterior(my_prior, my_loglike[-1], self.wrapper_prediction)] 
                print(len(my_posterior))
                ## SEZIONE TEMPORANEA
        
        else:
            my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
            my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
           # print(my_posterior)
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)

        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        error=np.abs(estimates-x_real)/np.abs(x_real+1e-10)
        return estimates, error
    


    def wrapper_prediction(self,x_test,multi_input=False):
        # ricontrolla, se usata solo per param inv si potra mettere private
    

        if (x_test.ndim==1):
            x_test=x_test.reshape(-1,1)
        #print(x_test.shape)
        x_test=x_test.T # VEDI
        if self.transformations:
            x_final = reduce(lambda acc, trasf: np.hstack([acc, trasf(acc)]), self.transformations, x_test)
        else:
            x_final = x_test
            # mettere messaggio di errore quando inputs non riempiton 

        x_final=np.tile(x_final,(self.inputs.shape[0],1))

        
        rep=self.prediction(np.concatenate((self.inputs,x_final),axis=1)).flatten()#.flatten()

        return rep
    
    # TO BE MODIFIED!!
    @compute_time
    def inverse(self, mean_prior, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
        # transformations is a list of lambda functions
        self.transformations=transformation

        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    
        if (N<=burn_in):
            warning_message = "number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning)        
        
        if(cov_prior is None):
            cov_prior=mean_prior*0.2
        if(cov_likelihood is None):
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
        if(mean_prior.shape[0]==1):
            my_prior=beta(1.,1.)
        else:
            my_prior = multivariate_normal(mean_prior, cov_prior) # modo per settare uniforme?
        
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)    # 
        
        my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
        my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
    
    @compute_time
    def inverse_cuqi(self, mean_prior, x_real=None,y_obs=None, N=1000, burn_in=500, cov_prior=0.5,sd_noise=0.1,adapt=False,scale=0.3,proposal_sd=0.3,x_init=None,diagnostic=True, number_chains=1, algo="MH"):

        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    

        if (N<=burn_in):
            warning_message = "number of steps insufficient, smaller or equal than burn-in"
            warnings.warn(warning_message, UserWarning)        
              
        if (x_init is None):
            x_init=np.random.rand(dim)
        elif isinstance(x_init, (int, float)):
            x_init=x_init*np.ones(dim)

        if algo=="NUTS":
            fun=Function(self.wrapper_prediction)
            A=CuqiModel(forward=self.wrapper_prediction,jacobian=fun.compute_jacobian, range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
            x=Gaussian(mean=mean_prior,cov=cov_prior)
        else:    
            A=CuqiModel(forward=self.wrapper_prediction,range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
            x=Uniform(np.zeros(dim),np.ones(dim)*mean_prior*2)  # prior  # mettere if con diverse prior?

        y=Gaussian(mean=A(x),cov=proposal_sd)
        
        if(y_obs is None):
            y_obs=y(x=x_real).sample()
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=sd_noise,size=y_obs.shape)

        


        # posterior=JointDistribution(y,x)(y=y_obs)
        
        # sampler=MH(posterior,x0=x_init)   # rendere variabile per altri sampler
        # samples=sampler.sample_adapt(N-burn_in,burn_in)
        
        # if(diagnostic is True):
        #     samples.plot_trace()
        #     mean=samples.mean()
        #     print(f"Mean values= {mean}")
        #     ESS=samples.compute_ess()
        #     print(f"ESS= {ESS}")
        #     samples.plot_autocorrelation()
        #     plot_hist(estimates,x_real, self.wrapper_prediction(mean), self.wrapper_prediction(x_real))
        
        estimates= MCMC_cuqi(y,x,y_obs,N,burn_in,number_chains,diagnostic=diagnostic,algo=algo, adapt=adapt, scale=scale)
        estimates=np.mean(estimates,axis=1)
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
            
    def objective(self,par): 
        #print(self.name)
        K.clear_session()
        CVres = kCrossVal(self.n,self.N,self.data_train,self.output_train,par,self.name,self.input_shape)  # ATTENZIONE!! NEL CASO LF, data_train E output_train dovrebbero essere HF (vedere codice Nlf di esempio), correggere
        return {"loss": CVres, "params": par, "status": STATUS_OK}   


    def HPO(self,data_train,output_train):

        MAX_EVAL = 5

        bayes_trials = Trials()
        opt_list = ["Adam", "Adamax"]
        kernel_list = ["uniform", "glorot_uniform"]
        aux_dic = {"opt": opt_list, "kernel_init": kernel_list}
        space = {
        "nodes": hp.qloguniform("nodes", np.log(4), np.log(64), 2),
        "l2weight": hp.loguniform("l2weight", np.log(0.0001), np.log(1)),
        "lr": hp.loguniform("lr", np.log(0.0001), np.log(0.1)),
        "kernel_init": hp.choice("kernel_init", kernel_list),
        "opt": hp.choice("opt", opt_list),
        }

        best_params = fmin(fn = self.objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = MAX_EVAL,
                        trials = bayes_trials)
        transfBestparam(best_params, aux_dic)
        #self.params=best_params
        
        print("Best parameters from the HPO:")
        print(best_params)
        return best_params
    


    
# class NetworkFactory:
#     # CHIAMA ANCHE DA MF
#     @staticmethod
#     def build_network(network_type,names=[],params=None,data_train=None,output_train=None,N=1000,n=10,train=True,do_HPO=False,verbose=False):    
        
#         if (network_type=="LF" or network_type=="MF" or network_type=="HF" or network_type=="Hflin" or network_type=="Hfper" ):
#             return Neural_Network(network_type,params,data_train,output_train,N,n,train,do_HPO,verbose)
        
#         if (network_type[:-4].isdigit() and network_type.endswith("step") ):#and len(names)==int(network_type[:-4])):
#             # in this case, data_train, output_train are lists of matrixes containing the data
#             return MultiFidelity(names,params,data_train,output_train,N,n,do_HPO,verbose)
        
#         if (network_type=="Inter"):
#             return  Intermediate(params,data_train,output_train,N,n,train,do_HPO,verbose)
        
#         if (network_type=="LSTM"):
#             return LSTM(params,data_train,output_train, N, train, do_HPO, verbose)

#         print("Invalid Network")
#         return -1
    

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
                      verbose: bool = False) -> Union[INetwork, int]:
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

        Returns:
        - Union[INetwork, int]: The created network object or -1 if the network type is invalid.
        """
        # Create a Neural_Network if network_type matches any specified types
        if network_type in {"LF", "MF", "HF", "Hflin", "Hfper"}:
            return Neural_Network(network_type, params, data_train, output_train, N, n, train, do_HPO, verbose)
        
        # Create a MultiFidelity network if network_type ends with "step" and begins with digits
        if network_type[:-4].isdigit() and network_type.endswith("step"):
            # in this case, data_train, output_train are lists of matrixes containing the data
            return MultiFidelity(names, params, data_train, output_train, N, n, do_HPO, verbose)
        
        # Create an Intermediate network
        if network_type == "Inter":
            return Intermediate(params, data_train, output_train, N, n, train, do_HPO, verbose)
        
        # Create an LSTM network
        if network_type == "LSTM":
            return LSTM(params, data_train, output_train, N, train, do_HPO, verbose)

        # Invalid network type
        print("Invalid Network")
        return -1






def add_noise(noise_std_data, noise_sta_output, data, output):
    output_flag=output
    data_flag=data
    for std1,std2 in zip(noise_std_data,noise_sta_output):
        noise_1 = np.random.normal(0, std1, output.shape)     
        noise_2 = np.random.normal(0, std2, data.shape)
        temp1=output+noise_1
        temp2=data+noise_2
        output_flag=np.concatenate((output_flag,temp1),axis=0)
        #print(data_flag.shape)
        #print(temp2.shape)
        data_flag=np.concatenate((data_flag,temp2))
    return (output_flag,data_flag)






# def kCrossVal(N,Nepo,x,y,params,name,input_shape,p=1):
#     #K.clear_session()
#     model = getModel(params,input_shape,name,y.shape[1]) # check che y.shape[1] sia corretto 
#     Nfold = int(N/p)
#     score = np.zeros([Nfold])
#     cv = KFold(n_splits=Nfold,shuffle=True)
#     splits = cv.split(x)
#     i=0
#     for train_index, test_index in splits:
#       x_train = x[train_index,:]
#       y_train = y[train_index]
#       x_val = x[test_index,:]
#       y_val = y[test_index]
#       #model = getModel(params,name)
#       model.fit(x_train,y_train,epochs=Nepo,batch_size=N-p,verbose=0)
#       score[i] = np.mean(np.square(y_val - model.predict(x_val)[:,0]))
#       i = i+1
#     return np.mean(score)


def kCrossVal(N: int, Nepo: int, x: np.ndarray, y: np.ndarray, params: Dict[str, Any], name: str, input_shape: int, output_shape: int, p: int = 1) -> float:
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
        p (int): Number of folds.

    Returns:
        float: Average cross-validation loss.
    """
    model = getModel(params, input_shape, name, output_shape)
    Nfold = int(N / p)
    scores = []

    kf = KFold(n_splits=Nfold, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[test_index], y[test_index]

        model.fit(x_train, y_train, epochs=Nepo, batch_size=N - p, verbose=0)
        predictions = model.predict(x_val)
        score = np.mean(np.square(y_val - predictions[:, 0]))
        scores.append(score)

    return np.mean(scores)


def kCrossValSingle(N, Nepo, x, y, params, name,input_shape):
    score = np.zeros([N])
    cv = KFold(N)
    splits = cv.split(x)
    model = getModel(params,input_shape, name,y.shape[1]) # check che y.shape[1] sia corretto 
    i = 0
    for train_index, test_index in splits:
        x_train = x[train_index]
        y_train = y[train_index]
        x_val = x[test_index]
        y_val = y[test_index]
        model.fit(x_train, y_train, epochs=Nepo, batch_size=N - 1, verbose=0)
        score[i] = np.square(y_val - model.predict(x_val))
        i = i + 1
    return np.mean(score)


def kCrossValGP(Nhf,Nlf,Nepo,xhf,yhf,xlf,ylf,params,name,input_shape,p=1):
    Nfolds = int(Nhf/p)
    score = np.zeros([Nfolds])
    cv = KFold(n_splits = Nfolds, shuffle = True)
    splits = cv.split(xhf)
    i=0
    N = Nhf + Nlf

    model = getModel(params,input_shape,name,yhf.shape[1]) # check che y.shape[1] sia corretto 
    for train_index, test_index in splits:
      xhf_train = xhf[train_index,:]
      yhf_train = yhf[train_index]
      xhf_val = xhf[test_index,:]
      yhf_val = yhf[test_index]
      x_train = np.concatenate((xhf_train,xlf))
      yhf_train = np.concatenate((yhf_train,np.full(Nlf,-10)))
      ylf_train = np.concatenate((np.full(Nhf-p,-10),ylf))
      model.fit(x_train,[yhf_train,ylf_train],epochs=int(params['epochs'])*Nepo,batch_size=N-p,verbose=0)
      score[i] = np.mean(np.square(yhf_val - model.predict(xhf_val)[0][:,0]))
      i = i+1
    return np.mean(score)



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

# def transfBestparam(bestparam, dic):
#     # trasforma kernel e opt da parametri numerici (indicatori booleani) a valore stringa
#     for key in bestparam:
#         if key in ["kernel_init", "opt"]:
#             bestparam[key] = dic[key][bestparam[key]]
#     return


def getOpti(name,lr):
  if name == 'Adam':
      return Adam(learning_rate=lr,amsgrad=True)
  elif name == 'Nadam':
      return Nadam(learning_rate=lr)
  elif name == 'Adamax':
      return Adamax(learning_rate=lr)
  elif name == 'RMSprop':
      return RMSprop(learning_rate=lr)
  elif name == 'standardadam':
      return 'adam'
  




  class FourierLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FourierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel_sin = self.add_weight(name='kernel_sin',
                                          shape=(self.output_dim,),  
                                          initializer='glorot_uniform',
                                          trainable=True)
        self.kernel_cos = self.add_weight(name='kernel_cos',
                                          shape=(self.output_dim,),   
                                          initializer='glorot_uniform',
                                          trainable=True)
        super(FourierLayer, self).build(input_shape)

    def call(self, x):
        result = tf.sin(tf.multiply(x, self.kernel_sin)) + tf.cos(tf.multiply(x, self.kernel_cos))
        return result

    def compute_output_shape(self, input_shape):
        return input_shape