import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
import tensorflow as tf
import arviz
import logging

from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
import sys
import os
import warnings
import ray
from cuqi.distribution import Uniform, Gaussian,JointDistribution, Beta
from cuqi.sampler import MH,NUTS, Gibbs, CWMH, pCN
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Discrete
#from cuqi.diagnostics import Geweke
import tinyDA as tda
from scipy.stats import multivariate_normal,beta
import arviz as az
import time 

from itertools import product
from typing import Any, Dict, Tuple, Callable, List

    
def custom_activation(x):
    return x + K.square(K.sin(x))

def  normalization(x):
    return (x - np.min(x)) / (
    np.max(x) - np.min(x)
)


def MCMC(my_posterior,N, burnin, n=1, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1, period=100, t0=0, rwmh_adaptive=False,algo="MH",dim=0):
    # my_posterior: list which should contain the right order of the posteriors
    MAP = tda.get_MAP(my_posterior) if dim != 1 else None

    if algo == "MH":
        my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive) # gamma= adaptivity coefficient
        my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
    elif algo=="AM":
        # adaptive metropolis
        my_proposal=tda.AdaptiveMetropolis(C0=rwmh_cov, adaptive=rwmh_adaptive,period=period, t0=t0)   # sd am scaling parameter, gamma
        my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
    elif algo=="CN":
        # preconditioned Crank Nicolson
        my_proposal=tda.CrankNicolson(scaling=rmwh_scaling, adaptive=rwmh_adaptive,period=period)
        my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
    elif algo=="DREAMZ":
        my_proposal=tda.DREAMZ(M0=10*dim,adaptive=rwmh_adaptive,period=period)
        my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
    elif algo=="MLDA":
        # proposal parameter: proposal for the coarsest
        my_proposal=tda.MLDA(posteriors=my_posterior, subsampling_rates=[5,5],adaptive_error_model='state-independent',initial_parameters=MAP,store_coarse_chain=True,proposal=tda.AdaptiveMetropolis(C0=rwmh_cov))
        my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n,initial_parameters=MAP,subsampling_rate=[5,5]) #,subsampling_rate=5, adaptive_error_model='state-independent'
    else: 
        raise ValueError("Unknown algorithm %s"%algo)
    

    idata = tda.to_inference_data(my_chains, burnin=burnin)
    estimates=np.array(az.summary(idata)['mean'])
    print(f"estimated values are {estimates}")
    if (diagnostic is True):
        print(az.summary(idata))
        az.plot_trace(idata)
        print("Autocorrelation...")
        az.plot_autocorr(idata)
        #az.plot_violin(idata)
        
    
    return estimates



# def MCMC(
#     my_posterior: Any, 
#     N: int, 
#     burnin: int, 
#     n: int = 1, 
#     diagnostic: bool = True, 
#     rwmh_cov: Optional[np.ndarray] = None, 
#     rmwh_scaling: float = 0.1, 
#     period: int = 100, 
#     t0: int = 0, 
#     rwmh_adaptive: bool = False, 
#     algo: str = "MH", 
#     dim: int = 0
# ) -> np.ndarray:
#     """
#     Perform MCMC sampling using the specified algorithm.

#     Args:
#         my_posterior (Any): The posterior distribution to sample from.
#         N (int): Number of MCMC iterations.
#         burnin (int): Number of burn-in iterations.
#         n (int, optional): Number of chains to run. Defaults to 1.
#         diagnostic (bool, optional): Whether to print and plot diagnostics. Defaults to True.
#         rwmh_cov (Optional[np.ndarray], optional): Covariance matrix for the proposal distribution. Defaults to None.
#         rmwh_scaling (float, optional): Scaling factor for the proposal distribution. Defaults to 0.1.
#         period (int, optional): Adaptation period for certain algorithms. Defaults to 100.
#         t0 (int, optional): Tuning parameter for adaptive algorithms. Defaults to 0.
#         rwmh_adaptive (bool, optional): Whether to use adaptive algorithms. Defaults to False.
#         algo (str, optional): Algorithm to use for MCMC sampling. Defaults to "MH".
#         dim (int, optional): Dimensionality of the problem, used for certain algorithms. Defaults to 0.

#     Returns:
#         np.ndarray: Estimated values from the MCMC sampling.
#     """
#     # Get the Maximum A Posteriori (MAP) estimate
#     MAP = tda.get_MAP(my_posterior) if dim != 1 else None

#     # Select the MCMC algorithm
#     if algo == "MH":
#         my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive)
#     elif algo == "AM":
#         my_proposal = tda.AdaptiveMetropolis(C0=rwmh_cov, adaptive=rwmh_adaptive, period=period, t0=t0)
#     elif algo == "CN":
#         my_proposal = tda.CrankNicolson(scaling=rmwh_scaling, adaptive=rwmh_adaptive, period=period)
#     elif algo == "DREAMZ":
#         my_proposal = tda.DREAMZ(M0=10*dim, adaptive=rwmh_adaptive, period=period)
#     else:
#         raise ValueError(f"Unknown algorithm {algo}")

#     # Sample using the selected proposal and posterior
#     my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
#     idata = tda.to_inference_data(my_chains, burnin=burnin)
#     estimates = np.array(az.summary(idata)['mean'])

#     # Print and plot diagnostics if enabled
#     print(f"Estimated values are {estimates}")
#     if diagnostic:
#         print(az.summary(idata))
#         az.plot_trace(idata)
#         print("Autocorrelation...")
#         az.plot_autocorr(idata)

#     return estimates




# def MCMC_cuqi(y,x,observation, N, burn_in, n=1, diagnostic=True,algo="MH",adapt=False, scale=0.3):
    
#     x_init=np.random.rand(observation.shape[0],n)
        
#     estimates=np.empty((observation.shape[0],0))
#     ESSs=np.empty((observation.shape[0],0))
#     #Geweke=np.empty((x_init.shape[0],0))
#     #Rhat=np.empty((observation.shape[0],0))

#     chains=np.empty((0,observation.shape[0],N-burn_in))
#     #chains=np.empty((observation.shape[0],N-burn_in))
#     post=np.empty((observation.shape[0],0))
#     posterior=JointDistribution(y,x)(y=observation)

#     for i in range(n):
        
#         if algo=="NUTS":
#             # Hamiltonian Monte Carlo
#             sampler=NUTS(posterior,x0=x_init[:,i])
#         elif algo=="MH":
#             # Metropolis Hastings
#             if adapt is False:
#                 sampler=MH(posterior,x0=x_init[:,i],scale=scale)
#             else:
#                 sampler=MH(posterior,x0=x_init[:,i])

#         # elif algo=="Gibbs":
#         #     # GIbbs Sampler
#         #     sampler=
#         # elif algo=="CWMH":
#         #     sampler=
#         elif algo=="pCN":
#             # preconditioned Crank Nicholson
#             sampler=pCN(posterior,x0=x_init[:,i])
#         else:
#             raise ValueError("Unknown algorithm %s"%algo)
#         if adapt is True:
#             samples=sampler.sample_adapt(N-burn_in,burn_in)
#         else:
#             samples=sampler.sample(N-burn_in,burn_in)

#         estimates=np.column_stack((estimates,samples.mean()[:, np.newaxis]))
#        # ESSs=np.column_stack((ESSs,samples.compute_ess()[:, np.newaxis]))
#   #      Rhat=np.column_stack((Rhat,samples.compute_rhat()[:, np.newaxis]))
#         #print(Geweke)
#         #Geweke=np.column_stack((Geweke,samples.diagnostics()[:, np.newaxis][0]))
#                 # chains=np.concatenate(chains, samplesMH_LF.samples)
#         #printsamples.shape)
#         chains = np.concatenate((chains, np.expand_dims(samples.samples, axis=0)), axis=0)
#         post=np.concatenate((post,samples.samples),axis=1)


#         print(                f"********************  # Mean values = {estimates.mean(axis=1)}  ********************"
#                 )
#     print(chains.shape)
#     for l in range(chains.shape[1]):
#         plt.figure(figsize=(10, 4))

#         for i in range(chains.shape[0]):
#             plt.plot(chains[i, l,:])

#         plt.xlabel('Sample')
#         plt.ylabel('Value')
#         plt.title(f'Trace Plot variable {l}')
#         plt.legend()
#         plt.show()
        
#     if(diagnostic is True):
#         if(n==1):
#             samples.plot_trace()
#             samples.plot_autocorrelation()
#         else:
#             num_bins=20
#             plt.figure()
#             for num in range(post.shape[0]):        
                
#                 bin_edges = np.linspace(np.min(post[num,:]), np.max(post[num,:]), num_bins + 1)
#                 hist, _ = np.histogram(post, bins=bin_edges)
#                 hist=hist/post.shape[1]
#                 bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#                 print(hist)
#                 plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', label=f'Var {num + 1}')

#                 plt.xlabel('Value')
#                 plt.ylabel('Probability')
#                 plt.title('Distribution')
#                 plt.legend()
#                 plt.show()
                
#             autocov=arviz.autocov(chains[:,0,:])
#             ess=arviz.ess(chains[:,0,:])
#             print(                f"********************  # ESS values = {ess}  ********************"
#                 )


#             plt.figure()
#             plt.plot(autocov[0,:])
#             plt.title('Autocovariance first chain')
#             plt.xlabel('Lag')
#             plt.ylabel('Autocovariance')
#             plt.legend()
#             plt.show()
    
#     return estimates

def MCMC_cuqi(
    y: Any, 
    x: Any, 
    observation: np.ndarray, 
    N: int, 
    burn_in: int, 
    n: int = 1, 
    diagnostic: bool = True, 
    algo: str = "MH", 
    adapt: bool = False, 
    scale: float = 0.3,
    parallel: bool = False
) -> np.ndarray:
    """
    Perform MCMC sampling using CUQI library.

    Args:
        y (Any): Dependent variable.
        x (Any): Independent variable.
        observation (np.ndarray): Observed data.
        N (int): Number of samples to draw.
        burn_in (int): Number of burn-in samples to discard.
        n (int, optional): Number of chains. Defaults to 1.
        diagnostic (bool, optional): Whether to plot diagnostic plots. Defaults to True.
        algo (str, optional): Sampling algorithm to use. Defaults to "MH".
        adapt (bool, optional): Whether to use adaptive sampling. Defaults to False.
        scale (float, optional): Scaling factor for MH algorithm. Defaults to 0.3.
        parallel (bool, optional): Whether to run chains in parallel. Defaults to False.

    Returns:
        np.ndarray: Array of estimated parameter means.
    """
    dim = observation.shape[0]
    x_init = np.random.rand(dim)
    estimates = np.empty((1, 0))
    chains = np.empty((0, 1, N - burn_in))
    post = np.empty((1, 0))
    posterior = JointDistribution(x,y)(y=observation)

    if parallel:
        # Parallel execution using Ray
        # logging.getLogger().setLevel(logging.WARNING)
        # tf.get_logger().setLevel('ERROR')
        # # ray.init(num_cpus=4, logging_level=logging.WARNING)
        
        # ray.init(logging_level=logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        tf.get_logger().setLevel('ERROR')    
        ray.init(ignore_reinit_error=True, logging_level=logging.WARNING, log_to_driver=False)


        futures = [chain_creation_parallel.remote(N, burn_in, diagnostic, algo, adapt, scale, posterior, x_init) for _ in range(n)]
        results = ray.get(futures)
        ray.shutdown()
    else:
        # Sequential execution
        results = [chain_creation(N, burn_in, diagnostic, algo, adapt, scale, posterior, x_init) for _ in range(n)]

    for result in results:
        estimates = np.column_stack((estimates, result[0]))
        chains = np.concatenate((chains, result[1]), axis=0)
        post = np.concatenate((post, result[2]), axis=1)

    # Plot trace plots for each parameter
    for l in range(chains.shape[1]):
        plt.figure(figsize=(10, 4))
        for i in range(chains.shape[0]):
            plt.plot(chains[i, l, :])
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title(f'Trace Plot for variable {l}')
        plt.legend([f'Chain {i+1}' for i in range(chains.shape[0])])
        plt.show()

    # Diagnostic plots
    if diagnostic:
        num_bins = 20
        plt.figure()
        for num in range(post.shape[0]):
            bin_edges = np.linspace(np.min(post[num, :]), np.max(post[num, :]), num_bins + 1)
            hist, _ = np.histogram(post[num, :], bins=bin_edges)
            hist = hist / post.shape[1]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', label=f'Var {num + 1}')
            plt.xlabel('Value')
            plt.ylabel('Probability')
            plt.title('Distribution')
            plt.legend()
            plt.show()

        autocov = az.autocov(chains[:, 0, :])
        ess = az.ess(chains[:, 0, :])
        print(f"ESS values = {ess}")
        plt.figure()
        plt.plot(autocov[0, :])
        plt.title('Autocovariance first chain')
        plt.xlabel('Lag')
        plt.ylabel('Autocovariance')
        plt.legend()
        plt.show()

    return estimates

@ray.remote
def chain_creation_parallel(
    N: int, 
    burn_in: int, 
    diagnostic: bool, 
    algo: str, 
    adapt: bool, 
    scale: float,
    posterior: Any, 
    x_init: np.ndarray
) -> tuple:
    """
    Wrapper function to parallelize chain creation using Ray.

    Args:
        N (int): Number of samples.
        burn_in (int): Number of burn-in samples.
        diagnostic (bool): Whether to plot diagnostics.
        algo (str): MCMC algorithm to use.
        adapt (bool): Whether to use adaptive sampling.
        scale (float): Scaling factor for MH algorithm.
        posterior (Any): Posterior distribution to sample from.
        x_init (np.ndarray): Initial parameter values.

    Returns:
        tuple: Estimates, chains, and posterior samples.
    """

    return chain_creation(N, burn_in, diagnostic, algo, adapt, scale, posterior, x_init)


def preprocess_input(x, mean):
    """
    Preprocess the input to ensure shapes are compatible for broadcasting.
    
    Parameters:
    x (numpy.ndarray): The input data.
    mean (numpy.ndarray): The mean data or other parameter.

    Returns:
    tuple: Processed x and mean.
    """
    if mean.ndim == 1 and mean.size != x.size:
        mean_reshaped = mean.reshape(1, -1)
    else:
        mean_reshaped = mean  # No reshaping needed if already compatible
    return x, mean_reshaped

def chain_creation(
    N: int, 
    burn_in: int, 
    diagnostic: bool, 
    algo: str, 
    adapt: bool, 
    scale: float,
    posterior: Any, 
    x_init: np.ndarray
) -> tuple:
    """
    Create a chain using specified MCMC algorithm.

    Args:
        N (int): Number of samples.
        burn_in (int): Number of burn-in samples.
        diagnostic (bool): Whether to plot diagnostics.
        algo (str): MCMC algorithm to use.
        adapt (bool): Whether to use adaptive sampling.
        scale (float): Scaling factor for MH algorithm.
        posterior (Any): Posterior distribution to sample from.
        x_init (np.ndarray): Initial parameter values.

    Returns:
        tuple: Estimates, chains, and posterior samples.
    """
    # Select the appropriate sampler
    if algo == "NUTS":
        sampler = NUTS(posterior, x0=x_init)
    elif algo == "MH":
        sampler = MH(posterior, scale=scale) if not adapt else MH(posterior)
    elif algo == "pCN":
        sampler = pCN(posterior, x0=x_init)
    else:
        raise ValueError(f"Unknown algorithm {algo}")

    # Perform sampling
    samples = sampler.sample_adapt(N - burn_in, burn_in) if adapt else sampler.sample(N - burn_in, burn_in)

    # Compute estimates and store chains
    estimates = samples.mean()[:, np.newaxis]
    chains = np.expand_dims(samples.samples, axis=0)
    post = samples.samples

    print(f"Mean values: {estimates.mean(axis=1)}")

    return (estimates, chains, post)

# Plot Histogram
def plot_hist(estimates: np.ndarray, real_x: np.ndarray, output1: np.ndarray, output2: np.ndarray) -> None:
    """
    Plot histogram comparing estimated and real values.

    Args:
        estimates (np.ndarray): Estimated values from the model.
        real_x (np.ndarray): True values to compare against.
        output1 (np.ndarray): First set of output values for comparison.
        output2 (np.ndarray): Second set of output values for comparison.

    Returns:
        None
    """
    # Calculate the absolute differences
    diff_output = np.abs(output1 - output2)
    diff_value = np.abs(real_x - estimates)
    print(f"The difference between estimated values {diff_value}\n")

    # Stack real and estimated values for plotting
    values = np.vstack((real_x, estimates))
    categories = np.arange(1, values.shape[1] + 1)
    bar_width = 0.35
    bar_positions = [categories - bar_width / 2 + i * bar_width for i in range(values.shape[0])]

    # Create the plot
    plt.figure()
    for i in range(values.shape[0]):
        plt.bar(bar_positions[i], values[i, :], width=bar_width)

    # Customize the plot
    plt.ylabel('Value')
    plt.title('Comparison of Real and Estimated Values')
    plt.xticks(categories)
    plt.legend(["Real value", "Estimate"])
    plt.show()

# def plot_hist(estimates, real_x, output1,output2):   
#     values2=estimates
#     values1=real_x
#     diff_output=np.abs(output1-output2)
#     diff_value=np.abs(values1-values2)
#     print(f"the difference between estimated values {diff_value}\n the difference between outputs of two models are {diff_output}")
#     values = np.vstack((values1, values2))

#     # Creare categorie in base alla lunghezza di values
#     categories = np.arange(1, values.shape[1] + 1)

#     # Larghezza delle colonne
#     bar_width = 0.35

#     # Posizioni delle colonne
#     bar_positions = [categories - bar_width/2 + i*bar_width for i in range(values.shape[0])]
#     plt.figure()
#     # Creazione del plot
#     for i in range(values.shape[0]):
#         plt.bar(bar_positions[i], values[i, :], width=bar_width)


#     plt.ylabel('Value')
#     plt.title('Input')
#     plt.xticks(categories)
#     plt.legend(["Real value", "Estimate"])


  
#     plt.show()
    
#     # values2 = output2
#     # values1=output1

#     # values = np.vstack((values1, values2))

#     # # Creazione del plot
#     # for i in range(values.shape[0]):
#     #     plt.bar(bar_positions[i], values[i, :], width=bar_width)


#     # plt.ylabel('Value')
#     # plt.title('Output')
#     # plt.xticks(categories)
#     # plt.legend(["Real value", "Estimate"])
#     # # Mostra il plot
#     # plt.show()
#     return
    
    
def custom_loss(y_pred,y_true):
    goodind = K.not_equal(y_pred,-10)
    #goodind = tf.math.logical_not(tf.math.is_nan(y_pred))
    y_pred_loss = tf.boolean_mask(y_pred,goodind)
    y_pred_true = tf.boolean_mask(y_true,goodind)
    return K.mean(K.square(y_pred_loss - y_pred_true))

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

def getModel(params,num_inputs,name, num_outputs):
    if(name == 'HF'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        #hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        output = Dense(num_outputs,activation='linear',name='HF')(hidden1)     
    elif (name == 'LF'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden3)
        output = Dense(num_outputs,activation='linear',name='LF')(hidden4)
        
    elif (name == 'Single'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(inputs)
        hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden1)
        hidden3 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden2)
        hidden4 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden3)
        output = Dense(num_outputs,activation='linear',name='Single')(hidden2)        
        
    elif (name == 'Hflin'):
        inputs = Input(shape=(num_inputs,))
        hiddenlin = Dense(64,activation='linear',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs,activation='linear',name='HFlin')(hiddenlin)
        
    elif(name == 'Hfper'):
        inputs = Input(shape=(num_inputs,))
        hiddenlin = Dense(64,activation=custom_activation,kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(num_outputs,activation='linear',name='HFper')(hiddenlin)    
        
    elif (name == 'GP'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden3)
        GPlayer = Dense(2,activation='linear',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
        outputLF = Dense(num_outputs,activation='linear',name='LF')(GPlayer)
        outputHF = Dense(num_outputs,activation='linear',name='HF')(GPlayer)   
        output = [outputHF,outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'],params['lr'])
        model.compile(loss=custom_loss,loss_weights=[params['alpha'],1-params['alpha']],optimizer=opti)    
        return model
    
    elif (name == 'Inter'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden1)
        outputLF = Dense(num_outputs,activation='linear',name='LF')(hidden2)
        outputadd = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
        merge = concatenate([outputLF,outputadd])
        hidden3 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(merge)
        hidden4 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden3)
        #hidden5 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
        #hidden6 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden5)
  
        #lincorr = Dense(int(params['nodes']),activation='linear',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(outputLF)
        #merge2 = concatenate([hidden3,lincorr])
        outputHF = Dense(4,activation='linear',name='HF')(hidden4)
        output = [outputHF,outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'],params['lr'])
        model.compile(loss=custom_loss,loss_weights=[params['alpha'],1-params['alpha']],optimizer=opti)
        return model
 
        
    model = Model(inputs=inputs,  outputs=output)
    opti = getOpti(params['opt'],params['lr'])
    model.compile(loss='mse',optimizer=opti,metrics=['mse'])
    return model




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

def run_simulation(
    datahf: np.ndarray, mean_prior: np.ndarray, cov_prior: np.ndarray, Yhf: np.ndarray, 
    sigma_noise: List[float], n_data: List[int], parameters: np.ndarray, sigma: np.ndarray, 
    rwmh_scaling: np.ndarray, rwmh_cov: np.ndarray, rwmh_adaptive: bool, 
    iterations: int, burnin: int, n_chains: int, final_model, algo: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a simulation to estimate parameters and calculate errors.
    
    Parameters:
    - datahf: 2D numpy array containing data.
    - mean_prior: 1D numpy array for the mean of the prior.
    - cov_prior: 2D numpy array for the covariance of the prior.
    - Yhf: 1D numpy array of observed values.
    - sigma_noise: List of noise levels.
    - n_data: List of number of data points.
    - parameters: 1D numpy array of parameters.
    - sigma: 1D numpy array of standard deviations for the likelihood.
    - rwmh_scaling: 1D numpy array of scaling factors for the RWMH algorithm.
    - rwmh_cov: 2D numpy array for the RWMH covariance.
    - rwmh_adaptive: Boolean indicating if RWMH is adaptive.
    - iterations: Integer for the number of iterations.
    - burnin: Integer for the burn-in period.
    - n_chains: Integer for the number of chains.
    - final_model: The model object with the param_inverse method.
    - algo: String indicating the algorithm to use.
    
    Returns:
    - best_estimate: The best parameter estimate.
    - best_error: The error corresponding to the best estimate.
    """
    
    # Initialize error and estimate arrays
    error_shape = (len(sigma_noise), len(n_data), len(sigma), len(rwmh_scaling))
    error = np.zeros(error_shape)
    estimates = np.zeros(error_shape)
    
    # Iterate over all combinations of parameters using itertools.product
    for (i, noise), (k, n), (t, s), (j, r) in product(enumerate(sigma_noise), enumerate(n_data), enumerate(sigma), enumerate(rwmh_scaling)):
        t_eval = np.linspace(0., 5., n).reshape(-1, 1)  # Generate evaluation times
        nearest_x, y_obs = process_data(datahf, parameters, t_eval, Yhf)  # Get nearest x values and observations
        cov_likelihood = calculate_cov_likelihood(s, t_eval)  # Compute the covariance for the likelihood

        # Perform parameter estimation and calculate error
        estimates[i, k, t, j], error[i, k, t, j] = final_model.param_inverse(
            mean_prior, t_eval, cov_prior=cov_prior, rmwh_scaling=r, 
            cov_noise=noise, cov_likelihood=cov_likelihood, y_obs=y_obs, 
            x_real=parameters, number_chains=n_chains, N=iterations, 
            burn_in=burnin, diagnostic=True, rwmh_cov=rwmh_cov, 
            rwmh_adaptive=rwmh_adaptive, algo=algo
        )
    
    # Identify the index of the minimum error
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, "
          f"rwmh_scaling={rwmh_scaling[smallest_index[3]]}")

    return best_estimate, best_error

def run_simulation_cuqi(
    xhf: np.ndarray,
    yhf:np.ndarray,
    mean_prior: np.ndarray,
    x_real: np.ndarray,
    N: int,
    burn_in: int,
    cov_prior: np.ndarray,
    sd_noise: list,
    adapt: bool,
    proposal_sd: list,
    number_chains: int,
    algo: str,
    x_data: np.ndarray,
    n_data: list,
    final_model,
    parallel: bool
) -> tuple:
    """
    Run a CUQI simulation to estimate parameters and compute error.

    Args:
        data (dict): Dictionary containing high-fidelity data (keys: "xhf" and "Yhf").
        mean_prior (np.ndarray): Prior mean vector.
        x_real (np.ndarray): Real x values.
        N (int): Number of samples.
        burn_in (int): Number of burn-in samples.
        cov_prior (np.ndarray): Prior covariance matrix.
        sd_noise (list): List of noise standard deviations to evaluate.
        adapt (bool): Whether to use adaptation in the algorithm.
        proposal_sd (list): List of proposal standard deviations to evaluate.
        number_chains (int): Number of MCMC chains.
        algo (str): Algorithm to use for MCMC.
        x_data (np.ndarray): Initial evaluation times.
        n_data (list): List of data sizes to evaluate.
        final_model: Final model object with inverse_cuqi method.
        parallel (bool): Whether to run MCMC chains in parallel.

    Returns:
        tuple: Best estimate and best error found during the simulation.
    """

    # Initialize estimates and error arrays
    estimates = np.zeros((len(sd_noise), len(n_data), len(proposal_sd)))
    error = np.zeros((len(sd_noise), len(n_data), len(proposal_sd)))

    # Iterate over noise levels
    for i, noise in enumerate(sd_noise):
        # Iterate over number of data points
        for k, n in enumerate(n_data):
            # Generate evaluation times
            x_data = np.linspace(0., 5., n).reshape(-1, 1)
            nearest_x, y_obs = process_data(xhf, x_real, x_data, yhf)
            
            # Iterate over proposal standard deviations
            for t, s in enumerate(proposal_sd):

                # Perform parameter estimation and calculate error
                estimates[i, k, t], error[i, k, t] = final_model.inverse_cuqi(
                    mean_prior=mean_prior,
                    x_real=x_real,
                    y_obs=y_obs,
                    N=N,
                    burn_in=burn_in,
                    cov_prior=cov_prior,
                    sd_noise=noise,  
                    adapt=adapt,
                    scale=s,
                    proposal_sd=s,
                    number_chains=number_chains,
                    algo=algo,
                    x_data=x_data,
                    parallel=parallel
                )
                
    # Find the smallest error and corresponding indices
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sd_noise={sd_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, proposal_standard_deviation={proposal_sd[smallest_index[2]]}")

    return best_estimate, best_error
