# import tensorflow.keras.backend as K
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Input, concatenate
# from tensorflow.keras.optimizers import Adam,Nadam,Adamax
# import tensorflow as tf
# import arviz

# from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
# from hyperopt.pyll.stochastic import sample
# from hyperopt.pyll.base import scope
# from sklearn.model_selection import KFold
# import numpy as np
# from matplotlib import pyplot as plt
# from time import perf_counter
# import sys
# import os
# import warnings

# from cuqi.distribution import Uniform, Gaussian,JointDistribution, Beta
# from cuqi.sampler import MH,NUTS, Gibbs, CWMH, pCN
# from cuqi.model import Model as CuqiModel
# from cuqi.geometry import Continuous1D, Discrete
# #from cuqi.diagnostics import Geweke
# import tinyDA as tda
# from scipy.stats import multivariate_normal,beta
# import arviz as az
# import time 
# from numba import jit


# from typing import  Optional, Any
# def custom_activation(x):
#     return x + K.square(K.sin(x))


# # def MCMC(my_posterior,N, burnin, n=1, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1, period=100, t0=0, rwmh_adaptive=False,algo="MH",dim=0):
# #     #print("check 1")
# #     if(dim!=1):
# #          MAP = tda.get_MAP(my_posterior)
# #     else:
# #         MAP=None
# #     #if(adaptive_MH is True):
# #    # print("check2")
# #     if algo == "MH":
# #         my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive) # gamma= adaptivity coefficient
# #     elif algo=="AM":
# #         # adaptive metropolis
# #         my_proposal=tda.AdaptiveMetropolis(C0=rwmh_cov, adaptive=rwmh_adaptive,period=period, t0=t0)   # sd am scaling parameter, gamma
# #     elif algo=="CN":
# #         # preconditioned Crank Nicolson
# #         my_proposal=tda.CrankNicolson(scaling=rmwh_scaling, adaptive=rwmh_adaptive,period=period)
# #     elif algo=="DREAMZ":
# #         my_proposal=tda.DREAMZ(M0=10*dim,adaptive=rwmh_adaptive,period=period)
# #     else: 
# #         raise ValueError("Unknown algorithm %s"%algo)
    
# #     my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
# #     idata = tda.to_inference_data(my_chains, burnin=burnin)
# #     estimates=np.array(az.summary(idata)['mean'])
# #     print(f"estimated values are {estimates}")
# #     if (diagnostic is True):
# #         print(az.summary(idata))
# #         az.plot_trace(idata)
# #         print("Autocorrelation...")
# #         az.plot_autocorr(idata)
        
    
# #     return estimates

# def MCMC(my_posterior: Any, N: int, burnin: int, n: int = 1, diagnostic: bool = True, rwmh_cov: Optional[np.ndarray] = None, 
#          rmwh_scaling: float = 0.1, period: int = 100, t0: int = 0, rwmh_adaptive: bool = False, algo: str = "MH", dim: int = 0) -> np.ndarray:

#     """
#     Perform MCMC sampling using specified algorithm.

#     Parameters:
#     - my_posterior: Posterior distribution object.
#     - N: Number of MCMC iterations.
#     - burnin: Number of burn-in iterations.
#     - n: Number of MCMC chains.
#     - diagnostic: Flag to enable diagnostic plots.
#     - rwmh_cov: Covariance matrix for RWMH proposal (optional).
#     - rmwh_scaling: Scaling factor for RWMH.
#     - period: Period for adaptive proposals.
#     - t0: Initial time step for adaptive proposals.
#     - rwmh_adaptive: Flag for adaptive RWMH.
#     - algo: MCMC algorithm to use ("MH", "AM", "CN", "DREAMZ").
#     - dim: Dimensionality of the problem.

#     Returns:
#     - estimates: MCMC estimates of the parameters.
#     """

#     if dim != 1:
#         MAP = tda.get_MAP(my_posterior)
#     else:
#         MAP = None

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

#     my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
#     idata = tda.to_inference_data(my_chains, burnin=burnin)
#     estimates = np.array(az.summary(idata)['mean'])

#     print(f"Estimated values are {estimates}")
#     if diagnostic:
#         print(az.summary(idata))
#         az.plot_trace(idata)
#         print("Autocorrelation...")
#         az.plot_autocorr(idata)

#     return estimates

# # def MCMC_cuqi(y,x,observation, N, burn_in, n=1, diagnostic=True,algo="MH",adapt=False, scale=0.3):
    
# #     x_init=np.random.rand(observation.shape[0],n)
        
# #     estimates=np.empty((observation.shape[0],0))
# #     ESSs=np.empty((observation.shape[0],0))
# #     #Geweke=np.empty((x_init.shape[0],0))
# #     #Rhat=np.empty((observation.shape[0],0))

# #     chains=np.empty((0,observation.shape[0],N-burn_in))
# #     #chains=np.empty((observation.shape[0],N-burn_in))
# #     post=np.empty((observation.shape[0],0))
# #     posterior=JointDistribution(y,x)(y=observation)

# #     for i in range(n):
        
# #         if algo=="NUTS":
# #             # Hamiltonian Monte Carlo
# #             sampler=NUTS(posterior,x0=x_init[:,i])
# #         elif algo=="MH":
# #             # Metropolis Hastings
# #             if adapt is False:
# #                 sampler=MH(posterior,x0=x_init[:,i],scale=scale)
# #             else:
# #                 sampler=MH(posterior,x0=x_init[:,i])

# #         # elif algo=="Gibbs":
# #         #     # GIbbs Sampler
# #         #     sampler=
# #         # elif algo=="CWMH":
# #         #     sampler=
# #         elif algo=="pCN":
# #             # preconditioned Crank Nicholson
# #             sampler=pCN(posterior,x0=x_init[:,i])
# #         else:
# #             raise ValueError("Unknown algorithm %s"%algo)
# #         if adapt is True:
# #             samples=sampler.sample_adapt(N-burn_in,burn_in)
# #         else:
# #             samples=sampler.sample(N-burn_in,burn_in)

# #         estimates=np.column_stack((estimates,samples.mean()[:, np.newaxis]))
# #        # ESSs=np.column_stack((ESSs,samples.compute_ess()[:, np.newaxis]))
# #   #      Rhat=np.column_stack((Rhat,samples.compute_rhat()[:, np.newaxis]))
# #         #print(Geweke)
# #         #Geweke=np.column_stack((Geweke,samples.diagnostics()[:, np.newaxis][0]))
# #                 # chains=np.concatenate(chains, samplesMH_LF.samples)
# #         #printsamples.shape)
# #         chains = np.concatenate((chains, np.expand_dims(samples.samples, axis=0)), axis=0)
# #         post=np.concatenate((post,samples.samples),axis=1)


# #         print(                f"********************  # Mean values = {estimates.mean(axis=1)}  ********************"
# #                 )
# #     print(chains.shape)
# #     for l in range(chains.shape[1]):
# #         plt.figure(figsize=(10, 4))

# #         for i in range(chains.shape[0]):
# #             plt.plot(chains[i, l,:])

# #         plt.xlabel('Sample')
# #         plt.ylabel('Value')
# #         plt.title(f'Trace Plot variable {l}')
# #         plt.legend()
# #         plt.show()
        
# #     if(diagnostic is True):
# #         if(n==1):
# #             samples.plot_trace()
# #             samples.plot_autocorrelation()
# #         else:
# #             num_bins=20
# #             plt.figure()
# #             for num in range(post.shape[0]):        
                
# #                 bin_edges = np.linspace(np.min(post[num,:]), np.max(post[num,:]), num_bins + 1)
# #                 hist, _ = np.histogram(post, bins=bin_edges)
# #                 hist=hist/post.shape[1]
# #                 bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# #                 print(hist)
# #                 plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', label=f'Var {num + 1}')

# #                 plt.xlabel('Value')
# #                 plt.ylabel('Probability')
# #                 plt.title('Distribution')
# #                 plt.legend()
# #                 plt.show()
                
# #             autocov=arviz.autocov(chains[:,0,:])
# #             ess=arviz.ess(chains[:,0,:])
# #             print(                f"********************  # ESS values = {ess}  ********************"
# #                 )


# #             plt.figure()
# #             plt.plot(autocov[0,:])
# #             plt.title('Autocovariance first chain')
# #             plt.xlabel('Lag')
# #             plt.ylabel('Autocovariance')
# #             plt.legend()
# #             plt.show()
    
# #     return estimates
# # def MCMC_cuqi(y, x, observation, N, burn_in, n=1, diagnostic=True, algo="MH", adapt=False, scale=0.3):
    
# #     x_init = np.random.rand(observation.shape[0], n)
# #     estimates = np.empty((observation.shape[0], 0))
# #     chains = np.empty((0, observation.shape[0], N - burn_in))
# #     post = np.empty((observation.shape[0], 0))
# #     posterior = JointDistribution(y, x)(y=observation)

# #     for i in range(n):
# #         if algo == "NUTS":
# #             sampler = NUTS(posterior, x0=x_init[:, i])
# #         elif algo == "MH":
# #             if not adapt:
# #                 sampler = MH(posterior, x0=x_init[:, i], scale=scale)
# #             else:
# #                 sampler = MH(posterior, x0=x_init[:, i])
# #         elif algo == "pCN":
# #             sampler = pCN(posterior, x0=x_init[:, i])
# #         else:
# #             raise ValueError("Unknown algorithm %s" % algo)
        
# #         if adapt:
# #             samples = sampler.sample_adapt(N - burn_in, burn_in)
# #         else:
# #             samples = sampler.sample(N - burn_in, burn_in)

# #         estimates = np.column_stack((estimates, samples.mean()[:, np.newaxis]))
# #         chains = np.concatenate((chains, np.expand_dims(samples.samples, axis=0)), axis=0)
# #         post = np.concatenate((post, samples.samples), axis=1)

# #         print(f"Mean values = {estimates.mean(axis=1)}")

# #     print(chains.shape)
# #     for l in range(chains.shape[1]):
# #         plt.figure(figsize=(10, 4))
# #         for i in range(chains.shape[0]):
# #             plt.plot(chains[i, l, :])
# #         plt.xlabel('Sample')
# #         plt.ylabel('Value')
# #         plt.title(f'Trace Plot variable {l}')
# #         plt.legend()
# #         plt.show()
    
# #     if diagnostic:
# #         if n == 1:
# #             samples.plot_trace()
# #             samples.plot_autocorrelation()
# #         else:
# #             num_bins = 20
# #             plt.figure()
# #             for num in range(post.shape[0]):        
# #                 bin_edges = np.linspace(np.min(post[num, :]), np.max(post[num, :]), num_bins + 1)
# #                 hist, _ = np.histogram(post[num, :], bins=bin_edges)
# #                 hist = hist / post.shape[1]
# #                 bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# #                 plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', label=f'Var {num + 1}')
# #                 plt.xlabel('Value')
# #                 plt.ylabel('Probability')
# #                 plt.title('Distribution')
# #                 plt.legend()
# #                 plt.show()
            
# #             autocov = arviz.autocov(chains[:, 0, :])
# #             ess = arviz.ess(chains[:, 0, :])
# #             print(f"ESS values = {ess}")
# #             plt.figure()
# #             plt.plot(autocov[0, :])
# #             plt.title('Autocovariance first chain')
# #             plt.xlabel('Lag')
# #             plt.ylabel('Autocovariance')
# #             plt.legend()
# #             plt.show()
    
# #     return estimates
# def MCMC_cuqi(y, x, observation, N, burn_in, n=1, diagnostic=True, algo="MH", adapt=False, scale=0.3):
#     dim = observation.shape[0]
#     x_init = np.random.rand(dim)
#     estimates = np.empty((1, 0))   # dimensione (i.e.numero ) dei parametri da stimare. Da dove lo tiri fuori?
#     chains = np.empty((0, 1, N - burn_in)) # ''
#     post = np.empty((1, 0))         # ''
#     posterior = JointDistribution(y, x)(y=observation)

#     for i in range(n):
#         if algo == "NUTS":
#             sampler = NUTS(posterior, x0=x_init)
#         elif algo == "MH":
#             if not adapt:
#                 sampler = MH(posterior, scale=scale)# x0=x_init,
#             else:
#                 sampler = MH(posterior)#, x0=x_init)
#         elif algo == "pCN":
#             sampler = pCN(posterior, x0=x_init)
#         else:
#             raise ValueError("Unknown algorithm %s" % algo)

#         if adapt:
#             samples = sampler.sample_adapt(N - burn_in, burn_in)
#         else:
#             samples = sampler.sample(N - burn_in, burn_in)

#         estimates = np.column_stack((estimates, samples.mean()[:, np.newaxis]))
#         chains = np.concatenate((chains, np.expand_dims(samples.samples, axis=0)), axis=0)
#         post = np.concatenate((post, samples.samples), axis=1)

#         print(f"Mean values = {estimates.mean(axis=1)}")

#     print(chains.shape)
#     for l in range(chains.shape[1]):
#         plt.figure(figsize=(10, 4))
#         for i in range(chains.shape[0]):
#             plt.plot(chains[i, l, :])
#         plt.xlabel('Sample')
#         plt.ylabel('Value')
#         plt.title(f'Trace Plot variable {l}')
#         plt.legend()
#         plt.show()

#     if diagnostic:
#         if n == 1:
#             samples.plot_trace()
#             samples.plot_autocorrelation()
#         else:
#             num_bins = 20
#             plt.figure()
#             for num in range(post.shape[0]):
#                 bin_edges = np.linspace(np.min(post[num, :]), np.max(post[num, :]), num_bins + 1)
#                 hist, _ = np.histogram(post[num, :], bins=bin_edges)
#                 hist = hist / post.shape[1]
#                 bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#                 plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', label=f'Var {num + 1}')
#                 plt.xlabel('Value')
#                 plt.ylabel('Probability')
#                 plt.title('Distribution')
#                 plt.legend()
#                 plt.show()

#             import arviz as az
#             autocov = az.autocov(chains[:, 0, :])
#             ess = az.ess(chains[:, 0, :])
#             print(f"ESS values = {ess}")
#             plt.figure()
#             plt.plot(autocov[0, :])
#             plt.title('Autocovariance first chain')
#             plt.xlabel('Lag')
#             plt.ylabel('Autocovariance')
#             plt.legend()
#             plt.show()

#     return estimates


# def plot_hist(estimates, real_x, output1,output2):   
#     values2=estimates
#     values1=real_x
#     diff_output=np.abs(output1-output2)
#     diff_value=np.abs(values1-values2)
#     print(f"the difference between estimated values {diff_value}\n")
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

#     return
    
    
# def custom_loss(y_pred,y_true):
#     goodind = K.not_equal(y_pred,-10)
#     #goodind = tf.math.logical_not(tf.math.is_nan(y_pred))
#     y_pred_loss = tf.boolean_mask(y_pred,goodind)
#     y_pred_true = tf.boolean_mask(y_true,goodind)
#     return K.mean(K.square(y_pred_loss - y_pred_true))





# def getOpti(name,lr):
#     if name == 'Adam':
#         return Adam(learning_rate=lr,amsgrad=True)
#     elif name == 'Nadam':
#         return Nadam(learning_rate=lr)
#     elif name == 'Adamax':
#         return Adamax(learning_rate=lr)
#     elif name == 'RMSprop':
#         return RMSprop(learning_rate=lr)
#     elif name == 'standardadam':
#         return 'adam'





# def getModel(params,num_inputs,name,num_outputs):
#     if(name == 'HF'):
#         inputs = Input(shape=(num_inputs,))
#         hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
#         #hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
#         output = Dense(num_outputs,activation='linear',name='HF')(hidden1)     
#     elif (name == 'LF'):
#         inputs = Input(shape=(num_inputs,))
#         hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(inputs)
#         hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden1)
#         hidden3 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden2)
#         hidden4 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden3)
#         output = Dense(num_outputs,activation='linear',name='LF')(hidden4)
        
#     elif (name == 'Single'):
#         inputs = Input(shape=(num_inputs,))
#         hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(inputs)
#         hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden1)
#         hidden3 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden2)
#         hidden4 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden3)
#         output = Dense(num_outputs,activation='linear',name='Single')(hidden2)        
        
#     elif (name == 'Hflin'):
#         inputs = Input(shape=(num_inputs,))
#         hiddenlin = Dense(64,activation='linear',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
#         output = Dense(num_outputs,activation='linear',name='HFlin')(hiddenlin)
        
#     elif(name == 'Hfper'):
#         inputs = Input(shape=(num_inputs,))
#         hiddenlin = Dense(64,activation=custom_activation,kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
#         output = Dense(num_outputs,activation='linear',name='HFper')(hiddenlin)    
#       # check p'arametri output  
#     elif (name == 'GP'):
#         inputs = Input(shape=(num_inputs,))
#         hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
#         hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
#         hidden3 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
#         hidden4 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden3)
#         GPlayer = Dense(2,activation='linear',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
#         outputLF = Dense(1,activation='linear',name='LF')(GPlayer)
#         outputHF = Dense(1,activation='linear',name='HF')(GPlayer)   
#         output = [outputHF,outputLF]
#         model = Model(inputs=inputs, outputs=output)
#         opti = getOpti(params['opt'],params['lr'])
#         model.compile(loss=custom_loss,loss_weights=[params['alpha'],1-params['alpha']],optimizer=opti)    
#         return model
    
#     elif (name == 'Inter'):
#         inputs = Input(shape=(num_inputs,))
#         hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(inputs)
#         hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden1)
#         outputLF = Dense(1,activation='linear',name='LF')(hidden2)
#         outputadd = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
#         merge = concatenate([outputLF,outputadd])
#         hidden3 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(merge)
#         hidden4 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden3)
#         #hidden5 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
#         #hidden6 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden5)
  
#         #lincorr = Dense(int(params['nodes']),activation='linear',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(outputLF)
#         #merge2 = concatenate([hidden3,lincorr])
#         outputHF = Dense(1,activation='linear',name='HF')(hidden4)
#         output = [outputHF,outputLF]
#         model = Model(inputs=inputs, outputs=output)
#         opti = getOpti(params['opt'],params['lr'])
#         model.compile(loss=custom_loss,loss_weights=[params['alpha'],1-params['alpha']],optimizer=opti)
#         return model
 
        
#     model = Model(inputs=inputs,  outputs=output)
#     opti = getOpti(params['opt'],params['lr'])
#     model.compile(loss='mse',optimizer=opti,metrics=['mse'])
#     return model


import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam, Nadam, Adamax
import tensorflow as tf
import arviz as az
from typing import Optional, Any, Dict, Tuple, List

import tinyDA as tda
from cuqi.distribution import JointDistribution
from cuqi.sampler import MH, NUTS, pCN

# Enable XLA JIT compilation
tf.config.optimizer.set_jit(True)

# Custom Activation Function
def custom_activation(x: tf.Tensor) -> tf.Tensor:
    """Custom activation function combining linear and non-linear transformations."""
    return x + K.square(K.sin(x))

# Custom Loss Function
def custom_loss(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """Custom loss function that ignores certain values in y_pred."""
    goodind = K.not_equal(y_pred, -10)
    y_pred_loss = tf.boolean_mask(y_pred, goodind)
    y_pred_true = tf.boolean_mask(y_true, goodind)
    return K.mean(K.square(y_pred_loss - y_pred_true))




# Get Optimizer
def getOpti(name: str, lr: float) -> tf.keras.optimizers.Optimizer:
    """Returns the optimizer based on the given name."""
    if name == 'Adam':
        return Adam(learning_rate=lr, amsgrad=True)
    elif name == 'Nadam':
        return Nadam(learning_rate=lr)
    elif name == 'Adamax':
        return Adamax(learning_rate=lr)
    elif name == 'RMSprop':
        return RMSprop(learning_rate=lr)
    elif name == 'standardadam':
        return 'adam'
    else:
        raise ValueError(f"Unknown optimizer name: {name}")

# Get Model
def getModel(params: Dict[str, Any], num_inputs: int, name: str, num_outputs: int) -> tf.keras.models.Model:
    """Returns a compiled Keras model based on the given parameters and model type."""
    inputs = Input(shape=(num_inputs,))
    if name == 'HF':
        hidden1 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HF')(hidden1)
    elif name == 'LF':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(hidden2)
        hidden4 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(hidden3)
        output = Dense(num_outputs, activation='linear', name='LF')(hidden4)
    elif name == 'Single':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']), name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']), name=f'{name}_hidden2')(hidden1)
        hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']), name=f'{name}_hidden3')(hidden2)
        hidden4 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], kernel_regularizer=l2(params['l2weight']), name=f'{name}_hidden4')(hidden3)
        output = Dense(num_outputs, activation='linear', name='Single')(hidden4)
    elif name == 'Hflin':
        hiddenlin = Dense(64, activation='linear', kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HFlin')(hiddenlin)
    elif name == 'Hfper':
        hiddenlin = Dense(64, activation=custom_activation, kernel_regularizer=l2(params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        output = Dense(num_outputs, activation='linear', name='HFper')(hiddenlin)
    elif name == 'GP':
        hidden1 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
        hidden3 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(hidden2)
        hidden4 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(hidden3)
        GPlayer = Dense(2, activation='linear', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_GP')(hidden4)
        outputLF = Dense(1, activation='linear', name='LF')(GPlayer)
        outputHF = Dense(1, activation='linear', name='HF')(GPlayer)
        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1-params['alpha']], optimizer=opti)
        return model
    elif name == 'Inter':
        hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
        hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
        outputLF = Dense(1, activation='linear', name='LF')(hidden2)
        outputadd = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'])(hidden2)
        merge = concatenate([outputLF, outputadd])
        hidden3 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(merge)
        hidden4 = Dense(int(params['nodes']), activation='tanh', kernel_regularizer=l2((1-params['alpha'])*params['l2weight']), kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(hidden3)
        outputHF = Dense(1, activation='linear', name='HF')(hidden4)
        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'], params['lr'])
        model.compile(loss=custom_loss, loss_weights=[params['alpha'], 1-params['alpha']], optimizer=opti)
        return model

    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params['opt'], params['lr'])
    model.compile(loss='mse', optimizer=opti, metrics=['mse'])
    return model

# MCMC Sampling Function
def MCMC(
    my_posterior: Any, 
    N: int, 
    burnin: int, 
    n: int = 1, 
    diagnostic: bool = True, 
    rwmh_cov: Optional[np.ndarray] = None, 
    rmwh_scaling: float = 0.1, 
    period: int = 100, 
    t0: int = 0, 
    rwmh_adaptive: bool = False, 
    algo: str = "MH", 
    dim: int = 0
) -> np.ndarray:
    """Perform MCMC sampling using the specified algorithm."""

    # Get the Maximum A Posteriori (MAP) estimate
    MAP = tda.get_MAP(my_posterior) if dim != 1 else None

    # Select the MCMC algorithm
    if algo == "MH":
        my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive)
    elif algo == "AM":
        my_proposal = tda.AdaptiveMetropolis(C0=rwmh_cov, adaptive=rwmh_adaptive, period=period, t0=t0)
    elif algo == "CN":
        my_proposal = tda.CrankNicolson(scaling=rmwh_scaling, adaptive=rwmh_adaptive, period=period)
    elif algo == "DREAMZ":
        my_proposal = tda.DREAMZ(M0=10*dim, adaptive=rwmh_adaptive, period=period)
    else:
        raise ValueError(f"Unknown algorithm {algo}")

    # Sample using the selected proposal and posterior
    my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
    idata = tda.to_inference_data(my_chains, burnin=burnin)
    estimates = np.array(az.summary(idata)['mean'])

    # Print and plot diagnostics if enabled
    print(f"Estimated values are {estimates}")
    if diagnostic:
        print(az.summary(idata))
        az.plot_trace(idata)
        print("Autocorrelation...")
        az.plot_autocorr(idata)

    return estimates

# MCMC Sampling Function for CUQI
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
    scale: float = 0.3
) -> np.ndarray:
    """Perform MCMC sampling using CUQI library."""
    dim = observation.shape[0]
    x_init = np.random.rand(dim)
    estimates = np.empty((1, 0))   # Dimension of parameters to estimate
    chains = np.empty((0, 1, N - burn_in)) 
    post = np.empty((1, 0))         
    posterior = JointDistribution(y, x)(y=observation)

    # Loop through the number of chains
    for i in range(n):
        if algo == "NUTS":
            sampler = NUTS(posterior, x0=x_init)
        elif algo == "MH":
            sampler = MH(posterior, scale=scale) if not adapt else MH(posterior)
        elif algo == "pCN":
            sampler = pCN(posterior, x0=x_init)
        else:
            raise ValueError(f"Unknown algorithm {algo}")

        samples = sampler.sample_adapt(N - burn_in, burn_in) if adapt else sampler.sample(N - burn_in, burn_in)

        estimates = np.column_stack((estimates, samples.mean()[:, np.newaxis]))
        chains = np.concatenate((chains, np.expand_dims(samples.samples, axis=0)), axis=0)
        post = np.concatenate((post, samples.samples), axis=1)

        print(f"Mean values = {estimates.mean(axis=1)}")

    # Plot trace plots
    for l in range(chains.shape[1]):
        plt.figure(figsize=(10, 4))
        for i in range(chains.shape[0]):
            plt.plot(chains[i, l, :])
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title(f'Trace Plot variable {l}')
        plt.legend()
        plt.show()

    # Diagnostic plots
    if diagnostic:
        if n == 1:
            samples.plot_trace()
            samples.plot_autocorrelation()
        else:
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

# Plot Histogram
def plot_hist(estimates: np.ndarray, real_x: np.ndarray, output1: np.ndarray, output2: np.ndarray) -> None:
    """Plot histogram comparing estimated and real values."""
    diff_output = np.abs(output1 - output2)
    diff_value = np.abs(real_x - estimates)
    print(f"The difference between estimated values {diff_value}\n")
    values = np.vstack((real_x, estimates))

    categories = np.arange(1, values.shape[1] + 1)
    bar_width = 0.35
    bar_positions = [categories - bar_width/2 + i*bar_width for i in range(values.shape[0])]
    plt.figure()
    for i in range(values.shape[0]):
        plt.bar(bar_positions[i], values[i, :], width=bar_width)

    plt.ylabel('Value')
    plt.title('Input')
    plt.xticks(categories)
    plt.legend(["Real value", "Estimate"])
    plt.show()

    return