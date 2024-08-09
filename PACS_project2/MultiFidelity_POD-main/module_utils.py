import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, LSTM, Dropout
from typing import Callable, Tuple, Any, Dict, Union, List, Optional
from itertools import product
import scipy.io

from tensorflow.keras.optimizers import Adam,Nadam,Adamax
import tensorflow as tf
import arviz
from keras.layers import Layer

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

from cuqi.distribution import Uniform, Gaussian,JointDistribution, Beta
from cuqi.sampler import MH,NUTS, Gibbs, CWMH, pCN
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Discrete
#from cuqi.diagnostics import Geweke
import tinyDA as tda
from scipy.stats import multivariate_normal,beta
import arviz as az
import time 


    
def custom_activation(x):
    return x + K.square(K.sin(x))

def  normalization(x, xmax,xmin):
    return (x - xmin) / (
    xmax - xmin
)

def  denormalization(x, xmax,xmin ):
    return x*(xmax-xmin) +xmin

# def MCMC(my_posterior,N, burnin, n=1, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1, period=100, t0=0, rwmh_adaptive=False,algo="MH",dim=0):
#     #print("check 1")
#     if(dim!=1):
#          MAP = tda.get_MAP(my_posterior)
#     else:
#         MAP=None
#     #if(adaptive_MH is True):
#    # print("check2")
#     if algo == "MH":
#         my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive) # gamma= adaptivity coefficient
#     elif algo=="AM":
#         # adaptive metropolis
#         my_proposal=tda.AdaptiveMetropolis(C0=rwmh_cov, adaptive=rwmh_adaptive,period=period, t0=t0)   # sd am scaling parameter, gamma
#     elif algo=="CN":
#         # preconditioned Crank Nicolson
#         my_proposal=tda.CrankNicolson(scaling=rmwh_scaling, adaptive=rwmh_adaptive,period=period)
#     elif algo=="DREAMZ":
#         my_proposal=tda.DREAMZ(M0=10*dim,adaptive=rwmh_adaptive,period=period)
#     else: 
#         raise ValueError("Unknown algorithm %s"%algo)
    
#     my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
#     idata = tda.to_inference_data(my_chains, burnin=burnin)
#     estimates=np.array(az.summary(idata)['mean'])
#     print(f"estimated values are {estimates}")
#     if (diagnostic is True):
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
    
def custom_activation(x):
    return x + K.square(K.sin(x))

class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(name='context_vector', shape=(input_shape[-1],), initializer='random_normal', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

def getModel(params,num_inputs,name,num_outputs):
    if(name == "LSTM"):
        inputs = Input(shape=(None, num_inputs))

        a = LSTM(params['nodes'], return_sequences = True)(inputs) 
        for i in range(params['lay']-1):
            a = Dropout(params['dropout'])(a)
            a = LSTM(params['nodes'], return_sequences = True)(a)
        for i in range(params['lay_dense']):
            a  = Dense(params['nodes_dense'])(a)
        
        output = Dense(num_outputs,activation='linear')(a)


    elif (name == "LSTM2"):
        inputs = Input(shape=(None, num_inputs))
        if params["sched"] is True:
            params['lr'] = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params['lr'],
            decay_steps=5000,
            decay_rate=0.9)

        a = LSTM(params['nodes'], return_sequences=True, kernel_regularizer=l2(params['l2_reg']))(inputs)
                # Adding a Fourier Layer
        a = FourierLayer(output_dim=params['nodes'])(a)
        for i in range(params['lay'] - 1):
            a = Dropout(params['dropout'])(a)
            a = LSTM(params['nodes'], return_sequences=True, kernel_regularizer=l2(params['l2_reg']))(a)
        
        # Adding a Fourier Layer
        a = FourierLayer(output_dim=params['nodes'])(a)
        
        for i in range(params['lay_dense']):
            a = Dense(params['nodes_dense'], activation=custom_activation, kernel_regularizer=l2(params['l2_reg']))(a)
            
        output = Dense(num_outputs, activation='linear')(a)



# inputs = Input(shape=(None, num_inputs))
# a = Bidirectional(LSTM(params['nodes'], return_sequences=True,
#                        kernel_regularizer=l2(params['l2_reg'])))(inputs)
# a = Dropout(params['dropout'])(a)
# a = Bidirectional(LSTM(params['nodes'], return_sequences=True,
#                        kernel_regularizer=l2(params['l2_reg'])))(a)

# # Adding a Fourier Layer
# a = FourierLayer(output_dim=params['nodes'])(a)

# a = Dense(params['nodes_dense'], activation=custom_activation,
#           kernel_regularizer=l2(params['l2_reg']))(a)

# output = Dense(num_outputs, activation='linear')(a)


    elif(name == 'HF'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        #hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        output = Dense(num_outputs,activation='linear',name='HF')(hidden1)     
    elif (name == 'LF'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
            
        )(inputs)
        hidden2 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
            
        )(hidden1)
        fourier_layer1 = FourierLayer(output_dim=64)(hidden2)

        hidden3 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
           
        )(fourier_layer1)
        
        hidden3=Dropout(0.05)(hidden3)

        hidden4 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
            #kernel_constraint=clip_norm(1.0)
        )(hidden3)

        
        output = Dense(num_outputs, activation="linear", name="LF")(hidden4)
        
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
      # check p'arametri output  
    elif (name == 'GP'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden3)
        GPlayer = Dense(2,activation='linear',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
        outputLF = Dense(1,activation='linear',name='LF')(GPlayer)
        outputHF = Dense(1,activation='linear',name='HF')(GPlayer)   
        output = [outputHF,outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'],params['lr'])
        model.compile(loss=custom_loss,loss_weights=[params['alpha'],1-params['alpha']],optimizer=opti)    
        return model
    
    elif (name == 'Inter'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden1)
        outputLF = Dense(1,activation='linear',name='LF')(hidden2)
        outputadd = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
        merge = concatenate([outputLF,outputadd])
        hidden3 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(merge)
        hidden4 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden3)
        #hidden5 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
        #hidden6 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden5)
  
        #lincorr = Dense(int(params['nodes']),activation='linear',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(outputLF)
        #merge2 = concatenate([hidden3,lincorr])
        outputHF = Dense(1,activation='linear',name='HF')(hidden4)
        output = [outputHF,outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'],params['lr'])
        model.compile(loss=custom_loss,loss_weights=[params['alpha'],1-params['alpha']],optimizer=opti)
        return model
 
        
    model = Model(inputs=inputs,  outputs=output)
    opti = getOpti(params['opt'],params['lr'])
    model.compile(loss='mse',optimizer=opti,metrics=['mse'])
    return model

def load_reaction_diffusion(params, fidelity, path, splitted = False):
    u_list = []
    for param in params:
        name = path + 'u_' + fidelity + '_' + "{:.3f}".format(param) 
        
        if splitted:
            u_test_list = []
            for i in [1,2]:
                u_test = scipy.io.loadmat(name + '_' + str(i) + '.mat')['u']
                u_test_list.append(u_test)
            u = np.concatenate(u_test_list, axis = 2)
        else:
            u = scipy.io.loadmat(name + '.mat')['u']
            
        u_list.append(u)
    
    data_u = np.stack(u_list, axis=3)
    x = scipy.io.loadmat(path + 'x_' + fidelity + '.mat')['x']
    t = scipy.io.loadmat(path + 't_' + fidelity + '.mat')['t']
    
    return data_u, x.flatten(), t.flatten()



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

def run_simulation(datahf_x: np.ndarray, datahf: np.ndarray, mean_prior: np.ndarray, cov_prior: np.ndarray, Yhf: np.ndarray, 
                   sigma_noise: List[float], n_data: List[int], n_data_x:List[int], parameters: np.ndarray, sigma: np.ndarray, 
                   rwmh_scaling: np.ndarray, rwmh_cov: np.ndarray, rwmh_adaptive: bool, 
                   iterations: int, burnin: int, n_chains: int, final_model: Any, algo: str, 
                   forward_low_fidelity: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a simulation to estimate parameters and calculate errors.
    
    Parameters:
    - datahf_x (np.ndarray): 1D array containing data.
    - datahf (np.ndarray): 2D array containing data  (t and parameter).
    - mean_prior (np.ndarray): 1D array for the mean of the prior.
    - cov_prior (np.ndarray): 2D array for the covariance of the prior.
    - Yhf (np.ndarray): 1D array of observed values.
    - sigma_noise (List[float]): List of noise levels.
    - n_data (List[int]): List of number of data points along t.
    - n_data_x (List[int]): List of number of data points along x.
    - parameters (np.ndarray): 1D array of parameters.
    - sigma (np.ndarray): 1D array of standard deviations for the likelihood.
    - rwmh_scaling (np.ndarray): 1D array of scaling factors for the RWMH algorithm.
    - rwmh_cov (np.ndarray): 2D array for the RWMH covariance.
    - rwmh_adaptive (bool): Boolean indicating if RWMH is adaptive.
    - iterations (int): Integer for the number of iterations.
    - burnin (int): Integer for the burn-in period.
    - n_chains (int): Integer for the number of chains.
    - final_model (Any): The model object with the param_inverse method.
    - algo (str): String indicating the algorithm to use.
    - forward_low_fidelity (Optional[Callable]): Low fidelity forward model function (optional).
    
    Returns:
    - best_estimate (np.ndarray): The best parameter estimate.
    - best_error (np.ndarray): The error corresponding to the best estimate.
    """
    
    # Initialize error and estimate arrays
    error_shape = (len(sigma_noise), len(n_data), len(n_data_x),len(sigma), len(rwmh_scaling))
    error = np.zeros(error_shape)
    estimates = np.zeros(error_shape)
    
    # Iterate over all combinations of parameters using itertools.product
    for (i, noise), (k, n),(w,n_x), (t, s), (j, r) in product(enumerate(sigma_noise), enumerate(n_data), enumerate(n_data_x), enumerate(sigma), enumerate(rwmh_scaling)):
        t_eval = np.linspace(np.min(datahf[:,0]), np.max(datahf[:,0]), n).reshape(-1, 1)  # Generate evaluation times
        x_eval=np.linspace(np.min(datahf_x),np.max(datahf_x),n_x).reshape(-1, 1)
        nearest_x, y_obs = process_data(datahf, parameters, t_eval, Yhf)  # Get nearest t values and observations
        cov_likelihood = calculate_cov_likelihood(s, t_eval)  # Compute the covariance for the likelihood
        
        # Perform parameter estimation and calculate error
        estimates[i, k, w,t, j], error[i, k,w, t, j] = final_model.param_inverse(
            mean_prior, x_eval, t_eval, cov_prior=cov_prior, rmwh_scaling=r, 
            cov_noise=noise, cov_likelihood=cov_likelihood, y_obs=y_obs, 
            x_real=parameters, number_chains=n_chains, N=iterations, 
            burn_in=burnin, diagnostic=True, rwmh_cov=rwmh_cov, 
            rwmh_adaptive=rwmh_adaptive, algo=algo, forward_low_fidelity=forward_low_fidelity
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