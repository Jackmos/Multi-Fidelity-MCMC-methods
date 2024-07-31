import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, LSTM, Dropout
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

def MCMC(my_posterior,N, burnin, n=1, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1, period=100, t0=0, rwmh_adaptive=False,algo="MH",dim=0):
    #print("check 1")
    if(dim!=1):
         MAP = tda.get_MAP(my_posterior)
    else:
        MAP=None
    #if(adaptive_MH is True):
   # print("check2")
    if algo == "MH":
        my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive) # gamma= adaptivity coefficient
    elif algo=="AM":
        # adaptive metropolis
        my_proposal=tda.AdaptiveMetropolis(C0=rwmh_cov, adaptive=rwmh_adaptive,period=period, t0=t0)   # sd am scaling parameter, gamma
    elif algo=="CN":
        # preconditioned Crank Nicolson
        my_proposal=tda.CrankNicolson(scaling=rmwh_scaling, adaptive=rwmh_adaptive,period=period)
    elif algo=="DREAMZ":
        my_proposal=tda.DREAMZ(M0=10*dim,adaptive=rwmh_adaptive,period=period)
    else: 
        raise ValueError("Unknown algorithm %s"%algo)
    
    my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
    idata = tda.to_inference_data(my_chains, burnin=burnin)
    estimates=np.array(az.summary(idata)['mean'])
    print(f"estimated values are {estimates}")
    if (diagnostic is True):
        print(az.summary(idata))
        az.plot_trace(idata)
        print("Autocorrelation...")
        az.plot_autocorr(idata)
        
    
    return estimates



def MCMC_cuqi(y,x,observation, N, burn_in, n=1, diagnostic=True,algo="MH",adapt=False, scale=0.3):
    
    x_init=np.random.rand(observation.shape[0],n)
        
    estimates=np.empty((observation.shape[0],0))
    ESSs=np.empty((observation.shape[0],0))
    #Geweke=np.empty((x_init.shape[0],0))
    #Rhat=np.empty((observation.shape[0],0))

    chains=np.empty((0,observation.shape[0],N-burn_in))
    #chains=np.empty((observation.shape[0],N-burn_in))
    post=np.empty((observation.shape[0],0))
    posterior=JointDistribution(y,x)(y=observation)

    for i in range(n):
        
        if algo=="NUTS":
            # Hamiltonian Monte Carlo
            sampler=NUTS(posterior,x0=x_init[:,i])
        elif algo=="MH":
            # Metropolis Hastings
            if adapt is False:
                sampler=MH(posterior,x0=x_init[:,i],scale=scale)
            else:
                sampler=MH(posterior,x0=x_init[:,i])

        # elif algo=="Gibbs":
        #     # GIbbs Sampler
        #     sampler=
        # elif algo=="CWMH":
        #     sampler=
        elif algo=="pCN":
            # preconditioned Crank Nicholson
            sampler=pCN(posterior,x0=x_init[:,i])
        else:
            raise ValueError("Unknown algorithm %s"%algo)
        if adapt is True:
            samples=sampler.sample_adapt(N-burn_in,burn_in)
        else:
            samples=sampler.sample(N-burn_in,burn_in)

        estimates=np.column_stack((estimates,samples.mean()[:, np.newaxis]))
       # ESSs=np.column_stack((ESSs,samples.compute_ess()[:, np.newaxis]))
  #      Rhat=np.column_stack((Rhat,samples.compute_rhat()[:, np.newaxis]))
        #print(Geweke)
        #Geweke=np.column_stack((Geweke,samples.diagnostics()[:, np.newaxis][0]))
                # chains=np.concatenate(chains, samplesMH_LF.samples)
        #printsamples.shape)
        chains = np.concatenate((chains, np.expand_dims(samples.samples, axis=0)), axis=0)
        post=np.concatenate((post,samples.samples),axis=1)


        print(                f"********************  # Mean values = {estimates.mean(axis=1)}  ********************"
                )
    print(chains.shape)
    for l in range(chains.shape[1]):
        plt.figure(figsize=(10, 4))

        for i in range(chains.shape[0]):
            plt.plot(chains[i, l,:])

        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title(f'Trace Plot variable {l}')
        plt.legend()
        plt.show()
        
    if(diagnostic is True):
        if(n==1):
            samples.plot_trace()
            samples.plot_autocorrelation()
        else:
            num_bins=20
            plt.figure()
            for num in range(post.shape[0]):        
                
                bin_edges = np.linspace(np.min(post[num,:]), np.max(post[num,:]), num_bins + 1)
                hist, _ = np.histogram(post, bins=bin_edges)
                hist=hist/post.shape[1]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                print(hist)
                plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', label=f'Var {num + 1}')

                plt.xlabel('Value')
                plt.ylabel('Probability')
                plt.title('Distribution')
                plt.legend()
                plt.show()
                
            autocov=arviz.autocov(chains[:,0,:])
            ess=arviz.ess(chains[:,0,:])
            print(                f"********************  # ESS values = {ess}  ********************"
                )


            plt.figure()
            plt.plot(autocov[0,:])
            plt.title('Autocovariance first chain')
            plt.xlabel('Lag')
            plt.ylabel('Autocovariance')
            plt.legend()
            plt.show()
    
    return estimates




def plot_hist(estimates, real_x, output1,output2):   
    values2=estimates
    values1=real_x
    diff_output=np.abs(output1-output2)
    diff_value=np.abs(values1-values2)
    print(f"the difference between estimated values {diff_value}\n")
    values = np.vstack((values1, values2))

    # Creare categorie in base alla lunghezza di values
    categories = np.arange(1, values.shape[1] + 1)

    # Larghezza delle colonne
    bar_width = 0.35

    # Posizioni delle colonne
    bar_positions = [categories - bar_width/2 + i*bar_width for i in range(values.shape[0])]
    plt.figure()
    # Creazione del plot
    for i in range(values.shape[0]):
        plt.bar(bar_positions[i], values[i, :], width=bar_width)


    plt.ylabel('Value')
    plt.title('Input')
    plt.xticks(categories)
    plt.legend(["Real value", "Estimate"])



    plt.show()

    return
    
    
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
        
        a = LSTM(params['nodes'], return_sequences=True)(inputs)
        for i in range(params['lay'] - 1):
            a = Dropout(params['dropout'])(a)
            a = LSTM(params['nodes'], return_sequences=True)(a)
        
        # Adding a Fourier Layer
        a = FourierLayer(output_dim=params['nodes'])(a)
        
        for i in range(params['lay_dense']):
            a = Dense(params['nodes_dense'], activation=custom_activation)(a)
            
        output = Dense(num_outputs, activation='linear')(a)


inputs = Input(shape=(None, num_inputs))
a = Bidirectional(LSTM(params['nodes'], return_sequences=True,
                       kernel_regularizer=l2(params['l2_reg'])))(inputs)
a = Dropout(params['dropout'])(a)
a = Bidirectional(LSTM(params['nodes'], return_sequences=True,
                       kernel_regularizer=l2(params['l2_reg'])))(a)

# Adding a Fourier Layer
a = FourierLayer(output_dim=params['nodes'])(a)

a = Dense(params['nodes_dense'], activation=custom_activation,
          kernel_regularizer=l2(params['l2_reg']))(a)

output = Dense(num_outputs, activation='linear')(a)




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


