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
from keras.regularizers import l2, l1
from sklearn.model_selection import KFold
import numpy as np
from keras.optimizers import Adam, Nadam, Adamax, RMSprop
import keras.backend as K
import tensorflow as tf
import keras as kr
import h5py
import sys
import os
import warnings

from cuqi.distribution import Uniform, Gaussian,JointDistribution
from cuqi.sampler import MH
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Discrete
import tinyDA as tda
from scipy.stats import multivariate_normal,beta
import arviz as az
import time

from abc import ABCMeta, abstractstaticmethod, abstractmethod
    

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

def sinusoidal_activation(x):
    return K.square(K.sin(x))

def  normalization(x):
    return (x - np.min(x)) / (
    np.max(x) - np.min(x)
)

def import_data(name):#-> Tuple[np.array, np.array]:
    """imports data defined in a .mat file

    Args:
        name : name of the file

    Returns:
        Tuple[np.array, np.array]: input and output of the NN
    """
    with h5py.File(name, "r") as file:
        
        R = file["betas"][()]

        U = file["U"][()]

        # V=file['V']
        # V=V[()]

    return (R, U)

def MCMC(my_posterior,N, burnin, n=1, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1, period=100, t0=0, rwmh_adaptive=False,algo="MH",dim=0):
    
    MAP = tda.get_MAP(my_posterior)
    #if(adaptive_MH is True):
    if algo == "MH":
        my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive)
    elif algo=="AM":
        # adaptive metropolis
        my_proposal=tda.AdaptiveMetropolis(C0=rwmh_cov, adaptive=rwmh_adaptive,period=period, t0=t0)
    elif algo=="CN":
        # preconditioned Crank Nicolson
        my_proposal=tda.CrankNicolson(scaling=rmwh_scaling, adaptive=rwmh_adaptive,period=period)
    elif algo=="DREAMZ":
        my_proposal=tda.DREAMZ(M0=10*dim,adaptive=rwmh_adaptive,period=period)
    else: 
        raise ValueError("Unknown algorithm %s"%algo)
    
    my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, initial_parameters=MAP,force_sequential=True)
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

def plot_hist(estimates, real_x, output1,output2):   
    values2=estimates
    values1=real_x[:,0] #! [:,0] aggiunto solo per questo caso !
    print(np.abs(values2-values1))
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


    # Mostra il plot
    plt.show()
    
    values2 = output2
    values1=output1

    values = np.vstack((values1, values2))

    # Creazione del plot
    for i in range(values.shape[0]):
        plt.bar(bar_positions[i], values[i, :], width=bar_width)


    plt.ylabel('Value')
    plt.title('Output')
    plt.xticks(categories)
    plt.legend(["Real value", "Estimate"])
    # Mostra il plot
    plt.show()
    return   


# NON USATA
def custom_loss(y_pred, y_true):
    goodind = K.not_equal(y_pred, -10)
    # -10 is used as special value to indicate NaN

    # goodind = tf.math.logical_not(tf.math.is_nan(y_pred))
    y_pred_loss = tf.boolean_mask(y_pred, goodind)
    y_pred_true = tf.boolean_mask(y_true, goodind)
    return K.mean(K.square(y_pred_loss - y_pred_true))  # MSE

def getOpti(name, lr):
    if name == "Adam":
        return Adam(learning_rate=lr, amsgrad=True)
    elif name == "Nadam":
        return Nadam(learning_rate=lr)
    elif name == "Adamax":
        return Adamax(learning_rate=lr)
    elif name == "RMSprop":
        return RMSprop(learning_rate=lr)
    elif name == "standardadam":
        return "adam"

def add_noise(noise_std_data, noise_sta_output, data, output):
    output_flag=output
    data_flag=data
    for std1,std2 in zip(noise_std_data,noise_sta_output):
        noise_1 = np.random.normal(0, std1, output.shape[0])
        noise_2 = np.random.normal(0, std2, data.shape)
        temp1=output+noise_1
        temp2=data+noise_2
        output_flag=np.concatenate((output_flag,temp1),axis=0)
        #print(data_flag.shape)
        #print(temp2.shape)
        data_flag=np.concatenate((data_flag,temp2))
    return (output_flag,data_flag)

def getModel(params,num_inputs, name, num_outputs):
    if name == "2step":
        inputs = Input(shape=(num_inputs,))  
        
        hidden1 = Dense(
            int(params["nodes"]),
            kernel_regularizer=l2(params["l2weight"]),
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
        )(
            inputs
        )  
        hidden1=Dropout(0.05)(hidden1)

        fourier_layer = FourierLayer(output_dim=int(params["nodes"]))(hidden1)
        
        hidden2 = Dense(
            int(params["nodes"]),
            kernel_regularizer=l2(params["l2weight"]),
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            
        )(
            fourier_layer
        ) 
                
        output = Dense(num_outputs, activation="linear", name="HF")(hidden2)


    elif name == "LF":
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

    elif name == "HF":
        inputs = Input(shape=(num_inputs,))  
        hidden1 = Dense(
            int(params["nodes"]),
            kernel_regularizer=l2(params["l2weight"]),
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
           
        )(
            inputs
        )  
        hidden1=Dropout(0.05)(hidden1)

        fourier_layer = FourierLayer(output_dim=int(params["nodes"]))(hidden1)
        
        hidden2 = Dense(
            int(params["nodes"]),
            kernel_regularizer=l2(params["l2weight"]),
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            #kernel_constraint=clip_norm(1.0)
        )(
            fourier_layer
        ) 
                
        output = Dense(num_outputs, activation="linear", name="HF")(hidden2)

    elif name == "Single":
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"]),
        )(inputs)
        hidden2 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"]),
        )(hidden1)
        hidden3 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"]),
        )(hidden2)
        hidden4 = Dense(
            64,
            activation="sigmoid",
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"]),
        )(hidden3)
        output = Dense(num_outputs, activation="sigmoid", name="Single")(hidden2)

    elif name == "Hflin":
        inputs = Input(
            shape=(num_inputs,)
        )  # second NN (NN_Lin) in the 3-steps architecture
        # Linear activation function: it approximates the high-fidelity data by a linear combiantion of the inputs
        # and is thus responsible for capturing the linear correlations between the datasets
        hiddenlin = Dense(
            64,
            activation=custom_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(inputs)
        output = Dense(num_outputs, activation="sigmoid", name="HFlin")(hiddenlin)

    elif name == "Hfper":
        inputs = Input(
            shape=(num_inputs,)
        )  # second NN (NN_Lin) in the 3-steps architecture
        # Linear activation function: it approximates the high-fidelity data by a linear combiantion of the inputs
        # and is thus responsible for capturing the linear correlations between the datasets
        hiddenper = Dense(
            64,
            activation=sinusoidal_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(inputs)
        hiddenper2 = Dense(
            64,
            activation=sinusoidal_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hiddenper)
        output = Dense(num_outputs, activation="linear", name="HFper")(hiddenper2)

    elif name == "3step":
        inputs = Input(
            shape=(num_inputs,)
        )  
        hidden1 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(inputs)
        output = Dense(num_outputs, activation="sigmoid", name="HF")(hidden1)
# check end parametri 
    elif name == "GP":
        # architecture which is supposed to mimic the action of a GP
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(inputs)
        hidden2 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden1)
        hidden3 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden2)
        hidden4 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden3)

        GPlayer = Dense(
            2,
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden4)

        outputLF = Dense(1, activation="linear", name="LF")(GPlayer)
        outputHF = Dense(1, activation="linear", name="HF")(GPlayer)

        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params["opt"], params["lr"])
        model.compile(
            loss=custom_loss,
            loss_weights=[params["alpha"], 1 - params["alpha"]],
            optimizer=opti,
        )
        return model

    elif name == "Inter":
        inputs = Input(shape=(num_inputs,))

        hidden1 = Dense(
            64, activation=custom_activation, kernel_initializer=params["kernel_init"]
        )(
            inputs
        )  # The same input layer is used for high and low fidelity data
        hidden2 = Dense(
            64, activation=custom_activation, kernel_initializer=params["kernel_init"]
        )(hidden1)
        outputLF = Dense(1, activation=custom_activation, name="LF")(
            hidden2
        )  # low-fidelity output is situate at the third hidden layer.
        outputadd = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden2)
        merge = kr.layers.concatenate([outputLF, outputadd])

        hidden3 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(merge)

        hidden4 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden3)

        outputHF = Dense(1, activation="sigmoid", name="HF")(hidden4)
        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params["opt"], params["lr"])
        model.compile(
            loss=custom_loss,
            loss_weights=[params["alpha"], 1 - params["alpha"]],
            optimizer=opti,
        )
        # alpha weighs the different fidelity level components of the error
        return model

    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params["opt"], params["lr"])
    model.compile(loss="mse", optimizer=opti, metrics=["mse"])
    return model