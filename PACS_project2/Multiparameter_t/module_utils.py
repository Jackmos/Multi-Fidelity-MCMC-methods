
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Dense, Input,Dropout
from tensorflow.keras.layers import concatenate     
from keras.regularizers import l2, l1
from sklearn.model_selection import KFold
import numpy as np
from keras.optimizers import Adam,Nadam,Adamax, RMSprop
import keras.backend as K
import tensorflow as tf
import keras as kr
import h5py

from cuqi.distribution import Uniform, Gaussian,JointDistribution
from cuqi.sampler import MH
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Discrete
import tinyDA as tda
from scipy.stats import multivariate_normal,beta
import arviz as az
import time


# Custom Activation Function
def custom_activation(x: tf.Tensor) -> tf.Tensor:
    """
    Custom activation function combining linear and non-linear transformations.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Transformed tensor.
    """
    return x + K.square(K.sin(x))



def  normalization(x):
    return (x - np.min(x)) / (
    np.max(x) - np.min(x)
)

def import_data(name):
    with h5py.File(name, "r") as file:
        
        R = file["betas"]
        R = R[()]
        
        t=file["t"]
        t=t[()]

        U = file["U"]
        U = U[()]

        # V=file['V']
        # V=V[()]

    return (R, U,t)

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



# Get Optimizer
def getOpti(name: str, lr: float) -> tf.keras.optimizers.Optimizer:
    """
    Returns the optimizer based on the given name.

    Args:
        name (str): Name of the optimizer.
        lr (float): Learning rate for the optimizer.

    Returns:
        tf.keras.optimizers.Optimizer: The selected optimizer.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    optimizers = {
        'Adam': Adam(learning_rate=lr, amsgrad=True),
        'Nadam': Nadam(learning_rate=lr),
        'Adamax': Adamax(learning_rate=lr),
        'standardadam': 'adam'
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer name: {name}")
    
    return optimizers[name]



def add_noise(noise_std_data: np.ndarray, 
              noise_sta_output: np.ndarray, 
              data: np.ndarray, 
              output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adds Gaussian noise to the input data and output, concatenating the noisy versions 
    to the original arrays.

    Args:
        noise_std_data (np.ndarray): Standard deviations for noise to be added to the data.
        noise_sta_output (np.ndarray): Standard deviations for noise to be added to the output.
        data (np.ndarray): The original data array.
        output (np.ndarray): The original output array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the noisy data and output arrays.
    """
    output_flag = output.copy()  # Initialize the output_flag with the original output
    data_flag = data.copy()      # Initialize the data_flag with the original data

    # Loop over each standard deviation pair and add noise to the data and output
    for std1, std2 in zip(noise_std_data, noise_sta_output):
        # Generate Gaussian noise with mean 0 and standard deviation std1 for the output
        noise_1 = np.random.normal(0, std1, output.shape[0])
        # Generate Gaussian noise with mean 0 and standard deviation std2 for the data
        noise_2 = np.random.normal(0, std2, data.shape)
        
        # Add the noise to the original output and data
        temp1 = output + noise_1[:, np.newaxis]
        temp2 = data + noise_2
        
        # Concatenate the noisy data to the original arrays
        output_flag = np.concatenate((output_flag, temp1), axis=0)
        data_flag = np.concatenate((data_flag, temp2), axis=0)

    return output_flag, data_flag

def getModel(params,num_inputs,name):
    if(name == '2step'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)

        output = Dense(1,activation='sigmoid',name='HF')(hidden3)
    elif (name == 'LF'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64,activation=custom_activation,kernel_initializer=params['kernel_init'],kernel_regularizer=l2(0.001))(inputs)
        hidden2 = Dense(64,activation=custom_activation,kernel_initializer=params['kernel_init'],kernel_regularizer=l2(0.001))(hidden1)
        hidden3 = Dense(64,activation=custom_activation,kernel_initializer=params['kernel_init'],kernel_regularizer=l2(0.001))(hidden2)
        hidden4 = Dense(64,activation=custom_activation,kernel_initializer=params['kernel_init'],kernel_regularizer=l2(0.001))(hidden3)
        output = Dense(1,activation=custom_activation,name='LF')(hidden4)
    elif (name == 'HF'):
        # inputs = Input(shape=(num_inputs,))
        # hidden1 = Dense(64,activation=custom_activation,kernel_initializer=params['kernel_init'],kernel_regularizer=l2(0.001))(inputs)
        # hidden1=Dropout(0.5)(hidden1)
        # output = Dense(1,activation='sigmoid',name='LF')(hidden1) 
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
        output = Dense(1,activation='sigmoid',name='HF')(hidden3)
    elif (name == 'Single'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation=custom_activation,kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(inputs)
        hidden2 = Dense(int(params['nodes']),activation=custom_activation,kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden1)
        #hidden3 = Dense(int(params['nodes']),activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden2)
        #hidden4 = Dense(int(params['nodes']),activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden3)
        output = Dense(1,activation='sigmoid',name='Single')(hidden2)

    elif (name == 'Hflin'):
        inputs = Input(shape=(num_inputs,))
        hiddenlin = Dense(64,activation=custom_activation,kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(1,activation=custom_activation,name='HFlin')(hiddenlin)

    elif(name == '3step'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        #hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        output = Dense(1,activation=custom_activation,name='HF')(hidden1)

    elif (name == 'GP'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden3)
        GPlayer = Dense(2,activation=custom_activation,kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
        outputLF = Dense(1,activation=custom_activation,name='LF')(GPlayer)
        outputHF = Dense(1,activation='linear',name='HF')(GPlayer)
        output = [outputHF,outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'],params['lr'])
        model.compile(loss=custom_loss,loss_weights=[params['alpha'],1-params['alpha']],optimizer=opti)
        return model

    elif (name == 'Inter'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64,activation=custom_activation,kernel_initializer=params['kernel_init'],kernel_regularizer=l2((1-params['alpha'])*params['l2weight']))(inputs)
        hidden2 = Dense(64,activation=custom_activation,kernel_initializer=params['kernel_init'],kernel_regularizer=l2((1-params['alpha'])*params['l2weight']))(hidden1)
        outputLF = Dense(1,activation=custom_activation,name='LF')(hidden2)
        outputadd = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
        merge = concatenate([outputLF,outputadd])
        hidden3 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(merge)
        hidden4 = Dense(int(params['nodes']),activation=custom_activation,kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden3)
        #hidden5 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
        #hidden6 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden5)

        #lincorr = Dense(int(params['nodes']),activation='linear',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(outputLF)
        #merge2 = concatenate([hidden3,lincorr])
        outputHF = Dense(1,activation='sigmoid',name='HF')(hidden4)
        output = [outputHF,outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'],params['lr'])
        model.compile(loss=custom_loss,loss_weights=[params['alpha'],1-params['alpha']],optimizer=opti)
        return model


    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params['opt'],params['lr'])
    model.compile(loss='mse',optimizer=opti,metrics=['mse'])
    return model

