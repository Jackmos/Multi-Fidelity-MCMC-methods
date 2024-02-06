
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

from abc import ABCMeta, abstractstaticmethod, abstractmethod

def custom_activation(x):
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

