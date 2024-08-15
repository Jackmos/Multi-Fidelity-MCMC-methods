from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
from keras.models import save_model
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter


import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer,concatenate, Dropout, Lambda
from tensorflow.keras.optimizers import Adam, Nadam, Adamax, RMSprop


from sklearn.model_selection import KFold
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
from typing import Tuple

from abc import ABCMeta, abstractstaticmethod, abstractmethod
    
def fft_layer(x):
    return tf.signal.fft(tf.cast(x, dtype=tf.complex64))

def ifft_layer(x):
    return tf.signal.ifft(x)


# Manually specify output shape to avoid NotImplementedError
def fft_output_shape(input_shape):
    return input_shape

def ifft_output_shape(input_shape):
    return input_shape

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

# class FFTLayer(Layer):
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(FFTLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Initialize weights (if any), here just for structure, as FFT itself doesn't use learnable weights
#         super(FFTLayer, self).build(input_shape)

#     def call(self, x):
#         # Cast input to complex64 and apply FFT
#         x_complex = tf.cast(x, dtype=tf.complex64)
#         fft_result = tf.signal.fft(x_complex)
#         # Return both magnitude and phase
#         magnitude = tf.math.abs(fft_result)
#         phase = tf.math.angle(fft_result)
#         return tf.concat([magnitude, phase], axis=-1)

#     def compute_output_shape(self, input_shape):
#         return input_shape

# class IFFTLayer(Layer):
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(IFFTLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Initialize weights (if any), here just for structure, as IFFT itself doesn't use learnable weights
#         super(IFFTLayer, self).build(input_shape)

#     def call(self, x):
#         # Split magnitude and phase
#         magnitude = x[:, :x.shape[-1] // 2]
#         phase = x[:, x.shape[-1] // 2:]
#         # Reconstruct complex numbers
#         real = magnitude * tf.math.cos(phase)
#         imag = magnitude * tf.math.sin(phase)
#         x_complex = tf.complex(real, imag)
#         # Apply IFFT
#         ifft_result = tf.signal.ifft(x_complex)
#         return tf.math.real(ifft_result)  # Return only the real part

#     def compute_output_shape(self, input_shape):
#         return input_shape

class IFFTLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(IFFTLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Applicare la Trasformata Inversa di Fourier
        inputs_complex = tf.cast(inputs, dtype=tf.complex64)
        ifft_result = tf.signal.ifft(inputs_complex)
        # Ritornare solo la parte reale
        return tf.math.real(ifft_result)

    def compute_output_shape(self, input_shape):
        # Assicurati che la dimensione dell'output sia coerente con output_dim
        return (input_shape[0], self.output_dim)

class FFTLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FFTLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Applicare la Trasformata di Fourier
        inputs_complex = tf.cast(inputs, dtype=tf.complex64)
        fft_result = tf.signal.fft(inputs_complex)
        # Ritornare solo la parte reale
        return tf.math.real(fft_result)

    def compute_output_shape(self, input_shape):
        # Assicurati che la dimensione dell'output sia coerente con output_dim
        return (input_shape[0], self.output_dim)
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


def sinusoidal_activation(x):
    return K.square(K.sin(x))

def  normalization(x):
    return (x - np.min(x)) / (
    np.max(x) - np.min(x)
)

def import_data(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Imports data defined in a .mat file.

    Args:
        name (str): Name of the .mat file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing two numpy arrays representing 
                                       the input (R) and output (U) data for the NN.
    """
    # Open the .mat file in read-only mode and automatically close it after reading
    with h5py.File(name, "r") as file:
        # Read the data for 'betas' and 'U' into numpy arrays
        R = file["betas"][()]  # Input data
        U = file["U"][()]      # Output data

    return R, U  # Return the input and output data as a tuple


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




def getModel(params: dict, num_inputs: int, name: str, num_outputs: int) -> Model:
    """
    Creates and compiles a Keras model based on the provided parameters and model name.

    Args:
        params (dict): Dictionary containing model parameters such as 'nodes', 'l2weight', etc.
        num_inputs (int): Number of input features.
        name (str): Name of the model architecture to create.
        num_outputs (int): Number of output nodes.

    Returns:
        Model: A compiled Keras model.
    """

    inputs = Input(shape=(num_inputs,))
    

    if name == "LF":
        # Model architecture for the "LF" model
        hidden1 = Dense(
            units=64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001)
        )(inputs)
        
        hidden2 = Dense(
            units=64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001)
        )(hidden1)
        
        fourier_layer1 = FourierLayer(output_dim=64)(hidden2)
        
        hidden3 = Dense(
            units=64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001)
        )(fourier_layer1)
        hidden3 = Dropout(0.05)(hidden3)
        
        hidden4 = Dense(
            units=64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001)
        )(hidden3)
        
        output = Dense(units=num_outputs, activation="linear", name="LF")(hidden4)

    elif name == "HF":
        # Model architecture for the "HF" model
        hidden1 = Dense(
            units=int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(inputs)
        hidden1 = Dropout(0.05)(hidden1)
        
        fourier_layer = FourierLayer(output_dim=int(params["nodes"]))(hidden1)
        
        hidden2 = Dense(
            units=int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(fourier_layer)
        
        output = Dense(units=num_outputs, activation="linear", name="HF")(hidden2)

    elif name == "Single":
        # Model architecture for the "Single" model
        hidden1 = Dense(
            units=64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"])
        )(inputs)
        
        hidden2 = Dense(
            units=64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"])
        )(hidden1)
        
        hidden3 = Dense(
            units=64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"])
        )(hidden2)
        
        hidden4 = Dense(
            units=64,
            activation="sigmoid",
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"])
        )(hidden3)
        
        output = Dense(units=num_outputs, activation="sigmoid", name="Single")(hidden2)

    elif name == "Hflin":
        # Model architecture for the "Hflin" model
        hiddenlin = Dense(
            units=64,
            activation=custom_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(inputs)
        
        output = Dense(units=num_outputs, activation="sigmoid", name="HFlin")(hiddenlin)

    elif name == "Hfper":
        # Model architecture for the "Hfper" model
        hiddenper = Dense(
            units=64,
            activation=sinusoidal_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(inputs)
        
        hiddenper2 = Dense(
            units=64,
            activation=sinusoidal_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(hiddenper)
        
        output = Dense(units=num_outputs, activation="linear", name="HFper")(hiddenper2)



                
        # # Applicazione della FFT
        # fft_layer = FFTLayer(output_dim=64)(inputs)

        # # Strati Densi per modificare le componenti frequenziali
        # hidden = Dense(units=64, activation='relu', kernel_regularizer=l2(params["l2weight"]), kernel_initializer=params["kernel_init"])(fft_layer)
        # hidden = Dense(units=64, activation='relu', kernel_regularizer=l2(params["l2weight"]), kernel_initializer=params["kernel_init"])(hidden)

        # # Applicazione dell'IFFT
        # ifft_layer = IFFTLayer(output_dim=64)(hidden)

        # # Output del modello
        # output = Dense(units=num_outputs, activation="linear", name="output_hf")(ifft_layer)



    elif name == "GP":
        # Model architecture for the "GP" model
        hidden1 = Dense(
            units=int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(inputs)
        
        hidden2 = Dense(
            units=int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(hidden1)
        
        hidden3 = Dense(
            units=int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(hidden2)
        
        hidden4 = Dense(
            units=int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(hidden3)
        
        GPlayer = Dense(
            units=2,
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(hidden4)
        
        outputLF = Dense(1, activation="linear", name="LF")(GPlayer)
        outputHF = Dense(1, activation="linear", name="HF")(GPlayer)
        
        output = [outputHF, outputLF]
        
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params["opt"], params["lr"])
        model.compile(
            loss=custom_loss,
            loss_weights=[params["alpha"], 1 - params["alpha"]],
            optimizer=opti
        )
        return model

    elif name == "Inter":
        # Model architecture for the "Inter" model
        hidden1 = Dense(
            units=64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"]
        )(inputs)
        
        fourier_layer1 = FourierLayer(output_dim=64)(hidden1)
        
        hidden2 = Dense(
            units=64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"]
        )(fourier_layer1)
        
        outputLF = Dense(1, activation=custom_activation, name="LF")(hidden2)
        
        outputadd = Dense(
            units=int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(hidden2)
        
        merge = kr.layers.concatenate([outputLF, outputadd])
        
        hidden3 = Dense(
            units=int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(merge)
        
        fourier_layer2 = FourierLayer(output_dim=int(params["nodes"]))(hidden2)

        hidden4 = Dense(
            units=int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"]
        )(fourier_layer2)
        
        outputHF = Dense(1, activation="sigmoid", name="HF")(hidden4)
        output = [outputHF, outputLF]
        
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params["opt"], params["lr"])
        model.compile(
            loss=custom_loss,
            loss_weights=[params["alpha"], 1 - params["alpha"]],
            optimizer=opti
        )
        return model

    # Default model compilation for all other architectures
    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params["opt"], params["lr"])
    model.compile(loss="mse", optimizer=opti, metrics=["mse"])
    
    return model