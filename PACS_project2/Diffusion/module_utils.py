from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
from keras.models import save_model
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
import pandas as pd

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




def save_performance(base_dir, r2_HF_df, mse_HF_df, r2_LF_df, mse_LF_df, U_HF_list, U_LF_list):
    # Create the directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # File paths
    r2_HF_path = os.path.join(base_dir, "r2_HF_lhs.txt")
    mse_HF_path = os.path.join(base_dir, "mse_HF_lhs.txt")
    r2_LF_path = os.path.join(base_dir, "r2_LF_lhs.txt")
    mse_LF_path = os.path.join(base_dir, "mse_LF_lhs.txt")
    U_HF_list_path = os.path.join(base_dir, "U_HF_list.data")
    U_LF_list_path = os.path.join(base_dir, "U_LF_list.data")

    # Save dataframes to files
    r2_HF_df.to_csv(r2_HF_path, header=True, index=False, sep="\t", mode="a")
    mse_HF_df.to_csv(mse_HF_path, header=True, index=False, sep="\t", mode="a")
    r2_LF_df.to_csv(r2_LF_path, header=True, index=False, sep="\t", mode="a")
    mse_LF_df.to_csv(mse_LF_path, header=True, index=False, sep="\t", mode="a")

    # Save lists to binary files
    with open(U_HF_list_path, "wb") as hf_file:
        pickle.dump(U_HF_list, hf_file)

    with open(U_LF_list_path, "wb") as lf_file:
        pickle.dump(U_LF_list, lf_file)


def save_performance(base_dir, r2_HF_df, mse_HF_df, r2_LF_df, mse_LF_df, U_HF_list, U_LF_list):
    # Save High-Frequency data
    save_single_performance(base_dir, r2_HF_df, mse_HF_df, U_HF_list, 
                            "r2_HF_lhs.txt", "mse_HF_lhs.txt", "U_HF_list.data")

    # Save Low-Frequency data
    save_single_performance(base_dir, r2_LF_df, mse_LF_df, U_LF_list, 
                            "r2_LF_lhs.txt", "mse_LF_lhs.txt", "U_LF_list.data")
    
def save_single_performance(base_dir, r2_df, mse_df, list_data, r2_filename, mse_filename, list_filename):
    # Create the directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # File paths
    r2_path = os.path.join(base_dir, r2_filename)
    mse_path = os.path.join(base_dir, mse_filename)
    list_path = os.path.join(base_dir, list_filename)

    # Save dataframes to files
    r2_df.to_csv(r2_path, header=True, index=False, sep="\t", mode="a")
    mse_df.to_csv(mse_path, header=True, index=False, sep="\t", mode="a")

    # Save list to binary file
    with open(list_path, "wb") as list_file:
        pickle.dump(list_data, list_file)



def create_folder(folder_name):
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    return folder_path

def shuffle_and_select(data, target, n_samples):
    perm = np.random.permutation(len(data))
    return data[perm][:n_samples], target[perm][:n_samples]



# def plot_results(reaction_test, reaction_train, U_test, U_train, U_pred, model_type, color1, color2, label1, label2):
#     """
#     Plot results comparing the ground truth, training data, and predictions.
#     """
#     plt.figure()
    
#     # Ensure U_train is flattened to match the dimension of reaction_test[:, 0]
#     plt.plot(reaction_test[:, 0], U_test, color=color1, linestyle="--", linewidth=2.5, label=label1)
#     plt.plot(reaction_train[:, 0], U_train, "o", markersize=6, color=color1, alpha=0.8, label=f"{model_type} training points")
#     plt.plot(reaction_test[:, 0], U_pred, color=color2, linestyle="-", linewidth=3, label=label2)
    
#     plt.legend(prop={"size": 9}, loc="best", frameon=True, fancybox=False, shadow=False, facecolor="white", edgecolor="black")
#     plt.grid(True, which='both', linestyle=':', linewidth=0.5)
#     plt.show()

def plot_results(reaction_test, U_test, U_pred, model_type,
                 reaction_train=None, U_train=None, color1='blue', color2='red', label1='Ground Truth', label2='Prediction'):
    """
    Plot results comparing the ground truth, training data (if provided), and predictions.
    
    Parameters:
    - reaction_test: ndarray, test reaction data.
    - U_test: ndarray, ground truth for test data.
    - U_pred: ndarray, predicted values for test data.
    - model_type: str, the type of model used (used for labeling training points).
    - color1: str, color for the ground truth line and training points. Default is 'blue'.
    - color2: str, color for the prediction line. Default is 'red'.
    - label1: str, label for the ground truth line. Default is 'Ground Truth'.
    - label2: str, label for the prediction line. Default is 'Prediction'.
    - reaction_train: ndarray, optional, training reaction data. Default is None.
    - U_train: ndarray, optional, ground truth for training data. Default is None.
    """
    plt.figure()
    
    # Plot the test data and predictions
    plt.plot(reaction_test[:, 0], U_test, color=color1, linestyle="--", linewidth=2.5, label=label1)
    plt.plot(reaction_test[:, 0], U_pred, color=color2, linestyle="-", linewidth=3, label=label2)
    
    # Plot the training data if provided
    if reaction_train is not None and U_train is not None:
        plt.plot(reaction_train[:, 0], U_train, "o", markersize=6, color=color1, alpha=0.8, label=f"{model_type} training points")
    
    # Configure the legend and grid
    plt.legend(prop={"size": 9}, loc="best", frameon=True, fancybox=False, shadow=False, facecolor="white", edgecolor="black")
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.show()



def update_results(df, discretization, diffusion, value, metric):
    new_entry = {'Discretization': discretization, 'diffusion': diffusion, metric: value}
    return pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)





def select_random_data(reaction_data, U_data, n_samples):
    """
    Select a random subset of the data for training or validation.
    """
    permutation = np.random.permutation(len(reaction_data))
    return reaction_data[permutation][:n_samples], U_data[permutation][:n_samples]

def augment_with_sin(reaction_data):
    """
    Augment the reaction data with a sinusoidal transformation.
    """
    return np.c_[reaction_data, np.abs(np.sin(5 * np.pi * reaction_data[:, 0] - 5 * np.pi / 6))]

# def evaluate_and_plot_network_1(model, reaction_LF_test_original, U_LF_test_original, reaction_LF, U_train_LF, test_mse_LF_list, r2_LF_list):
#     """
#     Evaluate the first network in the model and plot the results.
#     """
#     ULF = model.model_list[0].prediction(reaction_LF_test_original)
#     print("Low fidelity NN")
    
#     test_mse_LF, r2_LF = model.model_list[0].performance(reaction_LF_test_original, U_LF_test_original)
#     test_mse_LF_list.append(test_mse_LF)
#     r2_LF_list.append(r2_LF)
    
#     plt.figure()
#     plt.plot(reaction_LF_test_original[:, 0], U_LF_test_original, color="#1F77B4", linestyle="--", linewidth=2.5, label="LF model")
#     plt.plot(reaction_LF[:, 0], U_train_LF, "o", markersize=6, color="#1F77B4", alpha=0.8, label="LF training points")
#     plt.plot(reaction_LF_test_original[:, 0], ULF, color="#2CA02C", linestyle="-", linewidth=3, label="Predicted LF model")
#     plt.legend(prop={"size": 9}, loc="best", frameon=True, fancybox=False, shadow=False, facecolor="white", edgecolor="black")
#     plt.grid(True, which='both', linestyle=':', linewidth=0.5)
#     plt.show()

# def evaluate_and_plot_network_2(model, reaction_HF_test_original, U_HF_test_original, reaction_HF, U_HF, input_per_train, test_mse_per_list, r2_per_list):
#     """
#     Evaluate the second network in the model and plot the results.
#     """
#     input_per = np.concatenate((reaction_HF_test_original, model.model_list[0].prediction(reaction_HF_test_original).reshape(-1, 1)), axis=1)
#     input_per_train = np.concatenate((reaction_HF, model.model_list[0].prediction(reaction_HF).reshape(-1, 1)), axis=1)

#     print("Second model")
#     test_mse_per, r2_per = model.model_list[1].performance(input_per, U_HF_test_original)
#     test_mse_per_list.append(test_mse_per)
#     r2_per_list.append(r2_per)

#     plt.figure()
#     plt.plot(reaction_HF_test_original[:, 0], U_HF_test_original, color="#9467BD", linestyle="-", linewidth=2.5, label="HF model")
#     plt.plot(input_per_train[:, 0], U_HF, "o", markersize=6, color="#9467BD", alpha=0.8, label="HF training points")
#     plt.plot(reaction_LF_test_original[:, 0], U_LF_test_original, color="#1F77B4", linestyle="--", linewidth=2.5, label="LF model")
#     plt.plot(input_per[:, 0], model.model_list[1].prediction(input_per), color="#D62728", linestyle="-", linewidth=3, label="Predicted PER model")
#     plt.legend(prop={"size": 9}, loc="best", frameon=True, fancybox=False, shadow=False, facecolor="white", edgecolor="black")
#     plt.grid(True, which='both', linestyle=':', linewidth=0.5)
#     plt.show()

# def evaluate_and_plot_network_3(model, reaction_HF_test_original, U_HF_test_original, reaction_HF, U_HF, input_HF_train, test_mse_HF_list, r2_HF_list):
#     """
#     Evaluate the third network in the model and plot the results.
#     """
#     input_HF = np.concatenate((reaction_HF_test_original, model.model_list[0].prediction(reaction_HF_test_original).reshape(-1, 1)), axis=1)
#     input_HF = np.concatenate((input_HF, model.model_list[1].prediction(input_HF).reshape(-1, 1)), axis=1)

#     input_HF_train = np.concatenate((reaction_HF, model.model_list[0].prediction(reaction_HF).reshape(-1, 1)), axis=1)
#     input_HF_train = np.concatenate((input_HF_train, model.model_list[1].prediction(input_HF_train).reshape(-1, 1)), axis=1)

#     print("Third model")
#     test_mse_HF, r2_HF = model.performance(reaction_HF_test_original, U_HF_test_original)
#     test_mse_HF_list.append(test_mse_HF)
#     r2_HF_list.append(r2_HF)

#     plt.figure()
#     plt.plot(reaction_HF_test_original[:, 0], U_HF_test_original, color="#FF7F0E", linestyle="-", linewidth=2.5, label="HF model")
#     plt.plot(input_HF_train[:, 0], U_HF, "o", markersize=6, color="#FF7F0E", alpha=0.8, label="HF training points")
#     plt.plot(reaction_LF_test_original[:, 0], U_LF_test_original, color="#1F77B4", linestyle="--", linewidth=2.5, label="LF model")
#     plt.plot(input_HF[:, 0], model.model_list[2].prediction(input_HF), color="#2CA02C", linestyle="-", linewidth=3, label="Predicted HF model")
#     plt.legend(prop={"size": 9}, loc="best", frameon=True, fancybox=False, shadow=False, facecolor="white", edgecolor="black")
#     plt.grid(True, which='both', linestyle=':', linewidth=0.5)
#     plt.show()



def evaluate_network(model, network_index, reaction_test_original, U_test_original, reaction_train, U_train, test_mse_list, r2_list):
    """
    Evaluate the network at a specified index in the model and plot the results.
    
    Parameters:
    - model: The model containing the list of networks.
    - network_index: Index of the network in the model's network list (0-based).
    - reaction_test_original: Original test input data.
    - U_test_original: Original test output data.
    - reaction_train: Original training input data.
    - U_train: Original training output data.
    - test_mse_list: List to append the test MSE values.
    - r2_list: List to append the R^2 values.
    """
    
    # Prepare input data for the current network stage
    if network_index == 0:
        input_test = reaction_test_original
        input_train = reaction_train
    else:
        # Concatenate previous network predictions as additional inputs
        input_test = np.concatenate([reaction_test_original] + [model.model_list[i].prediction(reaction_test_original).reshape(-1, 1) for i in range(network_index)], axis=1)
        input_train = np.concatenate([reaction_train] + [model.model_list[i].prediction(reaction_train).reshape(-1, 1) for i in range(network_index)], axis=1)

    # Evaluate the current network
    print(f"Evaluating network {network_index + 1}")
    test_mse, r2 = model.model_list[network_index].performance(input_test, U_test_original)
    test_mse_list.append(test_mse)
    r2_list.append(r2)

    # Plotting
    plt.figure()
    plt.plot(reaction_test_original[:, 0], U_test_original, color="#1F77B4", linestyle="--", linewidth=2.5, label="True Test Data")
    plt.plot(input_train[:, 0], U_train, "o", markersize=6, color="#FF7F0E", alpha=0.8, label="Training Points")
    plt.plot(reaction_test_original[:, 0], model.model_list[network_index].prediction(input_test), color="#2CA02C", linestyle="-", linewidth=3, label=f"Predicted Model {network_index + 1}")
    plt.legend(prop={"size": 9}, loc="best", frameon=True, fancybox=False, shadow=False, facecolor="white", edgecolor="black")
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.show()




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