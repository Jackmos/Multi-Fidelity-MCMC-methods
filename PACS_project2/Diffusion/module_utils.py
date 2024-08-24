from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
from keras.models import save_model
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
import pandas as pd
import pickle

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
from typing import Tuple, List, Optional

from abc import ABCMeta, abstractstaticmethod, abstractmethod




def save_performance(base_dir: str, 
                     r2_HF_df: pd.DataFrame, 
                     mse_HF_df: pd.DataFrame, 
                     r2_LF_df: pd.DataFrame, 
                     mse_LF_df: pd.DataFrame, 
                     U_HF_list: List[float], 
                     U_LF_list: List[float]) -> None:
    """
    Saves performance metrics and lists to specified files.
    
    Parameters:
    - base_dir: Directory where the files will be saved.
    - r2_HF_df: DataFrame containing high-fidelity R2 values.
    - mse_HF_df: DataFrame containing high-fidelity MSE values.
    - r2_LF_df: DataFrame containing low-fidelity R2 values.
    - mse_LF_df: DataFrame containing low-fidelity MSE values.
    - U_HF_list: List of high-fidelity U values.
    - U_LF_list: List of low-fidelity U values.
    """
    # Save High-Fidelity data
    save_single_performance(base_dir, r2_HF_df, mse_HF_df, U_HF_list, 
                            "r2_HF_lhs.txt", "mse_HF_lhs.txt", "U_HF_list.data")

    # Save Low-Fidelity data
    save_single_performance(base_dir, r2_LF_df, mse_LF_df, U_LF_list, 
                            "r2_LF_lhs.txt", "mse_LF_lhs.txt", "U_LF_list.data")
    

def save_single_performance(base_dir: str, 
                            r2_df: pd.DataFrame, 
                            mse_df: pd.DataFrame, 
                            list_data: List[float], 
                            r2_filename: str, 
                            mse_filename: str, 
                            list_filename: str) -> None:
    """
    Saves a single set of performance metrics and a list to specified files.
    
    Parameters:
    - base_dir: Directory where the files will be saved.
    - r2_df: DataFrame containing R2 values.
    - mse_df: DataFrame containing MSE values.
    - list_data: List of values to be saved in binary format.
    - r2_filename: Filename for saving the R2 DataFrame.
    - mse_filename: Filename for saving the MSE DataFrame.
    - list_filename: Filename for saving the list in binary format.
    """
    # Create the directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # File paths
    r2_path = os.path.join(base_dir, r2_filename)
    mse_path = os.path.join(base_dir, mse_filename)
    list_path = os.path.join(base_dir, list_filename)

    # Save DataFrames to files
    r2_df.to_csv(r2_path, header=True, index=False, sep="\t", mode="a")
    mse_df.to_csv(mse_path, header=True, index=False, sep="\t", mode="a")

    # Save list to binary file
    with open(list_path, "wb") as list_file:
        pickle.dump(list_data, list_file)



def create_folder(folder_name: str) -> str:
    """
    Creates a folder in the current working directory if it doesn't exist.
    
    Parameters:
    - folder_name: Name of the folder to be created.
    
    Returns:
    - folder_path: Path to the created or existing folder.
    """
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    return folder_path


#@jit(nopython=True)
def shuffle_and_select(data: np.ndarray, 
                       target: np.ndarray, 
                       n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffles the data and target arrays, and selects a subset of samples.
    
    Parameters:
    - data: Array of data samples.
    - target: Array of target values.
    - n_samples: Number of samples to select.
    
    Returns:
    - A tuple of selected data and target arrays.
    """
    assert len(data) == len(target), "Data and target arrays must have the same length"
    perm = np.random.permutation(len(data))
    return data[perm][:n_samples], target[perm][:n_samples]


def plot_results(reaction_test: np.ndarray, 
                 U_test: np.ndarray, 
                 U_pred: np.ndarray, 
                 model_type: str,
                 reaction_train: Optional[np.ndarray] = None, 
                 U_train: Optional[np.ndarray] = None, 
                 color1: str = 'blue', 
                 color2: str = 'red', 
                 label1: str = 'Solution', 
                 label2: str = 'Prediction') -> None:
    """
    Plot results comparing the true solution, training data (if provided), and predictions.
    
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


def update_results(df: pd.DataFrame, 
                   discretization: str, 
                   diffusion: float, 
                   value: float, 
                   metric: str) -> pd.DataFrame:
    """
    Update the DataFrame with new results.
    
    Parameters:
    - df: The DataFrame to update.
    - discretization: Discretization method used.
    - diffusion: Diffusion coefficient.
    - value: The metric value to add.
    - metric: The name of the metric to update.
    
    Returns:
    - Updated DataFrame with the new entry added.
    """
    new_entry = {'Discretization': discretization, 'diffusion': diffusion, metric: value}
    return pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)


def select_random_data(reaction_data: np.ndarray, 
                       U_data: np.ndarray, 
                       n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a random subset of the data for training or validation.
    
    Parameters:
    - reaction_data: ndarray, full dataset of reaction data.
    - U_data: ndarray, full dataset of U values corresponding to the reaction data.
    - n_samples: Number of samples to select.
    
    Returns:
    - Tuple containing the randomly selected reaction data and corresponding U values.
    """
    permutation = np.random.permutation(len(reaction_data))
    return reaction_data[permutation][:n_samples], U_data[permutation][:n_samples]

def augment_with_sin(reaction_data: np.ndarray) -> np.ndarray:
    """
    Augment the reaction data with a sinusoidal transformation. 
    Used to improve Neural NEtwork regression performance
    
    Parameters:
    - reaction_data: ndarray, the input reaction data to augment.
    
    Returns:
    - Augmented ndarray with an additional sinusoidal feature.
    """
    return np.c_[reaction_data, np.abs(np.sin(5 * np.pi * reaction_data[:, 0] - 5 * np.pi / 6))]



def evaluate_network(model: Any, 
                     network_index: int, 
                     reaction_test_original: np.ndarray, 
                     U_test_original: np.ndarray, 
                     reaction_train: np.ndarray, 
                     U_train: np.ndarray, 
                     test_mse_list: List[float], 
                     r2_list: List[float]) -> None:
    """
    Evaluate the network at a specified index in the model and plot the results.
    
    Parameters:
    - model: The model containing the list of networks.
    - network_index: Index of the network in the model's network list (0-based).
    - reaction_test_original: Original test input data (ndarray).
    - U_test_original: Original test output data (ndarray).
    - reaction_train: Original training input data (ndarray).
    - U_train: Original training output data (ndarray).
    - test_mse_list: List to append the test MSE values (List[float]).
    - r2_list: List to append the R^2 values (List[float]).
    """
    
    # Prepare input data for the current network stage
    if network_index == 0:
        input_test = reaction_test_original
        input_train = reaction_train
    else:
        # Concatenate previous network predictions as additional inputs
        input_test = np.concatenate(
            [reaction_test_original] + [model.model_list[i].prediction(reaction_test_original).reshape(-1, 1) 
                                        for i in range(network_index)], axis=1)
        input_train = np.concatenate(
            [reaction_train] + [model.model_list[i].prediction(reaction_train).reshape(-1, 1) 
                                for i in range(network_index)], axis=1)

    # Evaluate the current network
    print(f"Evaluating network {network_index + 1}")
    test_mse, r2 = model.model_list[network_index].performance(input_test, U_test_original)
    test_mse_list.append(test_mse)
    r2_list.append(r2)

    # Plotting
    plt.figure()
    plt.plot(reaction_test_original[:, 0], U_test_original, color="#1F77B4", linestyle="--", linewidth=2.5, label="True Test Data")
    plt.plot(input_train[:, 0], U_train, "o", markersize=6, color="#FF7F0E", alpha=0.8, label="Training Points")
    plt.plot(reaction_test_original[:, 0], model.model_list[network_index].prediction(input_test), 
             color="#2CA02C", linestyle="-", linewidth=3, label=f"Predicted Model {network_index + 1}")
    plt.legend(prop={"size": 9}, loc="best", frameon=True, fancybox=False, shadow=False, 
               facecolor="white", edgecolor="black")
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.show()




class FourierLayer(Layer):
    def __init__(self, output_dim: int, **kwargs) -> None:
        """
        Custom Keras Layer implementing Fourier features.

        Parameters:
        - output_dim: int, the dimensionality of the output.
        """
        self.output_dim = output_dim
        super(FourierLayer, self).__init__(**kwargs)

    def build(self, input_shape: Tuple[int]) -> None:
        """
        Build the layer by initializing weights.

        Parameters:
        - input_shape: tuple, the shape of the input tensor.
        """
        self.kernel_sin = self.add_weight(name='kernel_sin',
                                          shape=(self.output_dim,),  
                                          initializer='glorot_uniform',
                                          trainable=True)
        self.kernel_cos = self.add_weight(name='kernel_cos',
                                          shape=(self.output_dim,),   
                                          initializer='glorot_uniform',
                                          trainable=True)
        super(FourierLayer, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Perform the forward pass and apply Fourier transformation.

        Parameters:
        - x: tf.Tensor, input tensor.

        Returns:
        - tf.Tensor, the transformed output tensor.
        """
        result = tf.sin(tf.multiply(x, self.kernel_sin)) + tf.cos(tf.multiply(x, self.kernel_cos))
        return result

    def compute_output_shape(self, input_shape: Tuple[int]) -> Tuple[int]:
        """
        Compute the output shape of the layer.

        Parameters:
        - input_shape: tuple, the shape of the input tensor.

        Returns:
        - tuple, the shape of the output tensor.
        """
        return input_shape



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


def sinusoidal_activation(x: K.tensor) -> K.tensor:
    """
    Custom sinusoidal activation function.

    Parameters:
    - x: K.tensor, input tensor.

    Returns:
    - K.tensor, the output tensor after applying the sinusoidal activation.
    """
    return K.square(K.sin(x))


def normalization(x: np.ndarray) -> np.ndarray:
    """
    Normalizes the input array to the range [0, 1].

    Parameters:
    - x: np.ndarray, input array.

    Returns:
    - np.ndarray, normalized array.
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))

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
              noise_std_output: np.ndarray, 
              data: np.ndarray, 
              output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adds Gaussian noise to the input data and output, concatenating the noisy versions 
    to the original arrays.

    Args:
        noise_std_data (np.ndarray): Standard deviations for noise to be added to the data.
        noise_std_output (np.ndarray): Standard deviations for noise to be added to the output.
        data (np.ndarray): The original data array.
        output (np.ndarray): The original output array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the noisy data and output arrays.
    """
    noisy_output = output.copy()  # Initialize with the original output
    noisy_data = data.copy()      # Initialize with the original data

    # Add Gaussian noise to the data and output based on provided standard deviations
    for std_data, std_output in zip(noise_std_data, noise_std_output):
        # Generate Gaussian noise for the output
        output_noise = np.random.normal(0, std_output, output.shape)
        # Generate Gaussian noise for the data
        data_noise = np.random.normal(0, std_data, data.shape)
        
        # Add the noise to the original data and output
        noisy_output = np.concatenate((noisy_output, output + output_noise), axis=0)
        noisy_data = np.concatenate((noisy_data, data + data_noise), axis=0)

    return noisy_output, noisy_data




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