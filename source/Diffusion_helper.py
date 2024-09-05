import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import h5py
import inspect

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout


import keras as kr
import os

from typing import Tuple, List, Optional,Any
from utils.helper_functions import Helpers_NN
from utils.functions_to_ray import FourierLayer, Activations

class Diffusion_model_helpers:

    @staticmethod
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


    @staticmethod
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
        Diffusion_model_helpers.save_single_performance(base_dir, r2_HF_df, mse_HF_df, U_HF_list, 
                                "r2_HF_lhs.txt", "mse_HF_lhs.txt", "U_HF_list.data")

        # Save Low-Fidelity data
        Diffusion_model_helpers.save_single_performance(base_dir, r2_LF_df, mse_LF_df, U_LF_list, 
                                "r2_LF_lhs.txt", "mse_LF_lhs.txt", "U_LF_list.data")
        

    @staticmethod
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



    @staticmethod
    def plot_results(reaction_test: np.ndarray, 
                    U_test: np.ndarray, 
                    U_pred: np.ndarray, 
                    model_type: str,
                    reaction_train: Optional[np.ndarray] = None, 
                    U_train: Optional[np.ndarray] = None, 
                    color1: str = 'blue', 
                    color2: str = 'red', 
                    label1: str = 'Solution', 
                    label2: str = 'Prediction',
                    save_dir: str = '',  # Default to empty string, meaning the caller's directory
                    filename: str = 'plot.png') -> None:
        """
        Plot results comparing the true solution, training data (if provided), and predictions,
        and save the plot to the caller's directory or a specified directory.
        
        Parameters:
        - reaction_test: ndarray, test reaction data.
        - U_test: ndarray, ground truth for test data.
        - U_pred: ndarray, predicted values for test data.
        - model_type: str, the type of model used (used for labeling training points).
        - color1: str, color for the ground truth line and training points. Default is 'blue'.
        - color2: str, color for the prediction line. Default is 'red'.
        - label1: str, label for the ground truth line. Default is 'Solution'.
        - label2: str, label for the prediction line. Default is 'Prediction'.
        - reaction_train: ndarray, optional, training reaction data. Default is None.
        - U_train: ndarray, optional, ground truth for training data. Default is None.
        - save_dir: str, directory to save the plot. Default is the caller's directory.
        - filename: str, name of the saved plot file. Default is 'plot.png'.
        """
        # Get the directory of the file that called this function
        caller_frame = inspect.stack()[1]
        caller_module = inspect.getmodule(caller_frame[0])
        caller_dir = os.path.dirname(os.path.abspath(caller_module.__file__))

        # Determine the save path
        if save_dir:
            save_path = os.path.join(save_dir, filename)
        else:
            save_path = os.path.join(caller_dir, filename)  # Save in the caller's directory

        # Plot the test data and predictions
        plt.figure()
        plt.plot(reaction_test[:, 0], U_test, color=color1, linestyle="--", linewidth=2.5, label=label1)
        plt.plot(reaction_test[:, 0], U_pred, color=color2, linestyle="-", linewidth=3, label=label2)
        
        # Plot the training data if provided
        if reaction_train is not None and U_train is not None:
            plt.plot(reaction_train[:, 0], U_train, "o", markersize=6, color=color1, alpha=0.8, label=f"{model_type} training points")
        
        # Configure the legend and grid
        plt.legend(prop={"size": 9}, loc="best", frameon=True, fancybox=False, shadow=False, facecolor="white", edgecolor="black")
        plt.grid(True, which='both', linestyle=':', linewidth=0.5)
        
        # Save the plot
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory

        print(f"Plot saved to {save_path}")



    @staticmethod
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



    @staticmethod
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
        new_entry = {'Discretization': discretization, 'Diffusion': diffusion, metric: value}
        return pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    @staticmethod
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

    @staticmethod
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


    @staticmethod
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



    @staticmethod
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
                activation=Activations.custom_activation,
                kernel_initializer=params["kernel_init"],
                kernel_regularizer=l2(0.001)
            )(inputs)
            
            hidden2 = Dense(
                units=64,
                activation=Activations.custom_activation,
                kernel_initializer=params["kernel_init"],
                kernel_regularizer=l2(0.001)
            )(hidden1)
            
            fourier_layer1 = FourierLayer(output_dim=64)(hidden2)
            
            hidden3 = Dense(
                units=64,
                activation=Activations.custom_activation,
                kernel_initializer=params["kernel_init"],
                kernel_regularizer=l2(0.001)
            )(fourier_layer1)
            hidden3 = Dropout(0.05)(hidden3)
            
            hidden4 = Dense(
                units=64,
                activation=Activations.custom_activation,
                kernel_initializer=params["kernel_init"],
                kernel_regularizer=l2(0.001)
            )(hidden3)
            
            output = Dense(units=num_outputs, activation="linear", name="LF")(hidden4)

        elif name == "HF":
            # Model architecture for the "HF" model
            hidden1 = Dense(
                units=int(params["nodes"]),
                activation=Activations.custom_activation,
                kernel_regularizer=l2(params["l2weight"]),
                kernel_initializer=params["kernel_init"]
            )(inputs)
            hidden1 = Dropout(0.05)(hidden1)
            
            fourier_layer = FourierLayer(output_dim=int(params["nodes"]))(hidden1)
            
            hidden2 = Dense(
                units=int(params["nodes"]),
                activation=Activations.custom_activation,
                kernel_regularizer=l2(params["l2weight"]),
                kernel_initializer=params["kernel_init"]
            )(fourier_layer)
            
            output = Dense(units=num_outputs, activation="linear", name="HF")(hidden2)

        elif name == "Single":
            # Model architecture for the "Single" model
            hidden1 = Dense(
                units=64,
                activation=Activations.custom_activation,
                kernel_initializer=params["kernel_init"],
                kernel_regularizer=l2(params["l2weight"])
            )(inputs)
            
            hidden2 = Dense(
                units=64,
                activation=Activations.custom_activation,
                kernel_initializer=params["kernel_init"],
                kernel_regularizer=l2(params["l2weight"])
            )(hidden1)
            
            hidden3 = Dense(
                units=64,
                activation=Activations.custom_activation,
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
                activation=Activations.custom_activation,
                kernel_regularizer=l2(params["l2weight"]),
                kernel_initializer=params["kernel_init"]
            )(inputs)
            
            output = Dense(units=num_outputs, activation="sigmoid", name="HFlin")(hiddenlin)

        elif name == "Hfper":
            # Model architecture for the "Hfper" model
            hiddenper = Dense(
                units=64,
                activation=Activations.sinusoidal_activation,
                kernel_regularizer=l2(params["l2weight"]),
                kernel_initializer=params["kernel_init"]
            )(inputs)
            
            hiddenper2 = Dense(
                units=64,
                activation=Activations.sinusoidal_activation,
                kernel_regularizer=l2(params["l2weight"]),
                kernel_initializer=params["kernel_init"]
            )(hiddenper)
            
            output = Dense(units=num_outputs, activation="linear", name="HFper")(hiddenper2)


        elif name == "Inter":
            # Model architecture for the "Inter" model
            hidden1 = Dense(
                units=64,
                activation=Activations.custom_activation,
                kernel_initializer=params["kernel_init"]
            )(inputs)
            
            fourier_layer1 = FourierLayer(output_dim=64)(hidden1)
            
            hidden2 = Dense(
                units=64,
                activation=Activations.custom_activation,
                kernel_initializer=params["kernel_init"]
            )(fourier_layer1)
            
            outputLF = Dense(1, activation=Activations.custom_activation, name="LF")(hidden2)
            
            outputadd = Dense(
                units=int(params["nodes"]),
                activation=Activations.custom_activation,
                kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
                kernel_initializer=params["kernel_init"]
            )(hidden2)
            
            merge = kr.layers.concatenate([outputLF, outputadd])
            
            hidden3 = Dense(
                units=int(params["nodes"]),
                activation=Activations.custom_activation,
                kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
                kernel_initializer=params["kernel_init"]
            )(merge)
            
            fourier_layer2 = FourierLayer(output_dim=int(params["nodes"]))(hidden2)

            hidden4 = Dense(
                units=int(params["nodes"]),
                activation=Activations.custom_activation,
                kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
                kernel_initializer=params["kernel_init"]
            )(fourier_layer2)
            
            outputHF = Dense(1, activation="sigmoid", name="HF")(hidden4)
            output = [outputHF, outputLF]
            
            model = Model(inputs=inputs, outputs=output)
            opti = Helpers_NN.getOpti(params["opt"], params["lr"])
            model.compile(
                loss=Helpers_NN.custom_loss,
                loss_weights=[params["alpha"], 1 - params["alpha"]],
                optimizer=opti
            )
            return model

        # Default model compilation for all other architectures
        model = Model(inputs=inputs, outputs=output)
        opti = Helpers_NN.getOpti(params["opt"], params["lr"])
        model.compile(loss="mse", optimizer=opti, metrics=["mse"])
        
        return model


