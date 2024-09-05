import os
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Callable
import matplotlib.lines as mlines
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from utils.helper_functions import Helpers_NN,Clean
"""
Contains the definition of the Benchmark cases and the loop to investigate parameters 
to solve the inverse problem.
"""

class Benchmark_functions:
    def __init__(self, case: str) -> None:
        """
        Initializes the FidelityFunctionModified object with a specified case.

        :param case: The case to use ('Basic', 'Discontinuous', or 'Oscillatory').
        """
        self.cases = {
            "Basic": self.create_basic_functions,
            "Discontinuous": self.create_discontinuous_functions,
            "Oscillatory": self.create_oscillatory_functions,
            "Basic_regression": self.create_basic_functions,
            "Discontinuous_regression": self.create_discontinuous_functions,
            "Oscillatory_regression": self.create_oscillatory_functions
        }

        if case not in self.cases:
            raise ValueError(f"Example {case} not recognized")

        self.modified_highfid, self.modified_lowfid = self.cases[case]()
        self.data = self.get_parameters(case)
        self.case=case

    @staticmethod
    def create_basic_functions() -> Tuple[Callable, Callable]:
        """
        Creates the high-fidelity and low-fidelity functions for the Basic case.
        
        :return: Tuple of modified high fidelity and low fidelity functions.
        """
        @jit(nopython=True)
        def modified_highfid(x: np.ndarray, delta: float) -> np.ndarray:
            period = 5.54
            phase_within_period = np.mod(x + delta / 6, period)
            return (6. * phase_within_period / 5 - 2.) ** 2 * np.sin(12. * phase_within_period / 5 - 4.)

        @jit(nopython=True)
        def modified_lowfid(x: np.ndarray, delta: float) -> np.ndarray:
            period2 = 5.96
            phase_within_period = np.mod(x + 0.2 + delta / 6, period2)
            return 0.5 * (6. * phase_within_period / 5 - 2.) ** 2 * np.sin(12. * phase_within_period / 5 - 4.) + 10 * (phase_within_period / 5 - 0.5) + 5.

        return modified_highfid, modified_lowfid

    @staticmethod
    def create_discontinuous_functions() -> Tuple[Callable, Callable]:
        """
        Creates the high-fidelity and low-fidelity functions for the Discontinuous case.
        
        :return: Tuple of modified high fidelity and low fidelity functions.
        """
        @jit(nopython=True)
        def modified_highfid(x: np.ndarray, delta: float) -> np.ndarray:
            lowfid_val = modified_lowfid(x, delta)
            return (2 * lowfid_val - 20 * x / 5 + 20) * (x / 5 < 0.5) + \
                   (4 + 2 * lowfid_val - 20 * x / 5 + 20 + delta) * (x / 5 > 0.5)

        @jit(nopython=True)
        def modified_lowfid(x: np.ndarray, delta: float) -> np.ndarray:
            return (0.5 * (6. * x / 5 - 2.) ** 2 * np.sin(12. * x / 5 - 4) + 
                    10 * (x / 5 - 0.5) - 5.) * (x < 2.5) + \
                   (3 + 0.5 * (6. * x / 5 - 2.) ** 2 * np.sin(12. * x / 5 - 4) + 
                    10 * (x / 5 - 0.5) - 5. + delta) * (x > 2.5)

        return modified_highfid, modified_lowfid

    @staticmethod
    def create_oscillatory_functions() -> Tuple[Callable, Callable]:
        """
        Creates the high-fidelity and low-fidelity functions for the Oscillatory case.
        
        :return: Tuple of modified high fidelity and low fidelity functions.
        """
        @jit(nopython=True)
        def modified_highfid(x: np.ndarray, delta: float) -> np.ndarray:
            return (x / 5 - np.sqrt(2)) * modified_lowfid(x, delta) ** 2

        @jit(nopython=True)
        def modified_lowfid(x: np.ndarray, delta: float) -> np.ndarray:
            return np.sin(delta * x)

        return modified_highfid, modified_lowfid

    def get_parameters(self, example: str) -> Dict[str, Any]:
        """
        Returns the parameters for the specified example case.

        :param example: The case to get parameters for.
        :return: Dictionary containing parameters and evaluated function values.
        """
        if example == "Basic":
            Nhf, NhPer, Nlf, NepoLF, NepoPer, NepoHF = 50, 40, 60, 2500, 1500, 2500
            deltas = np.linspace(0., 28., 4)
            deltas_val = np.linspace(np.min(deltas), np.max(deltas), 3)
            deltas_test = np.linspace(np.min(deltas), np.max(deltas), 5)

        elif example == "Discontinuous":
            Nhf, NhPer, Nlf, NepoLF, NepoPer, NepoHF = 16, 16, 40, 2000, 1000, 5200
            deltas = np.linspace(0., 15., 5)
            deltas_val = np.linspace(np.min(deltas), np.max(deltas), 4)
            deltas_test = np.linspace(np.min(deltas), np.max(deltas), 6)

        elif example == "Oscillatory":
            Nhf, NhPer, Nlf, NepoLF, NepoPer, NepoHF = 20, 20, 64, 1000, 1000, 2000
            deltas = np.linspace(1 / 5, 6 / 5, 5) * np.pi
            deltas_val = np.linspace(np.min(deltas), np.max(deltas), 4)
            deltas_test = np.linspace(np.min(deltas), np.max(deltas), 6)
        
        elif example == "Basic_regression":
            Nhf, NhPer, Nlf, NepoLF, NepoPer, NepoHF = 50, 50, 100, 2000, 2000, 2000
            deltas = np.array([0, 10, 20])
            deltas_val = np.array([0, 10, 20])
            deltas_test = np.array([0, 10, 20])

        elif example == "Discontinuous_regression":
            Nhf, NhPer, Nlf, NepoLF, NepoPer, NepoHF = 16, 16, 40, 2000, 2000, 5200
            deltas = np.linspace(0., 15., 5)
            deltas_val = np.linspace(np.min(deltas), np.max(deltas), 5)
            deltas_test = np.linspace(np.min(deltas), np.max(deltas), 5)

        elif example == "Oscillatory_regression":
            Nhf, NhPer, Nlf, NepoLF, NepoPer, NepoHF = 15, 15, 64, 1000, 1000, 3000
            deltas = np.linspace(2 / 5, 8 / 5, 4) * np.pi
            deltas_val = np.linspace(np.min(deltas), np.max(deltas), 4) 
            deltas_test = np.linspace(np.min(deltas), np.max(deltas), 4)

        else:
            raise ValueError(f"Unsupported example type: {example}")
        
        xhf = np.linspace(0, 5, Nhf)
        xhfPer = np.linspace(0, 5, NhPer)
        xlf = np.linspace(0, 5, Nlf)

        # Prepare high fidelity data
        datahf = self._create_meshgrid(xhf, deltas)
        Yhf = self.modified_highfid(datahf[:, 0], datahf[:, 1])

        datahfPer = self._create_meshgrid(xhfPer, deltas)
        YhfPer = self.modified_highfid(datahfPer[:, 0], datahfPer[:, 1])

        # Prepare low fidelity data
        datalf = self._create_meshgrid(xlf, deltas)
        Ylf = self.modified_lowfid(datalf[:, 0], datalf[:, 1])

        return {
            "modified_highfid": self.modified_highfid,
            "modified_lowfid": self.modified_lowfid,
            "Nhf": Nhf,
            "NhPer": NhPer,
            "Nlf": Nlf,
            "xhf": datahf,
            "xhfPer": datahfPer,
            "xlf": datalf,
            "x_vallf": self._create_meshgrid(np.linspace(0, 5, Nlf), deltas_val),
            "x_valhfper": self._create_meshgrid(np.linspace(0, 5, NhPer), deltas_val),
            "x_valhf": self._create_meshgrid(np.linspace(0, 5, Nhf), deltas_val),
            "x_test": np.linspace(0, 5, 10000),
            "Yhf": Yhf,
            "YhfPer": YhfPer,
            "Ylf": Ylf,
            "dataval_lf": self.modified_lowfid(
                self._create_meshgrid(np.linspace(0, 5, Nlf), deltas_val)[:, 0],
                self._create_meshgrid(np.linspace(0, 5, Nlf), deltas_val)[:, 1]
            ),
            "dataval_hfper": self.modified_highfid(
                self._create_meshgrid(np.linspace(0, 5, NhPer), deltas_val)[:, 0],
                self._create_meshgrid(np.linspace(0, 5, NhPer), deltas_val)[:, 1]
            ),
            "dataval_hf": self.modified_highfid(
                self._create_meshgrid(np.linspace(0, 5, Nhf), deltas_val)[:, 0],
                self._create_meshgrid(np.linspace(0, 5, Nhf), deltas_val)[:, 1]
            ),
            "datatest": self._create_meshgrid(np.linspace(0, 5, 10000), deltas_test),
            "NepoLF": NepoLF,
            "NepoPer": NepoPer,
            "NepoHF": NepoHF,
            "deltas": deltas
        }

    @staticmethod
    def _create_meshgrid(x: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """
        Creates a meshgrid and reshapes it for vectorized computation.
        
        :param x: Array of x values.
        :param deltas: Array of delta values.
        :return: Reshaped meshgrid array.
        """
        meshgrid = np.array(np.meshgrid(x, deltas)).T.reshape(-1, 2)
        ord_index = np.lexsort((meshgrid[:, 0], meshgrid[:, 1]))
        return meshgrid[ord_index]

    def get_data(self) -> Dict[str, Any]:
        """
        Returns the evaluated data for the selected case.

        :return: Dictionary containing evaluated data.
        """
        return self.data

    def plot_functions(self, output_folder='tutorial_output') -> None:
        """
        Plots the high fidelity and low fidelity functions along with their data points.
        Saves the plot to a file in the specified output directory.
        """

        # Use the static method to create the directories
        full_output_dir = Clean.create_output_directory('output', output_folder)

        x_values = np.linspace(0, 5, 1000)
        deltas = self.data["deltas"]

        plt.figure(figsize=(12, 8))
        for delta in deltas:
            y_highfid = self.data["modified_highfid"](x_values, delta)
            y_lowfid = self.data["modified_lowfid"](x_values, delta)
            plt.plot(x_values, y_highfid, label=fr'High Fidelity, $\delta$={delta}', linewidth=4)
            plt.plot(x_values, y_lowfid, label=fr'Low Fidelity, $\delta$={delta}', linestyle='--', linewidth=4)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.title(f'Fidelity Functions for Various Deltas - {self.case} case')

        # Save the plot in the full_output_dir
        plot_filename = f'{self.case}_fidelity_functions.png'
        plt.savefig(os.path.join(full_output_dir, plot_filename))
        plt.close()

    def plot_detailed_functions(self, output_folder='tutorial_output') -> None:
        """
        Plots the high fidelity and low fidelity functions along with their detailed data points.
        Saves the plot to a file in the specified output directory.
        """

        # Use the static method to create the directories
        full_output_dir = Clean.create_output_directory('output', output_folder)

        deltas = self.data["deltas"]
        x_test = np.linspace(0, 5, 10000)

        plt.figure(figsize=(12, 8))

        for delta in deltas:
            datatest_segment = self._create_meshgrid(x_test, [delta])
            plt.plot(datatest_segment[:, 0],
                    self.data["modified_highfid"](datatest_segment[:, 0], datatest_segment[:, 1]),
                    'r', linewidth=4)

        for delta in deltas:
            datatest_segment = self._create_meshgrid(x_test, [delta])
            plt.plot(datatest_segment[:, 0],
                    self.data["modified_lowfid"](datatest_segment[:, 0], datatest_segment[:, 1]),
                    'g', linewidth=4)

        red_line = mlines.Line2D([], [], color='red', label='High-Fidelity Sol')
        green_line = mlines.Line2D([], [], color='green', label='Low-Fidelity Sol')

        plt.xlabel('x')
        plt.legend(handles=[red_line, green_line])
        plt.grid(True)
        plt.title(f'Benchmark 1D - {self.case} case')

        plot_filename = f'{self.case}_detailed_fidelity_functions.png'
        plt.savefig(os.path.join(full_output_dir, plot_filename))
        plt.close()


    @staticmethod
    def save_comparison_plot(data, y, img_name, output_folder='tutorial_output', fidelity_level='High'):
        """
        Generate a comparison plot between the exact and predicted high-fidelity models
        and save it to the specified output folder with the given image name.

        Parameters:
        - data: Dictionary containing 'datatest' and 'modified_highfid' functions.
        - y: Array containing the predicted high-fidelity values.
        - img_name: The name of the image file to save (e.g., 'comparison_plot.png').
        - output_folder: The folder where the image will be saved (default is 'tutorial_output').
        - fidelity_level: exact data to plot as comparison (default is 'High', alternative is 'Low')
        Returns:
        - None
        """

        # Use the static method to create the directories
        base_output_dir = 'output'
        full_output_dir = Clean.create_output_directory(base_output_dir, output_folder)

        # Plot the real and estimated relation; allows a qualitative and immediate comparison
        unique_values = np.unique(data["xlf"][:, 1])
        plt.figure()

        for val in unique_values:
            mask = data["datatest"][:, 1] == val
            x_values = data["datatest"][mask, 0]
            if fidelity_level == 'High':
                y_exact = data["modified_highfid"](x_values, val)
            else:
                y_exact = data["modified_lowfid"](x_values, val)

            y_pred = y[mask]

            plt.plot(x_values, y_exact, 'r', label=f'exact {fidelity_level} fid.' if val == unique_values[0] else "")
            plt.plot(x_values, y_pred, 'k--', label=f'pred {fidelity_level} fid.' if val == unique_values[0] else "")

        plt.legend()
        plt.title(f'{fidelity_level} Fidelity Model')

        # Save the plot to the specified output folder with the provided image name
        output_file = os.path.join(full_output_dir, img_name)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        # Optionally, close the figure to free memory
        plt.close()


    @staticmethod
    def getModel(params: Dict[str, Any], num_inputs: int, name: str, num_outputs: int) -> tf.keras.models.Model:
        """
        Create and return a compiled Keras model based on the specified parameters and model type.

        Parameters:
        - params: Dictionary containing model configuration parameters, such as 
                'nodes', 'l2weight', 'kernel_init', 'alpha', and 'opt'.
        - num_inputs: Integer specifying the number of input features.
        - name: String indicating the model type ('HF', 'LF', 'Single', 'Hflin', 'Hfper', 'GP', 'Inter').
        - num_outputs: Integer specifying the number of output features.

        Returns:
        - A compiled Keras model tailored to the specified model type.
        """
        
        # Define the input layer with the given number of input features
        inputs = Input(shape=(num_inputs,))

        # High-Fidelity (HF) Model
        if name == 'HF':
            hidden1 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2(params['l2weight']),
                            kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
            output = Dense(num_outputs, activation='linear', name='HF')(hidden1)

        # Low-Fidelity (LF) Model
        elif name == 'LF':
            hidden = inputs
            for i in range(4):
                hidden = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden{i+1}')(hidden)
            output = Dense(num_outputs, activation='linear', name='LF')(hidden)

        # Single Model
        elif name == 'Single':
            hidden = inputs
            for i in range(4):
                hidden = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'],
                            kernel_regularizer=l2(params['l2weight']), name=f'{name}_hidden{i+1}')(hidden)
            output = Dense(num_outputs, activation='linear', name='Single')(hidden)

        # High-Fidelity Linear (Hflin) Model
        elif name == 'Hflin':
            hidden = Dense(64, activation='linear', kernel_regularizer=l2(params['l2weight']),
                        kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
            output = Dense(num_outputs, activation='linear', name='HFlin')(hidden)

        # High-Fidelity Periodic (Hfper) Model
        elif name == "Hfper":
            hidden = inputs
            for i in range(2):
                hidden = Dense(64, activation=sinusoidal_activation, kernel_regularizer=l2(params["l2weight"]),
                            kernel_initializer=params["kernel_init"], name=f'{name}_hidden{i+1}')(hidden)
            output = Dense(num_outputs, activation="linear", name="HFper")(hidden)

        # Intermediate (Inter) Model
        elif name == 'Inter':
            hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden1')(inputs)
            hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], name=f'{name}_hidden2')(hidden1)
            outputLF = Dense(1, activation='linear', name='LF')(hidden2)
            outputadd = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                            kernel_initializer=params['kernel_init'], name=f'{name}_hidden3')(hidden2)
            merge = concatenate([outputLF, outputadd])
            hidden3 = Dense(params['nodes'], activation='tanh', kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']),
                            kernel_initializer=params['kernel_init'], name=f'{name}_hidden4')(merge)
            outputHF = Dense(1, activation='linear', name='HF')(hidden3)
            output = [outputHF, outputLF]
            model = Model(inputs=inputs, outputs=output)
            opti = Helpers_NN.getOpti(params['opt'], params['lr'])
            model.compile(loss=Helpers_NN.custom_loss, loss_weights=[params['alpha'], 1 - params['alpha']], optimizer=opti)
            return model

        # For other cases, create a standard model
        model = Model(inputs=inputs, outputs=output)
        opti = Helpers_NN.getOpti(params['opt'], params['lr'])
        model.compile(loss='mse', optimizer=opti, metrics=['mse'])
        return model