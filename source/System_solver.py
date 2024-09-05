import numpy as np
import h5py
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple, List
from tensorflow.keras.optimizers import Adam, Nadam, Adamax
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from utils.helper_functions import *
from utils.functions_to_ray import *
import os

class SystemSolver(ABC):
    """
    Abstract base class for solving systems of differential equations.
    """
    def __init__(self):
        """
        Initialize the solver with default values.
        """
        self.time_points = None    # Time points for the simulation
        self.y_values_list = None  # List to store the solutions for each parameter set
        self.params = None         # List of parameter sets
        self.system_solver = None  # Placeholder for the specific solver instance

    @abstractmethod
    def equations(self, t: float, y: np.ndarray, *params) -> np.ndarray:
        """
        Define the system of differential equations.
        
        Args:
        - t: Current time point
        - y: Current state of the system (array of values)
        - params: Parameters for the differential equations
        """
        pass

    def _runge_kutta_2nd_order(self, y: np.ndarray, t: float, h: float, params: Tuple) -> np.ndarray:
        """
        Perform a single step using the second-order Runge-Kutta method.

        Args:
        - y: Current state of the system (array of values)
        - t: Current time point
        - h: Time step size
        - params: Parameters for the differential equations

        Returns:
        - y_next: State of the system at the next time point
        """
        k1 = h * self.equations(t, y, *params)
        k2 = h * self.equations(t + h, y + k1, *params)
        y_next = y + 0.5 * (k1 + k2)
        return y_next

    def _euler_method_step(self, y: np.ndarray, t: float, h: float, params: Tuple) -> np.ndarray:
        """
        Perform a single step using the Euler method.

        Args:
        - y: Current state of the system (array of values)
        - t: Current time point
        - h: Time step size
        - params: Parameters for the differential equations

        Returns:
        - y_next: State of the system at the next time point
        """
        y_next = y + h * self.equations(t, y, *params)
        return y_next

    def generate_dataset(self, T: float, params: np.ndarray, h: float, fidelity: str) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple]]:
        """
        Generate a dataset of the system's evolution over time for different parameter sets.

        Args:
        - T: Total simulation time
        - params: Array of parameter sets for the differential equations
        - h: Time step size
        - fidelity: Solver method to use ('HF' for high fidelity (Runge-Kutta), otherwise Euler)

        Returns:
        - time_points: Array of time points
        - y_values_list: List of solution arrays for each parameter set
        - params: List of parameter sets
        """
        if T <= 0 or h <= 0:
            raise ValueError("T and h must be positive")

        num_steps = int(T / h)
        self.time_points = np.linspace(0, T, num_steps + 1)
        self.y_values_list = []
        self.params = params

        for param_set in params:
            if self.dim_sys is not None:
                y_initial = np.zeros(self.dim_sys)
            else:
                raise ValueError("Dimension of the system unknown")
            y_initial[:] = 0.5
            y_values = np.zeros((num_steps + 1, len(y_initial)))
            y_values[0, :] = y_initial

            for i in range(num_steps):
                if fidelity == "HF":
                    y_values[i + 1, :] = self._runge_kutta_2nd_order(y_values[i, :], self.time_points[i], h, param_set)
                else:
                    y_values[i + 1, :] = self._euler_method_step(y_values[i, :], self.time_points[i], h, param_set)

            self.y_values_list.append(y_values)

        return self.time_points, self.y_values_list, self.params


    def save_dataset(self, filename: str = 'dataset.h5'):
        """
        Save the generated dataset to an HDF5 file.

        Args:
        - filename: Name of the file to save the dataset
        """
        try:
            with h5py.File(filename, 'w') as h5file:
                h5file.create_dataset('time_points', data=np.array(self.time_points))
                param_group = h5file.create_group('params')
                for i, param_set in enumerate(self.params):
                    param_group.create_dataset(f'set_{i}', data=np.array(param_set))

                y_group = h5file.create_group('y_values_list')
                for i, y_values in enumerate(self.y_values_list):
                    y_group.create_dataset(f'set_{i}', data=y_values)
        except Exception as e:
            print(f"Error saving dataset: {e}")

    def load_dataset(self, filename: str = 'dataset.h5') -> Tuple[np.ndarray, List[np.ndarray], List[Tuple]]:
        """
        Load a dataset from an HDF5 file.

        Args:
        - filename: Name of the file to load the dataset from

        Returns:
        - time_points: Array of time points
        - y_values_list: List of solution arrays for each parameter set
        - params: List of parameter sets
        """
        try:
            with h5py.File(filename, 'r') as h5file:
                self.time_points = np.array(h5file['time_points'])

                self.params = []
                param_group = h5file['params']
                for key in param_group.keys():
                    self.params.append(tuple(param_group[key][:]))

                self.y_values_list = []
                y_group = h5file['y_values_list']
                for key in y_group.keys():
                    self.y_values_list.append(np.array(y_group[key]))
        except Exception as e:
            print(f"Error loading dataset: {e}")

        return self.time_points, self.y_values_list, self.params


    def plot_dataset(self, folder_name: str = "ODE_graphic_results") -> None:
        """
        Save the plots of the system's evolution for different parameter sets.
        Group the plots into separate figures, each containing at most 3 subplots.

        Parameters:
        - folder_name: The name of the folder where plots will be saved.
                    Default is "ODE_graphic_results".
        """
        
        # Use the create_output_directory method to ensure the output directory exists
        output_dir = Clean.create_output_directory("output", folder_name)
        
        # Sort by the first parameter (or another criterion)
        sorted_indices = np.argsort([params[0] for params in self.params])
        self.y_values_list = [self.y_values_list[i] for i in sorted_indices]
        self.params = [self.params[i] for i in sorted_indices]

        num_plots = len(self.y_values_list)
        
        # Iterate over the plots in chunks of 3
        for plot_group_start in range(0, num_plots, 3):
            # Create a new figure for each group of 3
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Ensure axes is always a list, even if there's only one subplot
            axes = np.ravel(axes)
            
            # Select the current group of plots (up to 3)
            current_group = self.y_values_list[plot_group_start:plot_group_start + 3]
            current_params = self.params[plot_group_start:plot_group_start + 3]
            
            for i, (y_values, params) in enumerate(zip(current_group, current_params)):
                ax = axes[i]
                ax.plot(self.time_points, y_values[:, 0], label=f'y1(t)', marker='o')
                if y_values.shape[1] > 1:
                    ax.plot(self.time_points, y_values[:, 1], label=f'y2(t)', marker='x')
                if y_values.shape[1] > 2:
                    ax.plot(self.time_points, y_values[:, 2], label=f'y3(t)', marker='s')

                ax.set_xlabel('Time')
                ax.set_ylabel('Values')
                ax.legend()
                ax.set_title(f'System for params={params}')
            
            # Hide any unused subplots (if the total number isn't a multiple of 3)
            for j in range(len(current_group), 3):
                fig.delaxes(axes[j])
            
            # Set layout and save the figure
            fig.suptitle(f'System of Differential Equations for Parameters {plot_group_start + 1} to {plot_group_start + len(current_group)}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust to fit the title
            
            # Save the figure to the specified folder
            plot_filename = f'{output_dir}/plot_group_{plot_group_start + 1}_to_{plot_group_start + len(current_group)}.png'
            fig.savefig(plot_filename)
            plt.close(fig)  





    @staticmethod
    def plot_systems(x_HF, y_HF, y_test_pred, final_model_trained, unique_combinations, colors_data, colors_predictions, title_suffix, output_folder):
        """
        Plots the data and model predictions for each unique combination.

        Args:
            x_HF: The input data.
            y_HF: The real output data.
            y_test_pred: The predicted output data.
            final_model_trained: The trained model object with prediction methods.
            unique_combinations: The unique combinations of the last columns in x_HF.
            colors_data: The color palette for the data plots.
            colors_predictions: The color palette for the prediction plots.
            title_suffix: The suffix to add to each plot's title.
            output_folder: The folder where the plots will be saved.
        """
        for comb in unique_combinations:
            mask = np.all(x_HF[:, 1:] == comb, axis=1)

            plt.figure()

            for i in range(y_test_pred.shape[1]):
                color_data = colors_data[i]
                color_prediction = colors_predictions[i]

                plt.plot(x_HF[mask, 0], y_HF[mask, i], label=f'System {i+1} Data', color=color_data, linestyle='-', linewidth=2)

                plt.plot(x_HF[mask, 0], final_model_trained.model_list[0].prediction(x_HF[mask])[:, i] if title_suffix == "LF Network" 
                        else final_model_trained.prediction(x_HF[mask])[:, i], 
                        label=f'System {i+1} Prediction {title_suffix}', color=color_prediction, linestyle='--' if title_suffix == "LF Network" else '-.', linewidth=2)

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Plot for Combination {comb} {title_suffix}')
            plt.legend()
            plt.grid(True)

            base_dir = 'output'
            full_output_dir = Clean.create_output_directory(base_dir, output_folder)
            plt.savefig(os.path.join(full_output_dir, f'plot_combination_{comb}_{title_suffix.replace(" ", "_")}.png'))
            plt.close()

    @staticmethod
    def plot_lf_network(x_HF, y_HF, y_test_pred, final_model_trained, output_folder='tutorial_plots'):
        """
        Plots the data and predictions for the LF Network case.

        Args:
            x_HF: The input data.
            y_HF: The real output data.
            y_test_pred: The predicted output data.
            final_model_trained: The trained model object with prediction methods.
            output_folder: The folder where the plots will be saved.
        """
        unique_combinations = np.unique(x_HF[:, 1:], axis=0)
        colors_data = plt.cm.viridis(np.linspace(0, 0.5, y_test_pred.shape[1]))
        colors_predictions = plt.cm.plasma(np.linspace(0, 1, y_test_pred.shape[1]))

        SystemSolver.plot_systems(x_HF, y_HF, y_test_pred, final_model_trained, unique_combinations, colors_data, colors_predictions, "LF Network", output_folder)

    @staticmethod
    def plot_mf_network(x_HF, y_HF, y_test_pred, final_model_trained, output_folder='tutorial_plots'):
        """
        Plots the data and predictions for the MF Network case.

        Args:
            x_HF: The input data.
            y_HF: The real output data.
            y_test_pred: The predicted output data.
            final_model_trained: The trained model object with prediction methods.
            output_folder: The folder where the plots will be saved.
        """
        unique_combinations = np.unique(x_HF[:, 1:], axis=0)
        colors = plt.cm.viridis(np.linspace(0, 1, y_test_pred.shape[1]))

        SystemSolver.plot_systems(x_HF, y_HF, y_test_pred, final_model_trained, unique_combinations, colors, colors, "MF Network", output_folder)
        








    def get_dim(self)-> int:
        """
        Obtain the dimension of the system of equations
        """
        return self.dim_sys 

    @staticmethod
    def plot_per_parameter(x, y, y_pred, fwd_surrogate, folder_name: str = "results_per_parameter") -> None:
        """
        Static method to plot and save the results of the system's evolution for different parameter sets.
        
        Parameters:
        - x: NumPy array of input data with shape (28020, 5).
        - y: NumPy array of actual system data.
        - y_pred: NumPy array of predicted system data.
        - fwd_surrogate: 3 step NN
        - folder_name: Name of the folder where plots will be saved. Defaults to 'results_per_parameter'.
        """

        # Ensure the folder is saved within the "test" directory
        folder_path = os.path.join("test", folder_name)

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Extract the last 4 columns for combination filtering
        combinations = x[:, 1:]  # Taking the last 4 columns
        
        # Find unique combinations
        unique_combinations = np.unique(combinations, axis=0)

        # Generate two different color palettes
        colors_data = plt.cm.viridis(np.linspace(0, 0.5, y_pred.shape[1]))
        colors_predictions = plt.cm.plasma(np.linspace(0, 1, y_pred.shape[1]))

        # Iterate over each unique combination
        for comb in unique_combinations:
            # Filter the rows that match the current combination
            mask = np.all(combinations == comb, axis=1)
            filtered_data = x[mask]

            # Create a new figure for each combination
            plt.figure()
            
            # Plot each system's data and corresponding predictions
            for i in range(y_pred.shape[1]):
                color_data = colors_data[i]
                color_prediction = colors_predictions[i]

                # Plot the actual system data
                plt.plot(x[mask, 0], y[mask, i], label=f'System {i+1} Data', color=color_data, linestyle='-', linewidth=2)

                # Plot the model prediction with a different color
                plt.plot(
                    x[mask, 0],
                    fwd_surrogate.model_list[1].prediction(np.hstack((x[mask], fwd_surrogate.model_list[0].prediction(x[mask]))))[:, i],
                    label=f'System {i+1} Prediction LF Network', 
                    color=color_prediction, 
                    linestyle='--', 
                    linewidth=2
                )

            # Additional plot customization
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Plot for Combination {comb}')
            plt.legend()
            plt.grid(True)

            # Save the figure to the specified folder
            plot_filename = f"{folder_path}/plot_combination_{'_'.join(map(str, comb))}.png"
            plt.savefig(plot_filename)
            plt.close()

    @staticmethod    
    def getModel(params: dict, num_inputs: int, name: str, num_outputs: int) -> Model:
        """
        Returns a compiled Keras model based on the specified architecture.

        Parameters:
        - params: Dictionary containing parameters such as 'nodes', 'l2weight', 'kernel_init', 'opt', 'lr', and 'alpha'.
        - num_inputs: Integer representing the number of input features.
        - name: String specifying the model architecture ('HF', 'LF', 'Single', 'Hflin', 'Hfper', 'GP', or 'Inter').
        - num_outputs: Integer representing the number of output nodes.

        Returns:
        - model: A compiled Keras model.
        """
        inputs = Input(shape=(num_inputs,))
        
        if name == 'HF':
            # High-Fidelity model with a single hidden layer
            hidden1 = Dense(int(params['nodes']), activation='tanh', 
                            kernel_regularizer=l2(params['l2weight']), 
                            kernel_initializer=params['kernel_init'])(inputs)
            output = Dense(num_outputs, activation='linear', name='HF')(hidden1)

        elif name == 'LF':
            # Low-Fidelity model with 4 hidden layers
            hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(inputs)
            hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden1)
            hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden2)
            hidden4 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden3)
            output = Dense(num_outputs, activation='linear', name='LF')(hidden4)
            
        elif name == 'Single':
            # Single model with 4 hidden layers and L2 regularization
            hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], 
                            kernel_regularizer=l2(params['l2weight']))(inputs)
            hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], 
                            kernel_regularizer=l2(params['l2weight']))(hidden1)
            hidden3 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'], 
                            kernel_regularizer=l2(params['l2weight']))(hidden2)
            output = Dense(num_outputs, activation='linear', name='Single')(hidden2)

        elif name == 'Hflin':
            # High-Fidelity linear model with L2 regularization
            hiddenlin = Dense(64, activation='linear', kernel_regularizer=l2(params['l2weight']), 
                            kernel_initializer=params['kernel_init'])(inputs)
            output = Dense(num_outputs, activation='linear', name='HFlin')(hiddenlin)

        elif name == 'Hfper':
            # High-Fidelity model with a custom activation function
            hiddenlin = Dense(64, activation=Activations.custom_activation, kernel_regularizer=l2(params['l2weight']), 
                            kernel_initializer=params['kernel_init'])(inputs)
            output = Dense(num_outputs, activation='linear', name='HFper')(hiddenlin)

        elif name == 'Inter':
            # Intermediate model with a merge layer and dual outputs (HF and LF)
            hidden1 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(inputs)
            hidden2 = Dense(64, activation='tanh', kernel_initializer=params['kernel_init'])(hidden1)
            outputLF = Dense(num_outputs, activation='linear', name='LF')(hidden2)
            outputadd = Dense(int(params['nodes']), activation='tanh', 
                            kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                            kernel_initializer=params['kernel_init'])(hidden2)
            merge = concatenate([outputLF, outputadd])
            hidden3 = Dense(int(params['nodes']), activation='tanh', 
                            kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                            kernel_initializer=params['kernel_init'])(merge)
            hidden4 = Dense(int(params['nodes']), activation='tanh', 
                            kernel_regularizer=l2((1 - params['alpha']) * params['l2weight']), 
                            kernel_initializer=params['kernel_init'])(hidden3)
            outputHF = Dense(4, activation='linear', name='HF')(hidden4)
            output = [outputHF, outputLF]
            model = Model(inputs=inputs, outputs=output)
            opti = Helpers_NN.getOpti(params['opt'], params['lr'])
            model.compile(loss=Helpers_NN.custom_loss, loss_weights=[params['alpha'], 1 - params['alpha']], optimizer=opti)
            return model

        # For other models without multiple outputs
        model = Model(inputs=inputs, outputs=output)
        opti = Helpers_NN.getOpti(params['opt'], params['lr'])
        model.compile(loss='mse', optimizer=opti, metrics=['mse'])
        
        return model


class System1(SystemSolver):
    """
    Solver for a specific system of differential equations (Lotka-Volterra 1 param).
    """
    def __init__(self):
        super().__init__()
        self.dim_sys = 3  # Dimension of the system (number of equations)

    def equations(self, t: float, y: np.ndarray, mu: float) -> np.ndarray:
        """
        Define the system of differential equations for System1.

        Args:
        - t: Current time point
        - y: Current state of the system (array of values)
        - mu: Parameter for the differential equations

        Returns:
        - Array of derivatives (dy/dt)
        """
        dy1dt = y[0] * (mu - 0.1 * y[0] - 0.5 * y[1] - 0.5 * y[2])
        dy2dt = y[1] * (-mu + 0.5 * y[0] - 0.3 * y[2])
        dy3dt = y[2] * (-mu + 0.2 * y[0] + 0.5 * y[1])
        return np.array([dy1dt, dy2dt, dy3dt])

    def print_system(self):
        """
        Print the system of differential equations for System1.
        """
        print("The system of differential equations is:")
        print("dy1/dt = y1 * (mu - 0.1 * y1 - 0.5 * y2 - 0.5 * y3)")
        print("dy2/dt = y2 * (-mu + 0.5 * y1 - 0.3 * y3)")
        print("dy3/dt = y3 * (-mu + 0.2 * y1 + 0.5 * y2)")
        

    
class System2(SystemSolver):
    """
    Solver for a specific system of differential equations (Lotka-Volterra 4 params).
    """
    def __init__(self):
        """
        Initialize the solver for System2 with default values.
        """
        super().__init__()
        self.dim_sys = 2        # Dimension of the system (number of equations)

    def equations(self, t: float, y: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """
        Define the system of differential equations for System2.

        Args:
        - t: Current time point
        - y: Current state of the system (array of values)
        - a, b, c, d: Parameters for the differential equations

        Returns:
        - Array of derivatives (dy/dt)
        """
        dy1dt = a * y[0] - b * y[0] * y[1]
        dy2dt = c * y[0] * y[1] - d * y[1]
        return np.array([dy1dt, dy2dt])

    def print_system(self):
        """
        Print the system of differential equations for System2.
        """
        print("The system of differential equations is:")
        print("dy1/dt = a * y1 - b * y1 * y2")
        print("dy2/dt = c * y1 * y2 - d * y2")

class System3(SystemSolver):
    """
    Solver for a modified FitzHugh-Nagumo-like system of differential equations.
    """
    def __init__(self):
        """
        Initialize the solver for SystemModifiedFitzHughNagumo with default values.
        """
        super().__init__()
        self.dim_sys = 2        # Dimension of the system (number of equations)

    def equations(self, t: float, y: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
        """
        Define the system of differential equations for the modified FitzHugh-Nagumo system.

        Args:
        - t: Current time point
        - y: Current state of the system (array of values)
        - alpha, beta, gamma: Parameters for the differential equations

        Returns:
        - Array of derivatives (dy/dt)
        """
        # y[0] represents the activator variable (v)
        # y[1] represents the inhibitor variable (w)

        # Equation for the activator (v)
        dvdt = y[0] - (y[0]**3) / 3 - y[1] + alpha
        
        # Equation for the inhibitor (w)
        dwdt = beta * (y[0] + gamma - y[1])
        
        return np.array([dvdt, dwdt])

    def print_system(self):
        """
        Print the system of differential equations for SystemModifiedFitzHughNagumo.
        """
        print("The modified FitzHugh-Nagumo system of differential equations is:")
        print("dv/dt = v - v^3 / 3 - w + alpha")
        print("dw/dt = beta * (v + gamma - w)")


class System4(SystemSolver):
    """
    Solver for the forced Van der Pol oscillator system of differential equations.
    """
    def __init__(self):
        """
        Initialize the solver for the forced Van der Pol system with default values.
        """
        super().__init__()
        self.dim_sys = 2  # Dimension of the system (number of equations)

    def equations(self, t: float, y: np.ndarray, mu: float, omega: float) -> np.ndarray:
        """
        Define the system of differential equations for the forced Van der Pol oscillator.

        Args:
        - t: Current time point
        - y: Current state of the system (array of values)
        - mu: Parameter for the nonlinearity and damping
        - omega: Parameter for the external forcing frequency

        Returns:
        - Array of derivatives (dy/dt)
        """
        # y[0] is x (the primary variable)
        # y[1] is dx/dt (the derivative of x)
        
        # Equation for dx/dt
        dxdt = y[1]
        
        # Equation for d²x/dt², expressed as dy1dt (acceleration)
        dy1dt = mu * (1 - y[0]**2) * y[1] - y[0] + np.sin(omega * t)
        
        return np.array([dxdt, dy1dt])

    def print_system(self):
        """
        Print the system of differential equations for the forced Van der Pol oscillator.
        """
        print("The forced Van der Pol oscillator system of differential equations is:")
        print("dx/dt = y1")
        print("dy1/dt = mu * (1 - x^2) * y1 - x + sin(omega * t)")


# the file can be easily expanded adding new classes with the same form 