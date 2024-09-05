import numpy as np
from numpy import newaxis as _
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Any, List, Dict, Union, Tuple
from numba import njit, prange

import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout
from utils.helper_functions import Helpers_NN, Clean

ParamsType = Dict[str, Union[int, float, str, Any]]
OutputType = Union[Model, List[Model]]


class BurgerEquation:

    def __init__(self, nh: int = 101, nt: int = 151, T: float = 2.0, L: float = 1.0,
                 nre: int = 20, re_min: float = 80, re_max: float = 500, seed: int = 42):
        """
        Initialize the BurgerEquation class.

        Parameters:
        - nh (int): Number of spatial grid points.
        - nt (int): Number of time points.
        - T (float): Total time.
        - L (float): Length of the spatial domain.
        - nre (int): Number of Reynolds numbers to sample.
        - re_min (float): Minimum Reynolds number.
        - re_max (float): Maximum Reynolds number.
        - seed (int): Random seed for reproducibility.
        """
        self.n_POD=None
        self.basis=None
        self.nh, self.nt, self.T, self.L = nh, nt, T, L
        self.nre, self.re_min, self.re_max = nre, re_min, re_max
        

        # Space-time discretization
        self.x = np.linspace(0, L, nh)
        self.t = np.linspace(0, T, nt)
        
        # Reynolds numbers
        self.re = np.linspace(0, 1, nre)
        
        # Split into train/test
        self.re_train, self.re_test = train_test_split(self.re, test_size=0.2, random_state=seed)
        self.nre_train, self.nre_test = len(self.re_train), len(self.re_test)
        
        # Initialize data storage
        self.u_hf = np.zeros((self.nre, self.nt, self.nh))
        self.u_lf = np.zeros((self.nre, self.nt, self.nh))
        self.u_hf_test = np.zeros((self.nre_test, self.nt, self.nh))
        self.u_lf_test = np.zeros((self.nre_test, self.nt, self.nh))

        self.u_POD_lf_train,self.u_POD_hf_train, self.u_POD_lf_test, self.u_POD_hf_test=None,None,None,None
        
    @staticmethod
    @njit(fastmath=True)
    def u_HF(x: float, t: float, re: float) -> np.ndarray:
        """
        High-fidelity model: exact analytical solution.

        Parameters:
        - x (float): Spatial coordinate.
        - t (float): Time.
        - re (float): Reynolds number.

        Returns:
        - np.ndarray: High-fidelity solution at given x, t, and re.
        """
        A0 = np.exp(re / 8.0)
        return x / (t + 1) / (1.0 + np.exp(re * (x**2) / (4 * t + 4)) * ((t + 1) / A0)**0.5)

    @staticmethod
    @njit(fastmath=True)
    def u_LF(x: float, t: float, re: float) -> np.ndarray:
        """
        Low-fidelity model.

        Parameters:
        - x (float): Spatial coordinate.
        - t (float): Time.
        - re (float): Reynolds number.

        Returns:
        - np.ndarray: Low-fidelity solution at given x, t, and re.
        """
        A0 = np.exp(re / 8.0)
        return x / (t + 1) / (1.0 + np.exp(re * x / (4 * t + 4)) * ((t + 1) / A0)**0.5)
    
    def set_POD(self, POD: int, basis: np.ndarray):
        """
        Set the POD (Proper Orthogonal Decomposition) basis.

        Parameters:
        - POD (int): Number of POD modes.
        - basis (np.ndarray): POD basis matrix.
        """
        self.n_POD = POD
        self.basis = basis

    def _forward_low_fidelity(self, x_final: np.ndarray, data_points: np.ndarray, x_support: np.ndarray) -> np.ndarray:
        """
        Generate low fidelity model using POD basis.
        Useful for BIP 
        
        Parameters:
        - x_final (np.ndarray): Final input data.
        - x_support (np.ndarray): Support data points.
        - data_points (np.ndarray): Data points to project onto the POD basis.

        Returns:
        - new_inputs (np.ndarray): New input data incorporating low fidelity model.
        """
        if self.n_POD is None or self.basis is None:
            raise KeyError("POD basis not provided")

        dim_data = data_points.shape[0]
        dim_support = x_support.shape[0]

        u_lf_inv = np.zeros((1, dim_data, dim_support))

        for n in range(dim_data):       
            for i in range(dim_support):
                u_lf_inv[0, n, i] = self.u_LF(x_support[i,0], data_points[n, 0], self.denormalize(x_final[0]))

        # Reshape and project onto the POD basis
        u_lf_pod = np.reshape(u_lf_inv, (dim_data, dim_support))
        ulf_train = u_lf_pod @ self.basis
        ulf_train = np.reshape(ulf_train, (1, dim_data, self.n_POD))

        t_grid_lstm, re_grid_lstm = np.meshgrid(data_points[:,0], x_final[0])
        
        # Adjusting the axes order for concatenation
        t_grid_lstm = np.expand_dims(t_grid_lstm, axis=-1)
        re_grid_lstm = np.expand_dims(re_grid_lstm, axis=-1)

        # Concatenate along the last axis
        new_inputs = np.concatenate(( re_grid_lstm, ulf_train), axis=2)
        return new_inputs

    def denormalize(self, r: float) -> float:
        """
        Maps normalized Reynolds number [0, 1] onto original range [re_min, re_max].

        Parameters:
        - r (float): Normalized Reynolds number.

        Returns:
        - float: Denormalized Reynolds number.
        """
        return r * (self.re_max - self.re_min) + self.re_min


    def generate_data(self) -> None:
        """
        Generate training and test data for high-fidelity and low-fidelity models.
        Populates u_hf, u_lf, u_hf_test, u_lf_test attributes.
        """
        self.u_hf, self.u_lf = self._generate_data_parallel(self.re_train, self.nre_train)
        
        self.u_hf_test, self.u_lf_test = self._generate_data_parallel(self.re_test, self.nre_test)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _generate_data_parallel(re_set: np.ndarray, nre_set: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Internal static method to generate data in parallel.
        
        Parameters:
        - re_set (np.ndarray): Array of normalized Reynolds numbers.
        - nre_set (int): Number of Reynolds numbers in the set.
        
        Returns:
        - u_hf (np.ndarray): High-fidelity dataset.
        - u_lf (np.ndarray): Low-fidelity dataset.
        """
        nh, nt = 101, 151  # Assuming nh and nt are fixed
        L, T = 1.0, 2.0    # Spatial and temporal domain lengths
        re_min, re_max = 80, 500
        x = np.linspace(0, L, nh)
        t = np.linspace(0, T, nt)

        u_hf = np.zeros((nre_set, nt, nh))
        u_lf = np.zeros((nre_set, nt, nh))

        for p in prange(nre_set):
            re_value = re_set[p] * (re_max - re_min) + re_min
            A0_hf = np.exp(re_value / 8.0)
            A0_lf = np.exp(re_value / 8.0)

            for n in range(nt):
                t_n = t[n]
                x_sq = x ** 2
                x_lin = x

                # Vectorized computation for the high-fidelity model
                u_hf[p, n, :] = x / (t_n + 1) / (1.0 + np.exp(re_value * x_sq / (4 * t_n + 4)) * ((t_n + 1) / A0_hf)**0.5)
                
                # Vectorized computation for the low-fidelity model
                u_lf[p, n, :] = x / (t_n + 1) / (1.0 + np.exp(re_value * x_lin / (4 * t_n + 4)) * ((t_n + 1) / A0_lf)**0.5)
        
        return u_hf, u_lf        
 

    def _POD_ROM_set_train(self, u_POD_lf_train: np.ndarray, u_POD_hf_train: np.ndarray) -> None:
        """
        Set the POD-ROM training data for both low-fidelity (LF) and high-fidelity (HF) datasets.
        (used in "MOD_helper")
        Args:
        - u_POD_lf_train: Low-fidelity training data (POD reduced).
        - u_POD_hf_train: High-fidelity training data (POD reduced).

        Returns:
        - None: Sets the class attributes for training data.
        """
        self.u_POD_lf_train = u_POD_lf_train 
        self.u_POD_hf_train = u_POD_hf_train


    def _POD_ROM_set_test(self, u_POD_lf_test: np.ndarray, u_POD_hf_test: np.ndarray) -> None:
        """
        Set the POD-ROM test data for both low-fidelity (LF) and high-fidelity (HF) datasets.
        (used in "MOD_helper")

        Args:
        - u_POD_lf_test: Low-fidelity test data (POD reduced).
        - u_POD_hf_test: High-fidelity test data (POD reduced).

        Returns:
        - None: Sets the class attributes for test data.
        """
        self.u_POD_lf_test = u_POD_lf_test 
        self.u_POD_hf_test = u_POD_hf_test

    def get_input_dimensions(self) -> Tuple[int, int, int, int]:
        """
        Retrieve the input dimensions for the training and testing datasets.
        (used in "MOD_helper")

        Returns:
        - nre_train: Number of Reynolds numbers (or other parameter) in the training set.
        - nre_test: Number of Reynolds numbers (or other parameter) in the test set.
        - nt: Number of time steps in the dataset (same for both train and test).
        """
        return self.nre_train, self.nre_test, self.nt, self.nt


    def get_space_discretization(self) -> Tuple[int, int]:
        """
        Retrieve the spatial discretization for the system.
        (used in "MOD_helper")

        Returns:
        - nh: Number of spatial grid points (same for both returned values).
        """
        return self.nh, self.nh

    def definitionLSTM_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the LSTM inputs and outputs for training and testing based on POD reduced-order models.

        Raises:
        - ValueError: If any of the POD training or test data is None.

        Returns:
        - input_train: LSTM input for training (concatenation of t_grid, re_grid, and low-fidelity training data).
        - output_train: LSTM output for training (high-fidelity training data).
        - input_test: LSTM input for testing (concatenation of t_grid, re_grid, and low-fidelity test data).
        - output_test: LSTM output for testing (high-fidelity test data).
        """

        # Ensure that POD datasets are available
        if self.u_POD_lf_train is None or self.u_POD_hf_train is None or self.u_POD_lf_test is None or self.u_POD_hf_test is None:
            raise ValueError("u_POD is None, execute POD!")

        # Prepare LSTM inputs for training
        self.t_grid_lstm, self.re_grid_lstm = np.meshgrid(self.t, self.re_train)
        self.input_train = np.concatenate((self.t_grid_lstm[..., _], self.re_grid_lstm[..., _], self.u_POD_lf_train), axis=2)
        self.output_train = self.u_POD_hf_train

        # Prepare LSTM inputs for testing
        self.t_grid_lstm_test, self.re_grid_lstm_test = np.meshgrid(self.t, self.re_test)
        self.input_test = np.concatenate((self.t_grid_lstm_test[..., _], self.re_grid_lstm_test[..., _], self.u_POD_lf_test), axis=2)
        self.output_test = self.u_POD_hf_test 
            
        return self.input_train, self.output_train, self.input_test, self.output_test


    def plt_time_instants(self, re_values: List[float] = [100, 400], folder_name: str = 'Burger_output', file_name: str = 'burger_evolution') -> None:
        """
        Save plots of true values simulations for both high-fidelity and low-fidelity models
        at different time points and Reynolds numbers in a specified folder.
        
        Parameters:
        - re_values: List of Reynolds numbers to plot.
        - folder_name: Name of the folder where the plot image will be saved. If None, defaults to 'Burger_output'.
        - file_name: Name of the file to save the plot as. If None, defaults to 'burger_evolution'.
        """
        
        # Create the output directory structure
        base_dir = 'output'
        save_dir = Clean.create_output_directory(base_dir, folder_name)
        
        # Construct the full save path
        save_path = os.path.join(save_dir, file_name + '.png')

        times = [0, self.nt // 2, self.nt - 1]
        alphas = [1, 0.5, 0.25]

        plt.figure(figsize=(7, 5))

        for i, re in enumerate(re_values):
            for j, time in enumerate(times):
                hf_data = self.u_HF(self.x, self.t[time], re)
                lf_data = self.u_LF(self.x, self.t[time], re)

                # Plot High-fidelity
                plt.subplot(2, 2, 1 + i * 2)
                plt.plot(self.x, hf_data, label=f"$t$ = {self.t[time]:.2f}", alpha=alphas[j], color='steelblue')
                plt.title(f"High-fidelity $Re$ = {re}", fontsize=11)
                plt.xlabel('x')
                plt.legend(loc='upper right')

                # Plot Low-fidelity
                plt.subplot(2, 2, 2 + i * 2)
                plt.plot(self.x, lf_data, label=f"$t$ = {self.t[time]:.2f}", alpha=alphas[j], color='green')
                plt.title(f"Low-fidelity $Re$ = {re}", fontsize=11)
                plt.xlabel('x')
                plt.legend(loc='upper right')

        plt.tight_layout()

        # Save the plot without displaying it
        plt.savefig(save_path)
        plt.close()


    def plot_single_contour(self, ax: Any, t_grid: np.ndarray, x_grid: np.ndarray, data: np.ndarray, 
                            title: str, xlabel: str, ylabel: str, cmap: str = 'plasma', levels: int = 10, 
                            colorbar_label: str = 'u') -> None:
        """
        Plot a single contour plot.

        Parameters:
        - ax (Any): Matplotlib axes object to plot on.
        - t_grid (np.ndarray): Grid for the t-axis.
        Shape: (nt, nh)
        - x_grid (np.ndarray): Grid for the x-axis.
        Shape: (nt, nh)
        - data (np.ndarray): Data to plot.
        Shape: (nt, nh)
        - title (str): Title of the plot.
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.
        - cmap (str): Colormap to use.
        - levels (int): Levels for contour plot.
        - colorbar_label (str): Label for the colorbar.

        Returns:
        - None
        """
        surf = ax.contourf(t_grid, x_grid, data.T, cmap=cmap, levels=levels)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=15, rotation=0)
        cbar = plt.colorbar(surf, ax=ax)
        cbar.ax.set_ylabel(colorbar_label, fontsize=12, labelpad=15, rotation=0)
        ax.set_title(title, fontsize=14)


    def plot_data(self, folder_name: str = 'Burger_output', file_name: str = 'data_burger') -> None:
        """
        Save plots of the generated data for both high-fidelity and low-fidelity models 
        in a specified folder.

        Parameters:
        - folder_name: Name of the folder where the plots will be saved.
        - file_name: Format string for the file name to save the plots.
        """
        # Directory setup using the static method
        base_dir = 'output'
        save_dir = Clean.create_output_directory(base_dir, folder_name)

        sorted_indices = np.argsort(self.re_train)
        sorted_re_train = self.re_train[sorted_indices]
        sorted_u_lf = self.u_lf[sorted_indices]
        sorted_u_hf = self.u_hf[sorted_indices]

        abs_lf = np.abs(sorted_u_lf)
        abs_hf = np.abs(sorted_u_hf)
        max_abs = max(abs_lf.max(), abs_hf.max())
        levels = np.linspace(0, max_abs, 11)
        plots_per_figure = 3

        for i in range(0, self.nre_train, plots_per_figure):
            num_plots = min(plots_per_figure, self.nre_train - i)
            fig, axes = plt.subplots(nrows=num_plots, ncols=2, figsize=(12, 3 * num_plots))

            if num_plots == 1:
                axes = np.expand_dims(axes, axis=0)

            for j in range(num_plots):
                ind_re = i + j
                t_grid, x_grid = np.meshgrid(self.t, self.x)

                lf_title = f'Low-fidelity $Re = ${int(self.denormalize(sorted_re_train[ind_re]))}'
                hf_title = f'High-fidelity $Re = ${int(self.denormalize(sorted_re_train[ind_re]))}'

                self.plot_single_contour(axes[j, 0], t_grid, x_grid, sorted_u_lf[ind_re], lf_title, 't', 'x', levels=levels)
                self.plot_single_contour(axes[j, 1], t_grid, x_grid, sorted_u_hf[ind_re], hf_title, 't', 'x', levels=levels)

            plt.tight_layout()

            # Save the figure
            plot_filename = f"{file_name}_{i // plots_per_figure + 1}.png"
            save_path = os.path.join(save_dir, plot_filename)
            plt.savefig(save_path)
            plt.close(fig)

            


    def plot_comparison(self, output_pred: np.ndarray, basis: np.ndarray, ind_test: np.ndarray, folder_name: str = 'comparison_plots', file_name: str = 'comparison_{index}.png') -> None:
        """
        Save plots comparing low-fidelity, high-fidelity, and predicted data.

        Parameters:
        - output_pred (np.ndarray): Predicted output from the model. Shape: (n_test, n_POD)
        - basis (np.ndarray): POD basis matrix. Shape: (n_POD, nh)
        - ind_test (np.ndarray): Indices of the test cases to plot. Shape: (n_test,)
        - folder_name (str): Folder to save the plots. Default is 'comparison_plots'.
        - file_name (str): File name format to save the plots. Must include '{index}' for unique file naming.
        """
        # Directory setup using the create_output_directory method
        base_dir = 'output'
        save_dir = Clean.create_output_directory(base_dir, folder_name)

        # Calculate the predicted output using the POD basis
        u_pred = np.nan_to_num(output_pred @ basis.T, nan=0.0, posinf=0.0, neginf=0.0)
        t_grid, x_grid = np.meshgrid(self.t, self.x)

        # Define levels for contour plots
        abs_lf = np.abs(self.u_lf_test)
        abs_hf = np.abs(self.u_hf_test)
        max_abs = max(abs_lf.max(), abs_hf.max())
        levels = np.linspace(0, max_abs, 11)

        for i, ind_re in enumerate(ind_test):
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 7.5))
            re_value = int(self.denormalize(self.re_test[ind_re]))
            plt.suptitle(f'Comparison of Burger Models - Re = {re_value}', fontsize=14)

            lf_title = 'Low-fidelity'
            hf_title = 'High-fidelity'
            pred_title = 'MF-POD prediction'

            # Plot the low-fidelity data
            self.plot_single_contour(axes[0], t_grid, x_grid, self.u_lf_test[ind_re], lf_title, 't', 'x', levels=levels, colorbar_label='$u_{LF}$')

            # Plot the high-fidelity data
            self.plot_single_contour(axes[1], t_grid, x_grid, self.u_hf_test[ind_re], hf_title, 't', 'x', levels=levels, colorbar_label='$u_{HF}$')
            
            # Plot the predicted data
            u_pred_clipped = np.clip(u_pred[ind_re], levels.min(), levels.max())
            self.plot_single_contour(axes[2], t_grid, x_grid, u_pred_clipped, pred_title, 't', 'x', levels=levels, colorbar_label='$u_{POD}$')

            plt.tight_layout()

            # Save the plot
            plot_filename = file_name.format(index=i + 1)
            save_path = os.path.join(save_dir, plot_filename)
            plt.savefig(save_path)
            plt.close(fig)

            print(f"Plot saved to {save_path}")


    def plot_error(self, output_pred: np.ndarray, basis: np.ndarray, ind_test: np.ndarray, folder_name: str = 'error_plots', file_name: str = 'error_{index}.png') -> None:
        """
        Plot and save the relative and absolute errors for low-fidelity (LF) and multi-fidelity POD (MF-POD) predictions.

        Parameters:
        - output_pred (np.ndarray): Predicted output from the model. Shape: (n_test, n_POD)
        - basis (np.ndarray): Basis matrix used for POD. Shape: (n_POD, nh)
        - ind_test (np.ndarray): Indices of the test cases to plot. Shape: (n_test,)
        - folder_name (str): Folder to save the plots. Default is 'error_plots'.
        - file_name (str): File name format to save the plots. Must include '{index}' for unique file naming.
        """
        # Directory setup using the create_output_directory method
        base_dir = 'output'
        save_dir = Clean.create_output_directory(base_dir, folder_name)

        # Calculate the predicted outputs using the POD basis
        u_pred = output_pred @ basis.T

        # Calculate relative errors
        rel_err_lf = np.linalg.norm(self.u_lf_test - self.u_hf_test) / np.linalg.norm(self.u_hf_test)
        rel_err_pred = np.linalg.norm(u_pred - self.u_hf_test) / np.linalg.norm(self.u_hf_test)

        # Calculate absolute errors
        abs_err_lf = np.abs(self.u_lf_test - self.u_hf_test)
        abs_err_pred = np.abs(u_pred - self.u_hf_test)
        max_abs_err = max(abs_err_lf.max(), abs_err_pred.max())
        levels = np.linspace(0, max_abs_err, 11)

        # Create a time-space grid for plotting
        t_grid, x_grid = np.meshgrid(self.t, self.x)

        # Plot the errors for each test index and save the plots
        for i, ind_re in enumerate(ind_test):
            fig = plt.figure(figsize=(6, 5))
            plt.suptitle(f'Burger error - Re = {int(self.denormalize(self.re_test[ind_re]))}', fontsize=14)

            # Plot absolute error for low-fidelity (LF)
            ax = fig.add_subplot(211)
            self.plot_single_contour(ax, t_grid, x_grid, abs_err_lf[ind_re], 'Absolute error for LF', 't', 'x', cmap='bwr', levels=levels, colorbar_label='Err.')

            # Plot absolute error for multi-fidelity POD (MF-POD)
            ax = fig.add_subplot(212)
            self.plot_single_contour(ax, t_grid, x_grid, abs_err_pred[ind_re], 'Absolute error for MF-POD', 't', 'x', cmap='bwr', levels=levels, colorbar_label='Err.')

            plt.tight_layout()

            # Save the plot
            plot_filename = file_name.format(index=i + 1)
            save_path = os.path.join(save_dir, plot_filename)
            plt.savefig(save_path)
            plt.close(fig)  # Close the figure after saving

            print(f"Plot saved to {save_path}")
        
    def plot_(self, re: float) -> None:
        """
        Plot both low-fidelity and high-fidelity outputs for a given Reynolds number.

        Parameters:
        - re (float): Normalized Reynolds number.
        """
        re_value = self.denormalize(re)
        t_grid, x_grid = np.meshgrid(self.t, self.x)

        u_hf = np.array([[self.u_HF(self.x[i], self.t[n], re_value) for i in range(self.nh)] for n in range(self.nt)])
        u_lf = np.array([[self.u_LF(self.x[i], self.t[n], re_value) for i in range(self.nh)] for n in range(self.nt)])

        fig = plt.figure(figsize=(6, 5))

        ax = fig.add_subplot(211)
        surf = ax.contourf(t_grid, x_grid, u_lf.T, cmap='plasma', levels=10)
        plt.xlabel('t', fontsize=12)
        plt.ylabel('x', fontsize=12, labelpad=15, rotation=0)
        cbar = fig.colorbar(surf, ax=ax)
        cbar.ax.set_ylabel('u', fontsize=12, labelpad=15, rotation=0)
        plt.title(f'Low-fidelity $Re = ${re_value:.2f}', fontsize=14)

        ax = fig.add_subplot(212)
        surf = ax.contourf(t_grid, x_grid, u_hf.T, cmap='plasma', levels=10)
        plt.xlabel('t', fontsize=12)
        plt.ylabel('x', fontsize=12, labelpad=15, rotation=0)
        cbar = fig.colorbar(surf, ax=ax)
        cbar.ax.set_ylabel('u', fontsize=12, labelpad=15, rotation=0)
        plt.title(f'High-fidelity $Re = ${re_value:.2f}', fontsize=14)

        plt.tight_layout()
        plt.show()     
        
    @staticmethod
    def getModel(params: ParamsType, num_inputs: int, name: str, num_outputs: int) -> OutputType:
        """
        Create and return a compiled Keras model based on the given architecture name and parameters.

        Args:
            params (Dict): Dictionary containing model parameters like 'nodes', 'dropout', 'l2weight', etc.
            num_inputs (int): Number of input features.
            name (str): Name of the model architecture (only 'LSTM' in this case).
            num_outputs (int): Number of output neurons.

        Returns:
            model (Model or List[Model]): Compiled Keras model, or list of models for some architectures.
        """

        inputs = None
        output = None

        if name == "LSTM":
            # Input layer for LSTM model
            inputs = Input(shape=(None, num_inputs))
            a = inputs

            # LSTM layers with Dropout
            for i in range(params['lay']):
                a = Dropout(params['dropout'])(a)
                a = LSTM(params['nodes'], return_sequences=True)(a)

            # Dense output layer
            output = Dense(num_outputs, activation='linear')(a)
        else:
            ValueError("Only LSTM is available for this test case!")

        # General model compilation for other architectures
        model = Model(inputs=inputs, outputs=output)
        opti = Helpers_NN.getOpti(params['opt'], params['lr'])
        model.compile(loss='mse', optimizer=opti, metrics=['mse'])

        return model




