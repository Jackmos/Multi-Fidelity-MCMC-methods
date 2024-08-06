import numpy as np
from numpy import newaxis as _
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import extmath
from typing import Any, Optional

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
        self.POD=None
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

    @staticmethod
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
    
    def set_POD(self, POD:int, basis):
        self.POD = POD
        self.basis=basis
        
    def impose_input(self, t_eval, y_obs)

    def _forward_low_fidelity(self, x_final):
        
        if self.POD is None or self.basis is None:
            return Error

        u_lf_inv = np.zeros((1, self.nt, self.nh))
        
        for n in range(self.inputs.shape[1]):       
            for i in range(self.input2.shape[0]):
                u_lf_inv[0, n, i] = self.u_LF(self.input2[i], self.inputs[0,n,0], self.denormalize(x_final[0]))
                
                
                
        u_lf_pod=np.reshape(u_lf_inv,(self.nt,self.nh))
        ulf_train=u_lf_pod@self.basis
        ulf_train=np.reshape(ulf_train,(1,self.nt,self.n_POD))

        t_grid_lstm, re_grid_lstm = np.meshgrid(self.inputs[0,:,:1], x_final[0])
        new_inputs = np.concatenate((t_grid_lstm[:,:,_], re_grid_lstm[:,:,_], ulf_train), axis = 2)


        return new_input


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
        for p in range(self.nre_train):
            for n in range(self.nt):
                for i in range(self.nh):
                    re_value = self.denormalize(self.re_train[p])
                    self.u_hf[p, n, i] = self.u_HF(self.x[i], self.t[n], re_value)
                    self.u_lf[p, n, i] = self.u_LF(self.x[i], self.t[n], re_value)

        for p in range(self.nre_test):
            for n in range(self.nt):
                for i in range(self.nh):
                    re_value = self.denormalize(self.re_test[p])
                    self.u_hf_test[p, n, i] = self.u_HF(self.x[i], self.t[n], re_value)
                    self.u_lf_test[p, n, i] = self.u_LF(self.x[i], self.t[n], re_value)
                    
    
    def plot_ground_truth_simulations(self) -> None:
        """
        Plot ground truth simulations for both high-fidelity and low-fidelity models.
        """
        times = [0, self.nt // 2, self.nt - 1]
        alphas = [1, 0.5, 0.25]
        plt.figure(figsize=(7, 5))
        for j in range(3):
            plt.subplot(2, 2, 1)
            plt.plot(self.x, self.u_HF(self.x, self.t[times[j]], 100), 
                     label=f"$t$ = {self.t[times[j]]:.2f}", alpha=alphas[j], color='steelblue')
            plt.title("High-fidelity $Re$ = 100", fontsize=11)
            plt.xlabel('x')
            plt.legend(loc='upper right')

            plt.subplot(2, 2, 2)
            plt.plot(self.x, self.u_HF(self.x, self.t[times[j]], 400), 
                     label=f"$t$ = {self.t[times[j]]:.2f}", alpha=alphas[j], color='steelblue')
            plt.title("High-fidelity $Re$ = 400", fontsize=11)
            plt.xlabel('x')
            plt.legend(loc='upper right')

            plt.subplot(2, 2, 3)
            plt.plot(self.x, self.u_LF(self.x, self.t[times[j]], 100), 
                     label=f"$t$ = {self.t[times[j]]:.2f}", alpha=alphas[j], color='green')
            plt.title("Low-fidelity $Re$ = 100", fontsize=11)
            plt.xlabel('x')
            plt.legend(loc='upper right')

            plt.subplot(2, 2, 4)
            plt.plot(self.x, self.u_LF(self.x, self.t[times[j]], 400), 
                     label=f"$t$ = {self.t[times[j]]:.2f}", alpha=alphas[j], color='green')
            plt.title("Low-fidelity $Re$ = 400", fontsize=11)
            plt.xlabel('x')
            plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    # def plot_data(self) -> None:
    #     """
    #     Plot the generated data for both high-fidelity and low-fidelity models.
    #     """
    #     for ind_re in range(self.nre_train):
    #         t_grid, x_grid = np.meshgrid(self.t, self.x)
    #         fig = plt.figure(figsize=(6, 5))

    #         ax = fig.add_subplot(211)
    #         surf = ax.contourf(t_grid, x_grid, self.u_lf[ind_re].T, cmap='plasma', levels=10)
    #         plt.xlabel('t', fontsize=12)
    #         plt.ylabel('x', fontsize=12, labelpad=15, rotation=0)
    #         cbar = fig.colorbar(surf, ax=ax)
    #         cbar.ax.set_ylabel('u', fontsize=12, labelpad=15, rotation=0)
    #         plt.title(f'Low-fidelity $Re = ${int(self.denormalize(self.re_train[ind_re]))}', fontsize=14)

    #         ax = fig.add_subplot(212)
    #         surf = ax.contourf(t_grid, x_grid, self.u_hf[ind_re].T, cmap='plasma', levels=10)
    #         plt.xlabel('t', fontsize=12)
    #         plt.ylabel('x', fontsize=12, labelpad=15, rotation=0)
    #         cbar = fig.colorbar(surf, ax=ax)
    #         cbar.ax.set_ylabel('u', fontsize=12, labelpad=15, rotation=0)
    #         plt.title(f'High-fidelity $Re = ${int(self.denormalize(self.re_train[ind_re]))}', fontsize=14)

    #         plt.tight_layout()
    #         plt.show()

    def plot_single_contour(self, ax, t_grid, x_grid, data, title, xlabel, ylabel, cmap='plasma', levels=10, colorbar_label='u'):
        surf = ax.contourf(t_grid, x_grid, data.T, cmap=cmap, levels=levels)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=15, rotation=0)
        cbar = plt.colorbar(surf, ax=ax)
        cbar.ax.set_ylabel('u', fontsize=12, labelpad=15, rotation=0)
        ax.set_title(title, fontsize=14)
    
    def plot_data(self):
        """
        Plot the generated data for both high-fidelity and low-fidelity models.
        """
        abs_lf = np.abs(self.u_lf)
        abs_hf= np.abs(self.u_hf)
        max_abs = max(abs_lf.max(), abs_hf.max())
        levels = np.linspace(0, max_abs, 11)
        
        for ind_re in range(self.nre_train):
            t_grid, x_grid = np.meshgrid(self.t, self.x)
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 5))
            
            lf_title = f'Low-fidelity $Re = ${int(self.denormalize(self.re_train[ind_re]))}'
            self.plot_single_contour(axes[0], t_grid, x_grid, self.u_lf[ind_re], lf_title, 't', 'x', levels=levels)
            
            hf_title = f'High-fidelity $Re = ${int(self.denormalize(self.re_train[ind_re]))}'
            self.plot_single_contour(axes[1], t_grid, x_grid, self.u_hf[ind_re], hf_title, 't', 'x', levels=levels)
            
            plt.tight_layout()
            plt.show()

    def plot_comparison(self, output_pred, basis, ind_test):
        """
        Plot comparison of low-fidelity, high-fidelity, and predicted data.
        """
        u_pred = output_pred @ basis.T
        t_grid, x_grid = np.meshgrid(self.t, self.x)

        abs_lf = np.abs(self.u_lf_test)
        abs_hf= np.abs(self.u_hf_test)
        max_abs = max(abs_lf.max(), abs_hf.max())
        levels = np.linspace(0, max_abs, 11)

        for ind_re in ind_test:
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 7.5))
            re_value = int(self.denormalize(self.re_test[ind_re]))
            plt.suptitle(f'Re = {re_value}', fontsize=14)
            
            lf_title = 'Low-fidelity'
            self.plot_single_contour(axes[0], t_grid, x_grid, self.u_lf_test[ind_re], lf_title, 't', 'x', levels=levels, colorbar_label='$u_{LF}$')
            
            hf_title = 'High-fidelity'
            self.plot_single_contour(axes[1], t_grid, x_grid, self.u_hf_test[ind_re], hf_title, 't', 'x', levels=levels, colorbar_label='$u_{HF}$')
            
            pred_title = 'MF-POD prediction'
            self.plot_single_contour(axes[2], t_grid, x_grid, u_pred[ind_re], pred_title, 't', 'x', levels=levels, colorbar_label='$u_{POD}$')
            
            plt.tight_layout()
            plt.show()
    
    
    def plot_error(self, output_pred, basis, ind_test):
        
        u_pred = output_pred @ basis.T
        rel_err_lf = np.linalg.norm(self.u_lf_test - self.u_hf_test)/np.linalg.norm(self.u_hf_test)
        rel_err_pred = np.linalg.norm(u_pred - self.u_hf_test)/np.linalg.norm(self.u_hf_test)

        abs_err_lf = np.abs(self.u_lf_test - self.u_hf_test)
        abs_err_pred = np.abs(u_pred - self.u_hf_test)
        max_abs_err = max(abs_err_lf.max(), abs_err_pred.max())

        levels = np.linspace(0, max_abs_err, 11)
       
        t_grid, x_grid = np.meshgrid(self.t, self.x)

        for ind_re in ind_test:
            fig = plt.figure(figsize=(6, 5))
            plt.suptitle('Re = ' + str(int(self.denormalize(self.re_test[ind_re]))), fontsize = 14)
            
            ax = fig.add_subplot(211)
            self.plot_single_contour(ax, t_grid, x_grid, abs_err_lf[ind_re], 'Absolute error for LF', 't', 'x', cmap='bwr', levels=levels, colorbar_label='abs. err.')

            
            ax = fig.add_subplot(212)
            self.plot_single_contour(ax, t_grid, x_grid, abs_err_pred[ind_re], 'Absolute error for MF-POD', 't', 'x', cmap='bwr', levels=levels, colorbar_label='abs. err.')


            
            
            plt.tight_layout()
            plt.show()
       
    def plot_(self, re: float) -> None:
        """
        Plot both low-fidelity and high-fidelity outputs for a given Reynolds number.

        Parameters:
        - re (float): normalized Reynolds number
        """
        re_value = self.denormalize(re)

        # Generate LF and HF data
        u_hf = np.array([[self.u_HF(self.x[i], self.t[n], re_value) for i in range(self.nh)] for n in range(self.nt)])
        u_lf = np.array([[self.u_LF(self.x[i], self.t[n], re_value) for i in range(self.nh)] for n in range(self.nt)])

        # Create the plots
        t_grid, x_grid = np.meshgrid(self.t, self.x)
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
        
        
        

class ROM:
    def __init__(self, burger_eq: Any, n_POD: int):
        """
        Initialize the Reduced Order Model (ROM) class.

        Parameters:
        - burger_eq: An instance of the Burger equation solver or model.
        - n_POD: The number of Proper Orthogonal Decomposition (POD) modes to retain.
        """
        
        # Assign the provided burger_eq instance to the class attribute
        self.burger_eq = burger_eq
        
        # Assign the number of POD modes to retain to the class attribute
        self.n_POD = n_POD
        
        # Placeholder for the basis vectors obtained from POD
        self.basis: Optional[np.ndarray] = None
        
        # Placeholder for the singular values obtained from POD
        self.S: Optional[np.ndarray] = None
        
        # Placeholder for the low-fidelity POD coefficients for the training data
        self.u_lf_pod: Optional[np.ndarray] = None
        
        # Placeholder for the high-fidelity POD coefficients for the training data
        self.u_hf_pod: Optional[np.ndarray] = None
        
        # Placeholder for the low-fidelity POD coefficients for the test data
        self.u_lf_pod_test: Optional[np.ndarray] = None
        
        # Placeholder for the high-fidelity POD coefficients for the test data
        self.u_hf_pod_test: Optional[np.ndarray] = None

    def compute_randomized_SVD(self, S: np.ndarray, N_POD: int, N_h: int, n_channels: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the randomized Singular Value Decomposition (SVD) for the input matrix S.

        Parameters:
        - S (np.ndarray): The input matrix of shape (n_channels * N_h, m), where m is the number of columns.
        - N_POD (int): The number of principal components to compute.
        - N_h (int): The number of spatial grid points.
        - n_channels (int): The number of channels.

        Returns:
        - tuple[np.ndarray, np.ndarray]: A tuple containing the left singular vectors (U) and the singular values (Sigma).
          - U (np.ndarray): The matrix of left singular vectors of shape (n_channels * N_h, N_POD).
          - Sigma (np.ndarray): The array of singular values.
        """
        U = np.zeros((n_channels * N_h, N_POD))
        Sigma = np.zeros((n_channels, N_POD))  # Initializing Sigma to store singular values for each channel

        for i in range(n_channels):
            start_idx = i * N_h
            end_idx = (i + 1) * N_h
            U[start_idx:end_idx], sigma, _ = extmath.randomized_svd(
                S[start_idx:end_idx, :],
                n_components=N_POD,
                transpose=False,
                flip_sign=False,
                random_state=123
            )
            Sigma[i, :] = sigma

        return U, Sigma

    def get_basis(self):
        return self.basis


    def perform_POD(self):
        """
        Perform Proper Orthogonal Decomposition (POD) on high-fidelity (HF) and low-fidelity (LF) data.
        
        This method reshapes the high-fidelity and low-fidelity data into a suitable format, computes the POD basis 
        using randomized SVD on the high-fidelity data, and then projects both the high-fidelity and low-fidelity 
        training and test data onto the computed POD basis. The projected data is reshaped to be suitable for 
        subsequent use in machine learning models such as LSTM networks.
        
        Steps:
        1. Reshape the HF and LF data into 2D arrays where each row corresponds to a time snapshot and each column to a spatial grid point.
        2. Compute the POD basis using the reshaped HF data.
        3. Project the reshaped HF and LF data onto the POD basis to obtain the POD coefficients.
        4. Reshape the POD coefficients to the original 3D format for use in machine learning models.
        """
        # Reshape for POD
        self.u_hf_pod = np.reshape(self.burger_eq.u_hf, (self.burger_eq.nre * self.burger_eq.nt, self.burger_eq.nh))
        self.u_lf_pod = np.reshape(self.burger_eq.u_lf, (self.burger_eq.nre * self.burger_eq.nt, self.burger_eq.nh))
        self.u_lf_pod_test = np.reshape(self.burger_eq.u_lf_test, (self.burger_eq.nre_test * self.burger_eq.nt, self.burger_eq.nh))
        self.u_hf_pod_test = np.reshape(self.burger_eq.u_hf_test, (self.burger_eq.nre_test * self.burger_eq.nt, self.burger_eq.nh))

        # Compute randomized SVD
        self.basis, self.S = self.compute_randomized_SVD(self.u_hf_pod, self.n_POD, self.burger_eq.nh, 1)

        # Project data onto POD basis
        self.ulf_train = self.u_lf_pod @ self.basis
        self.uhf_train = self.u_hf_pod @ self.basis
        self.ulf_test = self.u_lf_pod_test @ self.basis
        self.uhf_test = self.u_hf_pod_test @ self.basis

        # Reshape data for LSTM network
        self.ulf_train = np.reshape(self.ulf_train, (self.burger_eq.nre, self.burger_eq.nt, self.n_POD))
        self.uhf_train = np.reshape(self.uhf_train, (self.burger_eq.nre, self.burger_eq.nt, self.n_POD))
        self.ulf_test = np.reshape(self.ulf_test, (self.burger_eq.nre_test, self.burger_eq.nt, self.n_POD))
        self.uhf_test = np.reshape(self.uhf_test, (self.burger_eq.nre_test, self.burger_eq.nt, self.n_POD))


        self.t_grid_lstm, self.re_grid_lstm = np.meshgrid(self.burger_eq.t, self.burger_eq.re)
        self.input_train = np.concatenate((self.t_grid_lstm[:,:,_], self.re_grid_lstm[:,:,_], self.ulf_train), axis = 2)
        self.output_train = self.uhf_train

        #Test
        self.t_grid_lstm_test, self.re_grid_lstm_test = np.meshgrid(self.burger_eq.t, self.burger_eq.re_test)
        self.input_test = np.concatenate((self.t_grid_lstm_test[:,:,_], self.re_grid_lstm_test[:,:,_], self.ulf_test), axis = 2)
        self.output_test = self.uhf_test 
        
        return self.input_train, self.output_train, self.input_test, self.output_test
        

    def plot_POD_coefficients(self, ind_re: int):
        """Plot POD coefficients: LF vs HF"""
        fig = plt.figure(figsize=(12, 6))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('POD coefficients: LF vs HF', fontsize=14)
        t = self.burger_eq.t
        for mode in range(min(6, self.n_POD)):
            ax = fig.add_subplot(231 + mode)
            plt.plot(t, self.ulf_train[ind_re, :, mode], label='LF', linewidth=2, color='green', linestyle='--')
            plt.plot(t, self.uhf_train[ind_re, :, mode], label='HF', linewidth=2, color='blue')
            ax.title.set_text('POD coord. ' + str(mode + 1))
            plt.xlabel('t')
            plt.legend()
        plt.show()
        
    def plot_POD_output(self, ind_re: int, output_pred:np.ndarray=None):
        
        if output_pred is None or output_pred.shape!=self.output_test.shape:
            raise ValueError(f"Error: output_pred is not properly defined. ")       
        
        if self.output_test is None:
            raise ValueError(f"Error: output_test not defined, run 'perform_POD' before. ")       

         
        fig = plt.figure(figsize=(12, 6))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('POD coefficients: LF vs HF', fontsize=14)
        t = self.burger_eq.t
        for mode in range(min(6, self.n_POD)):
            ax = fig.add_subplot(231 + mode)
            plt.plot(t, self.output_test[ind_re, :, mode].reshape(-1), 'b-',label='LF', linewidth=2)
            plt.plot(t,  output_pred[ind_re, :, mode].reshape(-1), 'r--',label='HF', linewidth=2)
            ax.title.set_text('POD coord. ' + str(mode + 1))
            plt.xlabel('t')
            plt.legend()
        plt.show()
        






def process_data(datahf: np.ndarray, parameters: np.ndarray, t_eval: np.ndarray, Yhf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the elements of a dataset nearest to the ones given.
    
    Parameters:
    - datahf: 2D numpy array where datahf[:,1] contains parameter values.
    - parameters: 1D numpy array of parameter values to find in datahf.
    - t_eval: 2D numpy array of evaluation times.
    - Yhf: 1D numpy array of corresponding y values.
    
    Returns:
    - nearest_x: 1D numpy array of x values closest to each t_eval.
    - y_obs: 1D numpy array of corresponding y values from Yhf.
    """
    indices = np.where(datahf[:, 1] == parameters[0])[0]
    if len(indices) == 0:
        raise ValueError(f"No observations related to parameter: {parameters[0]}")
    
    datahf_values = datahf[indices, 0].reshape(-1, 1)
    t_eval_values = t_eval.reshape(1, -1)
    differences = np.abs(datahf_values - t_eval_values)
    closest_indices = np.argmin(differences, axis=0)
    nearest_x = datahf[indices[closest_indices], 0]
    y_obs = Yhf[indices[closest_indices]]
    
    return nearest_x, y_obs


def calculate_cov_likelihood(sigma: float, t_eval: np.ndarray) -> np.ndarray:
    """
    Calculate the covariance matrix for the likelihood.
    
    Parameters:
    - sigma: Standard deviation for the likelihood.
    - t_eval: 2D numpy array of evaluation times.
    
    Returns:
    - cov_likelihood: 2D numpy array representing the covariance matrix.
    """
    return sigma ** 2 * np.eye(t_eval.shape[0])
# LSTM VERSION 
def run_simulation( class_model, # type?
    datahf: np.ndarray, mean_prior: np.ndarray, cov_prior: np.ndarray, Yhf: np.ndarray, 
    sigma_noise: List[float], n_data: List[int], parameters: np.ndarray, sigma: np.ndarray, 
    rwmh_scaling: np.ndarray, rwmh_cov: np.ndarray, rwmh_adaptive: bool, 
    iterations: int, burnin: int, n_chains: int, final_model, algo: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a simulation to estimate parameters and calculate errors.
    
    Parameters:
    - datahf: 2D numpy array containing data.
    - mean_prior: 1D numpy array for the mean of the prior.
    - cov_prior: 2D numpy array for the covariance of the prior.
    - Yhf: 1D numpy array of observed values.
    - sigma_noise: List of noise levels.
    - n_data: List of number of data points.
    - parameters: 1D numpy array of parameters.
    - sigma: 1D numpy array of standard deviations for the likelihood.
    - rwmh_scaling: 1D numpy array of scaling factors for the RWMH algorithm.
    - rwmh_cov: 2D numpy array for the RWMH covariance.
    - rwmh_adaptive: Boolean indicating if RWMH is adaptive.
    - iterations: Integer for the number of iterations.
    - burnin: Integer for the burn-in period.
    - n_chains: Integer for the number of chains.
    - final_model: The model object with the param_inverse method.
    - algo: String indicating the algorithm to use.
    
    Returns:
    - best_estimate: The best parameter estimate.
    - best_error: The error corresponding to the best estimate.
    """
    
    # Initialize error and estimate arrays
    error_shape = (len(sigma_noise), len(n_data), len(sigma), len(rwmh_scaling))
    error = np.zeros(error_shape)
    estimates = np.zeros(error_shape)
    
    # Iterate over all combinations of parameters using itertools.product
    for (i, noise), (k, n), (t, s), (j, r) in product(enumerate(sigma_noise), enumerate(n_data), enumerate(sigma), enumerate(rwmh_scaling)):
        t_eval = np.linspace(0., 5., n).reshape(-1, 1)  # Generate evaluation times
        nearest_x, y_obs = process_data(datahf, parameters, t_eval, Yhf)  # Get nearest x values and observations
        cov_likelihood = calculate_cov_likelihood(s, t_eval)  # Compute the covariance for the likelihood
        
        class_model.impose_input(t_eval, y_obs)

        # Perform parameter estimation and calculate error
        estimates[i, k, t, j], error[i, k, t, j] = final_model.param_inverse(
            mean_prior, t_eval, cov_prior=cov_prior, rmwh_scaling=r, 
            cov_noise=noise, cov_likelihood=cov_likelihood, y_obs=y_obs, 
            x_real=parameters, number_chains=n_chains, N=iterations, 
            burn_in=burnin, diagnostic=True, rwmh_cov=rwmh_cov, 
            rwmh_adaptive=rwmh_adaptive, algo=algo
        )
    
    # Identify the index of the minimum error
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, "
          f"rwmh_scaling={rwmh_scaling[smallest_index[3]]}")

    return best_estimate, best_error

def run_simulation_cuqi(
    data: dict,
    mean_prior: np.ndarray,
    x_real: np.ndarray,
    N: int,
    burn_in: int,
    cov_prior: np.ndarray,
    sd_noise: list,
    adapt: bool,
    proposal_sd: list,
    number_chains: int,
    algo: str,
    x_data: np.ndarray,
    n_data: list,
    final_model,
    parallel: bool
) -> tuple:
    """
    Run a CUQI simulation to estimate parameters and compute error.

    Args:
        data (dict): Dictionary containing high-fidelity data (keys: "xhf" and "Yhf").
        mean_prior (np.ndarray): Prior mean vector.
        x_real (np.ndarray): Real x values.
        N (int): Number of samples.
        burn_in (int): Number of burn-in samples.
        cov_prior (np.ndarray): Prior covariance matrix.
        sd_noise (list): List of noise standard deviations to evaluate.
        adapt (bool): Whether to use adaptation in the algorithm.
        proposal_sd (list): List of proposal standard deviations to evaluate.
        number_chains (int): Number of MCMC chains.
        algo (str): Algorithm to use for MCMC.
        x_data (np.ndarray): Initial evaluation times.
        n_data (list): List of data sizes to evaluate.
        final_model: Final model object with inverse_cuqi method.
        parallel (bool): Whether to run MCMC chains in parallel.

    Returns:
        tuple: Best estimate and best error found during the simulation.
    """

    # Initialize estimates and error arrays
    estimates = np.zeros((len(sd_noise), len(n_data), len(proposal_sd)))
    error = np.zeros((len(sd_noise), len(n_data), len(proposal_sd)))

    # Iterate over noise levels
    for i, noise in enumerate(sd_noise):
        # Iterate over number of data points
        for k, n in enumerate(n_data):
            # Generate evaluation times
            x_data = np.linspace(0., 5., n).reshape(-1, 1)
            nearest_x, y_obs = process_data(data["xhf"], x_real, x_data, data["Yhf"])
            
            # Iterate over proposal standard deviations
            for t, s in enumerate(proposal_sd):

                # Perform parameter estimation and calculate error
                estimates[i, k, t], error[i, k, t] = final_model.inverse_cuqi(
                    mean_prior=mean_prior,
                    x_real=x_real,
                    y_obs=y_obs,
                    N=N,
                    burn_in=burn_in,
                    cov_prior=cov_prior,
                    sd_noise=noise,  
                    adapt=adapt,
                    scale=s,
                    proposal_sd=s,
                    number_chains=number_chains,
                    algo=algo,
                    x_data=x_data,
                    parallel=parallel
                )
                
    # Find the smallest error and corresponding indices
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sd_noise={sd_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, proposal_standard_deviation={proposal_sd[smallest_index[2]]}")

    return best_estimate, best_error



