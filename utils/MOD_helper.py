import numpy as np
from numpy import newaxis as _
import matplotlib.pyplot as plt
from sklearn.utils import extmath
from typing import Any, Optional, List, Tuple
import scipy.io as sio
from utils.helper_functions import Clean

class ROM:
    
    def __init__(self, model: Any, n_POD: int):
        """
        Initialize the Reduced Order Model (ROM) class.

        Parameters:
        - burger_eq (Any): An instance of the Burger equation solver or model.
        - n_POD (int): The number of Proper Orthogonal Decomposition (POD) modes to retain.
        """
        self.model = model
        self.n_POD = n_POD
        
        if hasattr(self.model, 'get_input_dimensions') and callable(getattr(self.model, 'get_input_dimensions')):
            self.M_mu_train, self.M_mu_test, self.M_t_train,self.M_t_test = self.model.get_input_dimensions()
            self.M_test=self.M_mu_test*self.M_t_test
            self.M_train=self.M_mu_train*self.M_t_train

        else:
            self.M_mu_train=None
            self.M_mu_test=None
            self.M_t_train=None
            self.M_t_test=None
            self.M_train = None
            self.M_test = None

        # spatial discretization of the n- dimensional domain
        if hasattr(self.model, 'get_space_discretization') and callable(getattr(self.model, 'get_space_discretization')):
            self.N_lf, self.N_hf = self.model.get_space_discretization()
        else:
            # product of discretization on x and y if 2D
            # discretization on the line if 1D
            self.N_lf = None
            self.N_hf = None


        self.basis: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        
        # multi - fidelity POD values
        self.u_lf_pod: Optional[np.ndarray] = None
        self.u_hf_pod: Optional[np.ndarray] = None
        self.u_lf_pod_test: Optional[np.ndarray] = None
        self.u_hf_pod_test: Optional[np.ndarray] = None


    @staticmethod
    def compute_randomized_SVD(
        S: np.ndarray, 
        N_POD: int, 
        N_h: int, 
        n_channels: int, 
        name: str = '', 
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the randomized Singular Value Decomposition (SVD) for the input matrix S.
        The function computes the SVD separately for each channel in the dataset and 
        returns the left singular vectors (U) and singular values (Sigma).

        Parameters:
        -----------
        S : np.ndarray
            The input matrix with shape (n_channels * N_h, num_samples) where 
            num_samples is the number of samples in the dataset.
        N_POD : int
            The number of Principal Orthogonal Directions (POD) to retain.
        N_h : int
            The height of each channel, i.e., the number of rows associated with each channel.
        n_channels : int
            The number of channels in the dataset.
        name : str, optional
            The filename to save the left singular vectors (U). If not provided, the matrix is not saved.
        verbose : bool, optional
            If True, prints additional information during computation.

        Returns:
        --------
        U : np.ndarray
            The matrix containing left singular vectors for each channel, with shape 
            (n_channels * N_h, N_POD).
        Sigma : np.ndarray
            The matrix containing the singular values for each channel, with shape 
            (n_channels, N_POD).
        """

        if verbose:
            print('Computing randomized POD...')

        U = np.zeros((n_channels * N_h, N_POD))
        Sigma = np.zeros((n_channels, N_POD))  # Store singular values for each channel

        for i in range(n_channels):
            start_idx = i * N_h
            end_idx = (i + 1) * N_h
            # Perform randomized SVD on each channel
            U[start_idx:end_idx], Sigma, _ = extmath.randomized_svd(
                S[start_idx:end_idx, :],
                n_components=N_POD,
                transpose=False,
                flip_sign=False,
                random_state=123  # Fixing random state for reproducibility
            )
            #Sigma[i, :] = sigma  # Store singular values for the current channel     # <----------

        if verbose:
            # Calculate the information loss (1 - cumulative explained variance)
            I = 1.0 - np.cumsum(np.square(Sigma)) / np.sum(np.square(Sigma))
            print(f"Information loss at last component: {I[-1]}")

        if name:
            # Save the left singular vectors to a .mat file if a filename is provided
            sio.savemat(name, {'V': U[:, :N_POD]})

        return U, Sigma
    
    def get_basis(self) -> Optional[np.ndarray]:
        """
        Retrieve the POD basis.
        
        Returns:
        - Optional[np.ndarray]: The POD basis or None if not yet computed.
        """
        return self.basis


    def compute_POD_basis(self, n_POD_: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Proper Orthogonal Decomposition (POD) of the given data.

        :param u_data: Data to perform POD on, with dimensions (Nx, Ny, Nt * Nmu).
        :param n_POD_large: Number of POD modes to compute.
        :return: POM_u: POD modes, S_u: Singular values.
        """
        if n_POD_ is not None:
            self.n_POD=n_POD_

        if self.model.u_hf.ndim==3:    
            u_pod = np.reshape(self.model.u_hf, (self.M_train, self.N_hf))
        elif self.model.u_hf.ndim==4:
            u_pod = np.reshape(self.model.u_hf, (self.N_hf,self.M_train),'F')   # .T?
        else:
            raise ValueError(f"Input has not an appropriate dimension")
        
        self.basis, self.S = self.compute_randomized_SVD(u_pod, self.n_POD, self.N_hf, 1)

        if self.model.u_hf.ndim==3:    
            self.u_POD_hf_train=u_pod
        elif self.model.u_hf.ndim==4:
            self.u_POD_hf_train=u_pod.T
        return u_pod,self.basis, self.S

    def projection(self, u_pod: np.ndarray, num_basis: Optional[int], M_mu: Optional[int] = None, M_t: Optional[int] = None) -> np.ndarray:
        """
        Project the given POD data onto a reduced basis.

        Args:
        - u_pod: Input POD data (numpy array).
        - num_basis: Number of basis vectors to use for the projection. If None, uses all basis vectors.
        - M_mu: Number of parameter samples (optional). Defaults to the class attribute M_mu_train if not provided.
        - M_t: Number of time steps (optional). Defaults to the class attribute M_t_train if not provided.

        Returns:
        - u: Projected data, reshaped to dimensions (M_mu, M_t, -1).

        Raises:
        - ValueError: If the basis is not initialized (self.basis is None).
        """
        if M_mu is None:
            M_mu = self.M_mu_train
        if M_t is None: 
            M_t = self.M_t_train

        if self.basis is None:
            raise ValueError("Missing basis!")
        else:
            if num_basis is not None:
                u = u_pod @ self.basis[:, :num_basis]                 
            else:
                u = u_pod @ self.basis

        return np.reshape(u, (M_mu, M_t, -1))



    def project_onto_POD_test(self,num_basis:int=None) -> np.ndarray:
        """
        Project test data onto the POD basis.

        :param POM_u: POD modes obtained from compute_POD.
        :param u_data: Test data to project, with dimensions (Nx, Ny, Nt_test, N_mu_test).
        :return: Projected test data onto the POD basis, with dimensions (N_mu_test, Nt_test, n_POD_large).
        """

        if self.model.u_lf_test.ndim==3:    
            u_pod_lf = np.reshape(self.model.u_lf_test, (self.M_test, self.N_lf))
        elif self.model.u_lf_test.ndim==4: 
            u_pod_lf = np.reshape(self.model.u_lf_test, (self.N_hf, self.M_test),'F').T
        else:
            raise ValueError(f"Input has not an appropriate dimension")


        if self.model.u_lf_test.ndim==3:    
            u_pod_hf = np.reshape(self.model.u_hf_test, (self.M_test, self.N_hf))
        elif self.model.u_lf_test.ndim==4: 
            u_pod_hf = np.reshape(self.model.u_hf_test, (self.N_hf,self.M_test),'F').T
        else:
            raise ValueError(f"Input has not an appropriate dimension")


        self.u_POD_lf_test= self.projection(u_pod_lf,num_basis,self.M_mu_test, self.M_t_test)
        self.u_POD_hf_test= self.projection(u_pod_hf,num_basis,self.M_mu_test, self.M_t_test)


        self.model._POD_ROM_set_test(self.u_POD_lf_test,  self.u_POD_hf_test)

        return self.u_POD_lf_test, self.u_POD_hf_test
    
    def project_onto_POD_train(self, num_basis:int=None) -> np.ndarray:
        """
        Project test data onto the POD basis.

        :param POM_u: POD modes obtained from compute_POD.
        :param u_data: Test data to project, with dimensions (Nx, Ny, Nt_test, N_mu_test).
        :return: Projected test data onto the POD basis, with dimensions (N_mu_test, Nt_test, n_POD_large).
        """
        if self.model.u_lf_test.ndim==3:    
            u_pod_lf = np.reshape(self.model.u_lf, (self.M_train, self.N_lf))
        elif self.model.u_lf_test.ndim==4: 
            u_pod_lf = np.reshape(self.model.u_lf, (self.N_hf, self.M_train),'F').T
        else:
            raise ValueError(f"Input has not an appropriate dimension")

        self.u_POD_lf_train= self.projection(u_pod_lf,num_basis,self.M_mu_train, self.M_t_train)
        self.u_POD_hf_train= self.projection(self.u_POD_hf_train,num_basis,self.M_mu_train, self.M_t_train)

        self.model._POD_ROM_set_train(self.u_POD_lf_train,  self.u_POD_hf_train)

        return self.u_POD_lf_train, self.u_POD_hf_train


    def plot_singular_values_threshold(self, n_POD)->None:

        plt.figure(figsize = (10,4))
        plt.subplot(121)
        plt.plot(self.S,'*-')
        plt.axvline(x = n_POD-1)
        plt.yscale('log')
        plt.xlabel('# bases')

        plt.subplot(122)
        plt.plot(np.cumsum(self.S)/np.sum(self.S),'*-')
        plt.axvline(x = n_POD-1)
        plt.xlabel('# bases')
        plt.show()

    def plot_POD_coefficients(self, ind_re: int) -> None:
        """
        Plot POD coefficients: Low-Fidelity (LF) vs High-Fidelity (HF).
        
        Parameters:
        - ind_re (int): Index of the Reynolds number case to plot.
        """
        fig = plt.figure(figsize=(12, 6))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('POD coefficients: LF vs HF', fontsize=14)
        t = self.model.t

        for mode in range(min(6, self.n_POD)):
            ax = fig.add_subplot(231 + mode)
            ax.plot(t, self.u_POD_lf_train[ind_re, :, mode], label='LF', linewidth=2, color='green', linestyle='--')
            ax.plot(t, self.u_POD_hf_train[ind_re, :, mode], label='HF', linewidth=2, color='blue')
            ax.set_title(f'POD coord. {mode + 1}')
            ax.set_xlabel('t')
            ax.legend()
        plt.show()

    def plot_POD_output(self, ind_re: int, output_pred: np.ndarray) -> None:
        """
        Plot POD output coefficients: predicted vs test data.
        
        Parameters:
        - ind_re (int): Index of the Reynolds number case to plot.
        - output_pred (np.ndarray): Predicted output from the model.
        
        Raises:
        - ValueError: If output_pred is not properly defined or does not match the shape of output_test.
        - ValueError: If output_test is not defined.
        """
        if output_pred is None or output_pred.shape != self.model.output_test.shape:
            raise ValueError("Error: output_pred is not properly defined.")

        if self.model.output_test is None:
            raise ValueError("Error: output_test not defined, run 'perform_POD' before.")

        fig = plt.figure(figsize=(12, 6))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('POD coefficients: Predicted vs Test', fontsize=14)
        t = self.model.t

        for mode in range(min(6, self.n_POD)):
            ax = fig.add_subplot(231 + mode)
            ax.plot(t, self.u_POD_hf_test[ind_re, :, mode].reshape(-1), 'b-', label='Test', linewidth=2)
            ax.plot(t, output_pred[ind_re, :, mode].reshape(-1), 'r--', label='Predicted', linewidth=2)
            ax.set_title(f'POD coord. {mode + 1}')
            ax.set_xlabel('t')
            ax.legend()
        plt.show()


