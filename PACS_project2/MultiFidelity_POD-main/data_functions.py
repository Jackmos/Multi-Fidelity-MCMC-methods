import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Tuple, List
from multifidelity_NN_functions import compute_randomized_SVD
import scipy.io
from celluloid import Camera
from matplotlib.animation import PillowWriter
from IPython.display import HTML
import h5py
import pickle
from module_utils import *
sys.path.append('../utils')
from Structure import *
class ReactionDiffusionData:

    def __init__(self, path: str, tlf_0: float = 0.0, thf_0: float = 0.0, Tlf: float = 80.0, Thf: float = 40.0,
                 dt: float = 0.05, mu_0: float = 0.5, mu_1: float = 1.5, N_mu_train: int = 10, small_data: bool = True):
        """
        Initialize the ReactionDiffusionData object with given parameters and paths.
        
        :param path: Path to the dataset.
        :param tlf_0: Initial time for low-fidelity data.
        :param thf_0: Initial time for high-fidelity data.
        :param Tlf: Final time for low-fidelity data.
        :param Thf: Final time for high-fidelity data.
        :param dt: Time step size.
        :param mu_0: Start of parameter range for training data.
        :param mu_1: End of parameter range for training data.
        :param N_mu_train: Number of training parameters.
        :param small_data: Boolean flag to indicate if the dataset is small.
        """

        self.n_POD=None
        self.basis=None
        self.fwd_uLF=[]

        self.tlf_0 = tlf_0
        self.thf_0 = thf_0
        self.Tlf = Tlf
        self.Thf = Thf
        self.dt = dt
        
        self.small_data = small_data
        self.path = path
        
        self.mu_0 = mu_0
        self.mu_1 = mu_1

        self.N_mu_train = N_mu_train
        self.mu_train = np.linspace(self.mu_0, self.mu_1, self.N_mu_train, endpoint=True)

        self.Nx_lf = self.Ny_lf = self.Nt_lf_test = self.N_mu_test = None
        self.Nx_hf = self.Ny_hf = self.Nt_hf_test = None

        self.Nt_test = self.Nt_lf_test = self.Nt_hf_test = None
        self.u_hf_test = self.t_hf_test = None
        self.u_lf_test = self.t_lf_test = None

        self.Nt_lf_train = self.N_mu_train = None
        self.Nt_hf_train = self.N_mu_train = None

        self.Nt_train = self.Nt_lf_train = self.Nt_hf_train = None
        self.u_hf = self.x_HF = self.t_hf = None
        self.u_lf = self.x_LF = self.t_lf = None

        self._set_test_parameters()
        self._load_train_data()
        self._load_test_data()

        self.u_train_POD=None
        self.u_test_POD=None

    def _set_test_parameters(self) -> None:
        """
        Set test parameters based on the size of the dataset.
        """
        if self.small_data:
            self.N_mu_test = 2
            self.mu_test = np.array([0.875, 1.375])
        else:
            self.N_mu_test = 25
            self.mu_test = np.linspace(self.mu_0, self.mu_1, self.N_mu_test, endpoint=True)

    def get_shape_space_lf(self) -> Tuple[int, int]:
        """
        Get the shape of the low-fidelity space.
        
        :return: A tuple representing the dimensions of the low-fidelity space.
        """
        return self.Nx_lf, self.Ny_lf

    def get_shape_space_hf(self) -> Tuple[int, int]:
        """
        Get the shape of the high-fidelity space.
        
        :return: A tuple representing the dimensions of the high-fidelity space.
        """
        return self.Nx_hf, self.Ny_hf

    def get_shape_data_train(self) -> Tuple[int, int]:
        """
        Get the shape of the training data.
        
        :return: A tuple representing the number of training parameters and time steps.
        """
        return self.N_mu_train, self.Nt_train
    
    def get_shape_data_test(self) -> Tuple[int, int]:
        """
        Get the shape of the test data.
        
        :return: A tuple representing the number of test parameters and time steps.
        """
        return self.N_mu_test, self.Nt_test
    
    def set_n_POD(self, n_POD:int=9)-> None:
        self.n_POD=n_POD


    def _load_data(self, params: np.ndarray, fidelity: str, path: str, splitted: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from the given path based on the specified fidelity and parameters.
        
        :param params: Array of parameter values.
        :param fidelity: Fidelity level ('LF' for low-fidelity, 'HF' for high-fidelity).
        :param path: Path to the dataset.
        :param splitted: Boolean flag to indicate if the data is split into multiple files.
        :return: A tuple containing the data array, spatial coordinates, and temporal coordinates.
        """
        u_list = []
        for param in params:
            name = f"{path}u_{fidelity}_{param:.3f}"
            
            if splitted:
                u_test_list = [scipy.io.loadmat(f"{name}_{i}.mat")['u'] for i in [1, 2]]
                u = np.concatenate(u_test_list, axis=2)
            else:
                u = scipy.io.loadmat(f"{name}.mat")['u']
                
            u_list.append(u)
        
        data_u = np.stack(u_list, axis=3)
        x = scipy.io.loadmat(f"{path}x_{fidelity}.mat")['x']
        t = scipy.io.loadmat(f"{path}t_{fidelity}.mat")['t']
        
        return data_u, x.flatten(), t.flatten()      

    def _load_train_data(self) -> None:
        """
        Load training data.
        """
        self.u_lf, self.x_LF, self.t_lf = self._load_data(self.mu_train, 'LF', f"{self.path}train/")
        self.u_hf, self.x_HF, self.t_hf = self._load_data(self.mu_train, 'HF', f"{self.path}train/")

        self.Nx_lf, self.Ny_lf, self.Nt_lf_train, self.N_mu_train = self.u_lf.shape
        self.Nx_hf, self.Ny_hf, self.Nt_hf_train, self.N_mu_train = self.u_hf.shape

        self.Nt_train = self.Nt_lf_train = self.Nt_hf_train


    def _load_test_data(self) -> None:
        """
        Load test data.
        """
        self.u_lf_test, _, self.t_lf_test = self._load_data(self.mu_test, 'LF', f"{self.path}test/")
        self.u_hf_test, _, self.t_hf_test = self._load_data(self.mu_test, 'HF', f"{self.path}test/", splitted=True)

        self.Nx_lf, self.Ny_lf, self.Nt_lf_test, self.N_mu_test = self.u_lf_test.shape
        self.Nx_hf, self.Ny_hf, self.Nt_hf_test, self.N_mu_test = self.u_hf_test.shape

        self.Nt_test = self.Nt_lf_test = self.Nt_hf_test

    def visualize_train_data(self) -> None:
        """
        Visualize training data for both low-fidelity (u_lf) and high-fidelity (u_hf) data.
        This function accounts for possible differences in dimensionality between the two datasets.
        """
        # Adjust ticks based on the higher fidelity dataset
        loc_hf = np.arange(0, self.Nx_hf + 1, int(self.Nx_hf / 4))
        tick_hf = np.arange(0, 21, 5)
        idx_mu_list = [0, int(self.N_mu_train / 2), -1]

        # Plot for u_lf
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        for i, (ax, idx_mu) in enumerate(zip(axes, idx_mu_list)):
            u_lf_data = self.u_lf[:, :, -1, idx_mu]

            # Determine the shape for low-fidelity data
            Nx_lf = u_lf_data.shape[1]
            Ny_lf = u_lf_data.shape[0]

            loc_lf = np.arange(0, Nx_lf + 1, int(Nx_lf / 4))
            tick_lf = np.linspace(0, 20, len(loc_lf))

            # Plotting low-fidelity data
            surf = ax.imshow(u_lf_data, origin='lower', aspect='auto', cmap='viridis')

            # Set axis ticks and labels for low-fidelity data
            ax.set_xticks(loc_lf)
            ax.set_xticklabels(np.round(tick_lf).astype(int), fontsize=20)
            ax.set_yticks(loc_lf)
            ax.set_yticklabels(np.round(tick_lf).astype(int), fontsize=20)

            # Set labels and title
            ax.set_xlabel('x', rotation=0, fontsize=22)
            ax.set_ylabel('y', rotation=0, fontsize=22, labelpad=15)
            ax.set_title(f'Low-fidelity \n $\\mu = ${round(self.mu_train[idx_mu], 2)}, $t= ${round(self.t_lf[-1])}', fontsize=22)

            # Add colorbar
            cbar = plt.colorbar(surf, ax=ax)
            cbar.ax.tick_params(labelsize=20, pad=1)
        
        plt.tight_layout()
        plt.show()

        # Plot for u_hf
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        for i, (ax, idx_mu) in enumerate(zip(axes, idx_mu_list)):
            u_hf_data = self.u_hf[:, :, -1, idx_mu]

            # Plotting high-fidelity data
            surf = ax.imshow(u_hf_data, origin='lower', aspect='auto', cmap='viridis')

            # Set axis ticks and labels for high-fidelity data
            ax.set_xticks(loc_hf)
            ax.set_xticklabels(tick_hf, fontsize=20)
            ax.set_yticks(loc_hf)
            ax.set_yticklabels(tick_hf, fontsize=20)

            # Set labels and title
            ax.set_xlabel('x', rotation=0, fontsize=22)
            ax.set_ylabel('y', rotation=0, fontsize=22, labelpad=15)
            ax.set_title(f'High-fidelity \n $\\mu = ${round(self.mu_train[idx_mu], 2)}, $t= ${round(self.t_lf[-1])}', fontsize=22)

            # Add colorbar
            cbar = plt.colorbar(surf, ax=ax)
            cbar.ax.tick_params(labelsize=20, pad=1)
        
        plt.tight_layout()
        plt.show()

    def interpolate_data(self) -> None:
        """
        Interpolate low-fidelity data to match high-fidelity resolution.
        """
        coord_x_LF, coord_y_LF = np.meshgrid(self.x_LF, self.x_LF)
        coord_LF = np.stack((coord_x_LF, coord_y_LF), axis=2).reshape(-1, 2)

        coord_x_HF, coord_y_HF = np.meshgrid(self.x_HF, self.x_HF)
        coord_HF = np.stack((coord_x_HF, coord_y_HF), axis=2).reshape(-1, 2)

        self.u_lf = griddata(coord_LF, self.u_lf.reshape(-1, self.Nt_train, self.N_mu_train), coord_HF, method='nearest').reshape(self.Nx_hf, self.Ny_hf, self.Nt_train, self.N_mu_train)
        self.u_lf_test = griddata(coord_LF, self.u_lf_test.reshape(-1, self.Nt_lf_test, self.N_mu_test), coord_HF, method='nearest').reshape(self.Nx_hf, self.Ny_hf, self.Nt_lf_test, self.N_mu_test)

        self.N = self.Nx_hf * self.Ny_hf
    

    def _import_uLF_POD_evaluation(self, foldername:str=None) -> None:

        list_models = [f"{foldername}{i}.keras" for i in range(1, self.n_POD + 1)]

        for models in list_models:
            
            mod=NetworkFactory.build_network(network_type="LSTM_support")
            mod.load(models)
            self.fwd_uLF.append(mod)



    def _forward_low_fidelity(self, x_final: np.ndarray, data_points: np.ndarray, fwd_LSTM_folder: str) -> np.ndarray:
        """
        Generate low fidelity model using POD basis.
        Useful for BIP 

        Parameters:
        - x_final (np.ndarray): Final input data, parameter \mu.
        - data_points (np.ndarray): Data points, in this specific case time instants.
        - fwd_LSTM_folder (str): name of the folder and files with collection of LSTM models with the relation (mu,t)->u_LF_POD
        Returns:
        - new_inputs (np.ndarray): New input data incorporating low fidelity model.
        """
        
        if not self.fwd_uLF:
            self._import_uLF_POD_evaluation(fwd_LSTM_folder)
            
        data_norm=normalization(data_points, self.Thf,0.)# completa con self. tmax, self.t min
        x_result=normalization(x_final, self.mu_1, self.mu_0)

        domain =np.concatenate((x_result, data_norm),axis=1)  # (\mu,t)
        domain=domain.reshape(1,domain.shape[0], domain.shape[1]) 
        # # check dimensione 
        # x_final=domain[:, :, [1, 0]]
        # x_final[:,:,0]=denormalization(x_final[:,:,0], self.Thf,0.)
        # x_final[:,:,1]=denormalization(x_final[:,:,1], self.mu_1, self.mu_0)

        x_final=domain[:, :, [1]]
        x_final[:,:,0]=denormalization(x_final[:,:,1], self.mu_1, self.mu_0)

        for l in range(self.n_POD):
            x_final=np.concatenate((x_final, denormalization(self.fwd_uLF[l].prediction(domain),np.max(self.u_train_POD[l]),np.min(self.u_train_POD[l]))),axis=2)    # denormalized with u_LF because bigger set 
        

        return x_final




    def compute_POD(self, u_data: np.ndarray, n_POD_large: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Proper Orthogonal Decomposition (POD) of the given data.

        :param u_data: Data to perform POD on, with dimensions (Nx, Ny, Nt * Nmu).
        :param n_POD_large: Number of POD modes to compute.
        :return: POM_u: POD modes, S_u: Singular values.
        """
        u_pod = np.reshape(u_data, (self.Nx_hf * self.Ny_hf, self.Nt_train * self.N_mu_train), 'F')
        POM_u, S_u = compute_randomized_SVD(u_pod, n_POD_large, self.N, 1)
        return POM_u, S_u

    def project_onto_POD_train(self, POM_u: np.ndarray, u_data: np.ndarray) -> np.ndarray:
        """
        Project training data onto the POD basis.

        :param POM_u: POD modes obtained from compute_POD.
        :param u_data: Training data to project, with dimensions (Nx, Ny, Nt_train, N_mu_train).
        :return: Projected training data onto the POD basis, with dimensions (N_mu_train, Nt_train, n_POD_large).
        """
        u_pod = np.reshape(u_data, (self.N, self.Nt_train * self.N_mu_train), 'F')
        u_train = u_pod.T @ POM_u
        self.u_train_POD=np.reshape(u_train, (self.N_mu_train, self.Nt_train, -1))

        return self.u_train_POD
    
    def project_onto_POD_test(self, POM_u: np.ndarray, u_data: np.ndarray) -> np.ndarray:
        """
        Project test data onto the POD basis.

        :param POM_u: POD modes obtained from compute_POD.
        :param u_data: Test data to project, with dimensions (Nx, Ny, Nt_test, N_mu_test).
        :return: Projected test data onto the POD basis, with dimensions (N_mu_test, Nt_test, n_POD_large).
        """
        u_pod = np.reshape(u_data, (self.N, self.Nt_test * self.N_mu_test), 'F')
        u_test = u_pod.T @ POM_u
        self.u_test_POD=np.reshape(u_test, (self.N_mu_test, self.Nt_test, -1))
        return self.u_test_POD

    def plot_POD_coefficients(self, ulf_train: np.ndarray, uhf_train: np.ndarray, n_POD: int) -> None:
        """
        Plot POD coefficients for training data.

        :param ulf_train: Low-fidelity training data, with dimensions (Nx_lf, Ny_lf, Nt_train, n_POD_large).
        :param uhf_train: High-fidelity training data, with dimensions (Nx_hf, Ny_hf, Nt_train, n_POD_large).
        :param n_POD: Number of POD modes to plot.
        """
        fig = plt.figure(figsize=(15, 15))
        for i in range(n_POD):
            ax = fig.add_subplot(331 + i)
            plt.plot(self.t_lf, ulf_train[-1, :, i], label='LF')
            plt.plot(self.t_hf, uhf_train[-1, :, i], label='HF')
            ax.title.set_text(f'POM {i + 1}')
            plt.xlabel('t')
            plt.legend()
        plt.show()

    def comparison_POD_modes(self, n_POD: int, uMF_LSTM_test: np.ndarray, ulf_test: np.ndarray, uhf_test: np.ndarray) -> None:
        """
        Compare POD modes for different models.

        :param n_POD: Number of POD modes to compare.
        :param uMF_LSTM_test: Predictions from the MF-LSTM model, with dimensions (N_mu_test, Nt_test, n_POD_large).
        :param ulf_test: Low-fidelity test data, with dimensions (Nx_lf, Ny_lf, Nt_test, N_mu_test).
        :param uhf_test: High-fidelity test data, with dimensions (Nx_hf, Ny_hf, Nt_test, N_mu_test).
        """
        for mu in range(self.N_mu_test): 
            fig = plt.figure(figsize=(44, 19)) 
            plt.title('$\\mu$ = ' + str(self.mu_test[mu].round(3)) + '\n', fontsize=40)
            plt.axis('off')
            for POD_mode in range(8):
                ax = fig.add_subplot(241 + POD_mode)

                plt.plot(self.t_hf_test, uhf_test[mu, :, POD_mode],'-', color='#ff7f00', label='HF', linewidth=9)
                plt.plot(self.t_hf_test, uMF_LSTM_test[mu, :, POD_mode],'k-', label='MF', linewidth=3.5)
                plt.plot(self.t_hf_test, ulf_test[mu, :, POD_mode], '--', label='LF', color='#4daf4a', linewidth=3.2)

                plt.axvline(x=self.Thf, linestyle='--', color='black', linewidth=4)
                ax.set_title('POM ' + str(POD_mode + 1), fontsize=26)
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                plt.xlabel('t', rotation=0, fontsize=32)
            plt.show()

    def comparison_solutions(self, times: List[float], uMF_LSTM_test_rec: np.ndarray) -> None:
        """
        Compare solutions at specific times.

        :param times: List of specific times to compare.
        :param uMF_LSTM_test_rec: Reconstructed predictions from the MF-LSTM model, with dimensions (Nx_hf, Ny_hf, Nt_test, N_mu_test).
        """
        loc = np.arange(0, self.Nx_hf + 1, int(self.Nx_hf / 4))
        tick = np.arange(0, 21, 5)
        abs_err_test = np.abs(uMF_LSTM_test_rec - self.u_hf_test)
        
        for mu in range(self.N_mu_test):
            for time in times:
                fig = plt.figure(figsize=(30, 6))
                t = int(time / self.dt)
                plt.title(f'$\\mu = ${round(self.mu_test[mu], 3)}, $t = ${round(t * self.dt, 1)}\n', fontsize=24)
                plt.axis('off')

                ax = fig.add_subplot(141)
                surf = ax.imshow(self.u_lf_test[:, :, t, mu], origin='lower', vmin=-1, vmax=1)
                plt.xticks(loc, tick, fontsize=20)
                plt.yticks(loc, tick, fontsize=20)
                plt.xlabel('x', rotation=0, fontsize=22)
                plt.ylabel('y', rotation=0, fontsize=22, labelpad=13)
                ax.set_title('LF input', fontsize=22)
                cbar = plt.colorbar(surf)
                cbar.ax.tick_params(labelsize=20, pad=1)
                
                ax = fig.add_subplot(142)
                surf = ax.imshow(uMF_LSTM_test_rec[:, :, t, mu], origin='lower', vmin=-1, vmax=1)
                plt.xticks(loc, tick, fontsize=20)
                plt.yticks(loc, tick, fontsize=20)
                plt.xlabel('x', rotation=0, fontsize=22)
                plt.ylabel('y', rotation=0, fontsize=22, labelpad=13)
                ax.set_title('MF prediction', fontsize=22)
                cbar = plt.colorbar(surf)
                cbar.ax.tick_params(labelsize=20, pad=1)

                ax = fig.add_subplot(143)
                surf = ax.imshow(self.u_hf_test[:, :, t, mu], origin='lower', vmin=-1, vmax=1)
                plt.xticks(loc, tick, fontsize=20)
                plt.yticks(loc, tick, fontsize=20)
                plt.xlabel('x', rotation=0, fontsize=22)
                plt.ylabel('y', rotation=0, fontsize=22, labelpad=13)
                ax.set_title('HF reference', fontsize=22)
                cbar = plt.colorbar(surf)
                cbar.ax.tick_params(labelsize=20, pad=1)
                
                ax = fig.add_subplot(144)
                surf = ax.imshow(abs_err_test[:, :, t, mu], origin='lower', cmap='bwr')
                plt.xticks(loc, tick, fontsize=20)
                plt.yticks(loc, tick, fontsize=20)
                plt.xlabel('x', rotation=0, fontsize=22)
                plt.ylabel('y', rotation=0, fontsize=22, labelpad=13)
                ax.set_title('Abs. error', fontsize=22)
                cbar = plt.colorbar(surf)
                cbar.ax.tick_params(labelsize=20, pad=1)
                plt.show()
##########                
    def video_creation(self,uMF_LSTM_test_rec):
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))

        camera = Camera(fig)
        mu = 0
        for i in range(0, 1600, 5):
            surf = ax[0].imshow(self.u_lf_test[:, :, i, mu], vmin=-1, vmax=1)
            surf = ax[1].imshow(uMF_LSTM_test_rec[:, :, i, mu], vmin=-1, vmax=1)
            surf = ax[2].imshow(self.u_hf_test[:, :, i, mu], vmin=-1, vmax=1)
            surf = ax[3].imshow(np.abs(uMF_LSTM_test_rec[:, :, i, mu] - self.u_hf_test[:, :, i, mu]),cmap='bwr')
            
            ax[0].set_title("LF input", fontsize = 20)
            ax[1].set_title("MF-POD pred.", fontsize = 20)
            ax[2].set_title("HF reference", fontsize = 20)
            ax[3].set_title("Abs. error", fontsize = 20)
            
            camera.snap()
        fig.suptitle("$\mu$ = " + str(round(self.mu_test[mu],3)), fontsize = 22)
        animation = camera.animate()

        # Close the figure to prevent it from being displayed
        plt.close(fig)

        # Save the animation as a GIF using PillowWriter
        name = "./results/test_HF_" + str(round(self.mu_test[mu],3)) + ".gif"
        animation.save(name, writer=PillowWriter(fps=10))

        # Display the saved GIF
        HTML("<img src='" + name + "'>")