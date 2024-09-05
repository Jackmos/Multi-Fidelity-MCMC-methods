import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Tuple, List
#from multifidelity_NN_functions import compute_randomized_SVD
import scipy.io

from scipy.spatial import cKDTree
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout
from utils.network_utils  import *
from utils.helper_functions import *

class ReactionDiffusionData:

    def __init__(self, path: str, tlf_0: float = 0.0, thf_0: float = 0.0, Tlf: float = 80.0, Thf: float = 40.0,
                 dt: float = 0.05, mu_0: float = 0.5, mu_1: float = 1.5, N_mu_train: int = 10):
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
        """
        self.path = path            # path to dataset
        # set data
        self.tlf_0 = tlf_0
        self.thf_0 = thf_0
        self.Tlf = Tlf
        self.Thf = Thf
        self.dt = dt
            
        self.mu_0 = mu_0
        self.mu_1 = mu_1

        self.N_mu_train = N_mu_train
        self.mu_train = np.linspace(self.mu_0, self.mu_1, self.N_mu_train, endpoint=True)

        

        self.Nx_lf = self.Ny_lf = self.Nt_lf_test = self.N_mu_test = None    # lf dimensions
        self.Nx_hf = self.Ny_hf = self.Nt_hf_test = None                     # hf dimensions

        self.Nt_test = self.Nt_lf_test = None              # time test dimensions
        self.u_hf_test = self.t_hf_test = None
        self.u_lf_test = self.t_lf_test = None

        self.Nt_lf_train = self.N_mu_train = None
        self.Nt_hf_train = None

        self.Nt_train = self.Nt_lf_train = self.Nt_hf_train = None          # time train dimensions
        self.u_hf = self.x_HF = self.t_hf = None                            # high fidelity variables
        self.u_lf = self.x_LF = self.t_lf = None                            # low fidelity variables

        self._set_test_parameters()
        self._load_train_data()
        self._load_test_data()

        self.u_POD_lf_train= self.u_POD_lf_test=self.u_POD_hf_train= self.u_POD_hf_test=None                              # POD version variables


        # POD data useful for BIP
        self.n_POD=None
        self.basis=None
        self.fwd_uLF=[]



    def get_input_dimensions(self)-> Tuple[int,int,int, int]:
        
        return self.N_mu_train, self.N_mu_test, self.Nt_train,self.Nt_test


    def get_space_discretization(self)-> Tuple[int,int]:


        self.Nx_lf, self.Ny_lf=self.get_shape_space_lf()    # x and y low fidelity
        self.Nx_hf, self.Ny_hf=self.get_shape_space_hf()    # x and y high-fidelity    
        return self.Nx_lf*self.Ny_lf, self.Nx_hf*self.Ny_hf



    def _set_test_parameters(self) -> None:
        """
        Set test parameters 
        """
        self.N_mu_test = 2
        self.mu_test = np.array([0.875, 1.375])

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

    def visualize_train_data(self, save_dir: str = "plots_reaction_diffusion") -> None:
        """
        Visualize training data for both low-fidelity (u_lf) and high-fidelity (u_hf) data.
        This function accounts for possible differences in dimensionality between the two datasets.
        Saves the plots in the specified directory.
        
        :param save_dir: Directory where the plots will be saved. Defaults to 'reaction_diff'.
        """

        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Adjust ticks based on the higher fidelity dataset
        loc_hf = np.arange(0, self.Nx_hf + 1, int(self.Nx_hf / 4))
        tick_hf = np.arange(0, 21, 5)
        idx_mu_list = [0, int(self.N_mu_train / 2), -1]

        # Plot for u_lf
        for i, idx_mu in enumerate(idx_mu_list):
            plt.figure(figsize=(7, 6))
            u_lf_data = self.u_lf[:, :, -1, idx_mu]

            # Determine the shape for low-fidelity data
            Nx_lf = u_lf_data.shape[1]
            Ny_lf = u_lf_data.shape[0]

            loc_lf = np.arange(0, Nx_lf + 1, int(Nx_lf / 4))
            tick_lf = np.linspace(0, 20, len(loc_lf))

            loc_lfy = np.arange(0, Ny_lf + 1, int(Ny_lf / 4))
            tick_lfy = np.linspace(0, 20, len(loc_lfy))

            # Plotting low-fidelity data
            surf = plt.imshow(u_lf_data, origin='lower', aspect='auto', cmap='viridis')

            # Set axis ticks and labels for low-fidelity data
            plt.xticks(loc_lf, np.round(tick_lf).astype(int), fontsize=20)
            plt.yticks(loc_lfy, np.round(tick_lfy).astype(int), fontsize=20)

            # Set labels and title
            plt.xlabel('x', rotation=0, fontsize=22)
            plt.ylabel('y', rotation=0, fontsize=22, labelpad=15)
            plt.title(f'Low-fidelity \n $\\mu = ${round(self.mu_train[idx_mu], 2)}, $t= ${round(self.t_lf[-1])}', fontsize=22)

            # Add colorbar
            cbar = plt.colorbar(surf)
            cbar.ax.tick_params(labelsize=20, pad=1)

            # Save plot to the directory
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'low_fidelity_mu_{idx_mu}.png'))
            plt.close()

        # Plot for u_hf
        for i, idx_mu in enumerate(idx_mu_list):
            plt.figure(figsize=(7, 6))
            u_hf_data = self.u_hf[:, :, -1, idx_mu]

            # Plotting high-fidelity data
            surf = plt.imshow(u_hf_data, origin='lower', aspect='auto', cmap='viridis')

            # Set axis ticks and labels for high-fidelity data
            plt.xticks(loc_hf, tick_hf, fontsize=20)
            plt.yticks(loc_hf, tick_hf, fontsize=20)

            # Set labels and title
            plt.xlabel('x', rotation=0, fontsize=22)
            plt.ylabel('y', rotation=0, fontsize=22, labelpad=15)
            plt.title(f'High-fidelity \n $\\mu = ${round(self.mu_train[idx_mu], 2)}, $t= ${round(self.t_lf[-1])}', fontsize=22)

            # Add colorbar
            cbar = plt.colorbar(surf)
            cbar.ax.tick_params(labelsize=20, pad=1)

            # Save plot to the directory
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'high_fidelity_mu_{idx_mu}.png'))
            plt.close()
            
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
        # print(foldername)
        list_models = [f"{foldername}{i}.keras" for i in range(1, self.n_POD + 1)]

        for models in list_models:
            
            mod=NetworkFactory.build_network(NetworkConfig(network_type="LSTM_support"),self.getModel)
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
        
        time_scale=np.linspace(np.min(data_points),np.max(data_points),801).reshape(-1,1)
        mask=time_scale.flatten()
        reference=data_points.flatten()

        tree = cKDTree(mask.reshape(-1, 1))
        _, indices = tree.query(reference.reshape(-1, 1), k=1)

        if not self.fwd_uLF:
            self._import_uLF_POD_evaluation(fwd_LSTM_folder)
            
        time_scale=ReactionDiffusionData.normalization(time_scale, self.Tlf,0.)
        x_result=np.ones(time_scale.shape)*ReactionDiffusionData.normalization(x_final, self.mu_1, self.mu_0)[0,0]

        domain =np.concatenate((x_result, time_scale),axis=1)  # (\mu,t)
        domain=domain.reshape(1,domain.shape[0], domain.shape[1]) 

        x_final=ReactionDiffusionData.denormalization(domain[:, :, [0]][:,indices,:],self.mu_1,self.mu_0)


        for l in range(self.n_POD):
            value=ReactionDiffusionData.denormalization(self.fwd_uLF[l].prediction(domain),np.max(self.u_POD_lf_train[l]),np.min(self.u_POD_lf_train[l]))

            x_final=np.concatenate((x_final, value[:,indices,:]/(np.max(self.u_POD_lf_train))),axis=2)    # denormalized with u_LF because bigger set 
        

        return x_final



    

    def _POD_ROM_set_train(self, u_POD_lf_train,  u_POD_hf_train):



        self.u_POD_lf_train= u_POD_lf_train 
        self.u_POD_hf_train=u_POD_hf_train


    def _POD_ROM_set_test(self, u_POD_lf_test,  u_POD_hf_test):



        self.u_POD_lf_test= u_POD_lf_test 
        self.u_POD_hf_test=u_POD_hf_test




    def plot_POD_coefficients(self, ulf_train: np.ndarray, uhf_train: np.ndarray, n_POD: int, save_dir: str = "plots_reaction_diffusion") -> None:
        """
        Plot POD coefficients for training data and save the plot to the specified directory.

        :param ulf_train: Low-fidelity training data, with dimensions (Nx_lf, Ny_lf, Nt_train, n_POD_large).
        :param uhf_train: High-fidelity training data, with dimensions (Nx_hf, Ny_hf, Nt_train, n_POD_large).
        :param n_POD: Number of POD modes to plot.
        :param save_dir: Directory where the plot will be saved.
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig = plt.figure(figsize=(15, 15))
        for i in range(n_POD):
            ax = fig.add_subplot(331 + i)
            plt.plot(self.t_lf, ulf_train[-1, :, i], label='LF')
            plt.plot(self.t_hf, uhf_train[-1, :, i], label='HF')
            ax.title.set_text(f'POM {i + 1}')
            plt.xlabel('t')
            plt.legend()

        # Save the plot instead of showing it
        plot_path = os.path.join(save_dir, "POD_coefficients_plot.png")
        plt.savefig(plot_path)
        plt.close(fig)  

    def comparison_POD_modes(self, n_POD: int, uMF_LSTM_test: np.ndarray, ulf_test: np.ndarray, uhf_test: np.ndarray, save_dir: str =  "plots_reaction_diffusion") -> None:
        """
        Compare POD modes for different models and save the plots to the specified directory.

        :param n_POD: Number of POD modes to compare.
        :param uMF_LSTM_test: Predictions from the MF-LSTM model, with dimensions (N_mu_test, Nt_test, n_POD_large).
        :param ulf_test: Low-fidelity test data, with dimensions (Nx_lf, Ny_lf, Nt_test, N_mu_test).
        :param uhf_test: High-fidelity test data, with dimensions (Nx_hf, Ny_hf, Nt_test, N_mu_test).
        :param save_dir: Directory where the plots will be saved.
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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

            # Save the plot for the current value of mu
            plot_path = os.path.join(save_dir, f"comparison_POD_modes_mu_{mu}.png")
            plt.savefig(plot_path)
            plt.close(fig)

    def comparison_solutions(self, times: List[float], uMF_LSTM_test_rec: np.ndarray, save_dir: str =  "plots_reaction_diffusion") -> None:
        """
        Compare solutions at specific times and save the plots to the specified directory.

        :param times: List of specific times to compare.
        :param uMF_LSTM_test_rec: Reconstructed predictions from the MF-LSTM model, with dimensions (Nx_hf, Ny_hf, Nt_test, N_mu_test).
        :param save_dir: Directory where the plots will be saved.
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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

                # Save the plot for the current mu and time
                plot_path = os.path.join(save_dir, f"comparison_solution_mu_{mu}_time_{t}.png")
                plt.savefig(plot_path)
                plt.close(fig)

    # Function to normalize input data between 0 and 1 based on min and max values.
    @staticmethod
    def normalization(x: tf.Tensor, xmax: tf.Tensor, xmin: tf.Tensor) -> tf.Tensor:
        """
        Normalizes input tensor `x` to a range between 0 and 1 using
        the provided minimum and maximum values.

        Parameters:
        - x (tf.Tensor): Input tensor to normalize.
        - xmax (tf.Tensor): Tensor containing the maximum values for normalization.
        - xmin (tf.Tensor): Tensor containing the minimum values for normalization.

        Returns:
        - tf.Tensor: Normalized tensor.
        """
        return (x - xmin) / (xmax - xmin)

    # Function to reverse normalization, scaling data back to its original range.
    @staticmethod
    def denormalization(x: tf.Tensor, xmax: tf.Tensor, xmin: tf.Tensor) -> tf.Tensor:
        """
        Reverts the normalization process, scaling the tensor `x` back to its original range
        defined by the provided minimum and maximum values.

        Parameters:
        - x (tf.Tensor): Normalized tensor to be denormalized.
        - xmax (tf.Tensor): Tensor containing the maximum values for denormalization.
        - xmin (tf.Tensor): Tensor containing the minimum values for denormalization.

        Returns:
        - tf.Tensor: Denormalized tensor.
        """
        return x * (xmax - xmin) + xmin


    @staticmethod
    def getModel(
        params: Dict[str, Union[int, float, str, bool]], 
        num_inputs: int, 
        name: str, 
        num_outputs: int
    ) -> Model:
        """
        Builds and compiles a Keras model based on the specified architecture.

        Parameters:
        - params (Dict): A dictionary containing various parameters for the model.
        - num_inputs (int): Number of input features.
        - name (str): Name of the model architecture to build.
        - num_outputs (int): Number of output features.

        Returns:
        - model (Model): A compiled Keras model.
        """

        inputs = Input(shape=(None, num_inputs) if 'LSTM' in name else (num_inputs,))

        if name == "LSTM":
            a = LSTM(params['nodes'], return_sequences=True)(inputs)
            for _ in range(params['lay'] - 1):
                a = Dropout(params['dropout'])(a)
                a = LSTM(params['nodes'], return_sequences=True)(a)
            for _ in range(params['lay_dense']):
                a = Dense(params['nodes_dense'])(a)
            output = Dense(num_outputs, activation='linear')(a)

        elif name == "LSTM_support":
            a = LSTM(params['nodes'], return_sequences=True)(inputs)
            for _ in range(params['lay'] - 1):
                a = Dropout(params['dropout'])(a)
                a = LSTM(params['nodes'], return_sequences=True)(a)
            a = Dense(64, activation=Activations.sinusoidal_activation, kernel_initializer='uniform')(a)
            for _ in range(params['lay_dense']):
                a = Dense(params['nodes_dense'], activation=Activations.custom_activation)(a)
            output = Dense(num_outputs, activation='linear')(a)


        # Compile and return the model for non-GP and non-Inter models
        model = Model(inputs=inputs, outputs=output)
        opti = Helpers_NN.getOpti(params['opt'], params['lr'])
        model.compile(loss='mse', optimizer=opti, metrics=['mse'])
        return model

























        