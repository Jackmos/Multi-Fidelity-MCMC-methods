# # High Fidelity Neural Network
# Neural Network which uses HF dataset and exploits the High fidelity component of the MF model  

#########################     LIBRARIES     ##########################
import keras.backend as K
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf
from source.Diffusion_helper import *
from utils.functions_to_ray import *


from utils.network_utils import *

from pathlib import Path

# path to the current notebook
current_file_path = Path().resolve()
# path to the current folder

# reproducibility
def set_seed():
        
    seed = 42
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)

def main_function():
    set_seed()
    # # Data preparation

    file_path_HF = "test/DATA_reaction_diffusion_test_case/reaction_diffusion_HF.mat"
    (reaction_HF_test, U_HF_test) = Diffusion_model_helpers.import_data(file_path_HF)

    U_HF_test = U_HF_test[:, -1, 44,44]

    reaction_HF_test = Helpers_NN.normalization(reaction_HF_test)
    U_HF_test = Helpers_NN.normalization(U_HF_test)


    reaction_HF_test_original=np.c_[reaction_HF_test, np.abs(np.sin(5*np.pi*reaction_HF_test[:, 0]-5*np.pi/6))]
    U_HF_test_original=U_HF_test


    # data augmentation
    noise_stddev1=[0.005,0.003, 0.01, 0.003]
    noise_stddev2=[0.01,0.005,0.02,0.005]
    (U_HF_test,reaction_HF_test)=Helpers_NN.add_noise(noise_stddev1,noise_stddev2,reaction_HF_test,U_HF_test.reshape(-1,1))
    reaction_HF_test=np.c_[reaction_HF_test, np.abs(np.sin(5*np.pi*reaction_HF_test[:, 0]-5*np.pi/6))]


    permutation = np.random.permutation(len(reaction_HF_test))
    # number of epochs for second NN: NN_HF
    NepoHF = 8000  
    Nhf = 100

    reaction_HF = reaction_HF_test[permutation,:][0:Nhf,:]
    U_HF = U_HF_test[permutation][0:Nhf]

    # # Neural Network


    K.clear_session()
    best_params = {
        'kernel_init': 'glorot_uniform',
        'l2weight': 0.03711750915883216,
        'lr': 0.024244197598433916,
        'opt': 'Adamax'
    }


    ####################    NN training and PREDICTION    #######################

    print("\nHF Model:")

    definition_HF={
            "network_type": "LF",   # I leave the same NN,  even if we use HF data instead of LF
            "network_parameters": best_params,
            "dataset_train": reaction_HF,
            "output_train": U_HF,
            "epochs_number": NepoHF,
            "batch_size": Nhf,
            "train": True,
            "do_HPO": False,
            "verbose": False
        }

    model= NetworkFactory.build_network(NetworkConfig(**definition_HF),Diffusion_model_helpers.getModel)
    UHF = model.prediction(reaction_HF_test_original)

    (mse_HF,R_HF) = model.performance(reaction_HF_test_original,U_HF_test_original)


    Diffusion_model_helpers.plot_results(reaction_HF_test_original,U_HF_test_original, model.prediction(reaction_HF_test_original), "HF",  reaction_HF,  U_HF, "#FF7F0E", "#2CA02C", "HF model", "Predicted HF model")




if __name__ == "__main__":
    main_function()