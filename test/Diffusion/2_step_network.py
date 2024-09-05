# # Two Steps MultiFidelity Neural Network

# Code to show the performance of the Neural Network when dealing with different LF datasets 

import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import os
import keras
import tensorflow as tf

from utils.network_utils import *
from utils.functions_to_ray import *

from source.Diffusion_helper import *

# reproducibility
def set_seed():
        
    seed = 42
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)


def main_function():

    # ## Data Preparation


    # import of the discretization and diffusion parameter values
    Discretizations= np.loadtxt(
        "test/DATA_reaction_diffusion_test_case/Discretizations.txt"
    ).astype(
        int
    )[::-1]

    diffusion= np.loadtxt(
        "test/DATA_reaction_diffusion_test_case/diffusion.txt"
    ).astype(
        int
    )

    # introduction of the high fidelity part of the dataset
    file_path_HF = "test/DATA_reaction_diffusion_test_case/reaction_diffusion_HF.mat"
    (reaction_HF_test, U_HF_test) = Diffusion_model_helpers.import_data(file_path_HF)

    # I consider only the reaction term as a variable
    U_HF_test = U_HF_test[:, -1, 44,44]
    # normalization of the data
    reaction_HF_test = Helpers_NN.normalization(reaction_HF_test)
    U_HF_test=Helpers_NN.normalization(U_HF_test)

    # transformation of the dataseet
    reaction_HF_test_original=np.c_[reaction_HF_test, np.abs(np.sin(5*np.pi*reaction_HF_test[:, 0]-5*np.pi/6))]
    U_HF_test_original=U_HF_test

    #
    U_HF_test=U_HF_test.reshape(-1,1)

    
    # data augmentation introducing a noise
    noise_stddev1=[0.01,0.005,0.02,0.005]
    noise_stddev2=[0.005,0.003,0.01,0.003]

    (U_HF_test,reaction_HF_test)=Helpers_NN.add_noise(noise_stddev1,noise_stddev2,reaction_HF_test,U_HF_test)
    reaction_HF_test=np.c_[reaction_HF_test, np.abs(np.sin(5*np.pi*reaction_HF_test[:, 0]-5*np.pi/6))]

    
    # plot of the new dataset
    plt.plot(reaction_HF_test[:,0], U_HF_test, "ro", linewidth=1, label="HF model with noise")
    plt.plot(reaction_HF_test_original[:,0], U_HF_test_original, "r-", linewidth=1, label="HF model")
    plt.show()

    
    # Number of data of High and Low Fidelity components of the dataset
    n_HF = np.array([100])
    Nlf = np.array([200])

    batch_size=[200,100]


    # number of epochs 
    Nepo=[7000,8000]



    r2_HF_df = pd.DataFrame(columns=['Discretization','diffusion','R2'])  # dataframe which stores HF R^2
    r2_LF_df = pd.DataFrame(columns=['Discretization','diffusion','R2'])  # dataframe which stores LF R^2
    mse_HF_df = pd.DataFrame(columns=['Discretization','diffusion','MSE'])  # dataframe which stores HF MSE
    mse_LF_df = pd.DataFrame(columns=['Discretization','diffusion','MSE'])  # dataframe which stores LF MSE



    # ### Cycle on the Discretization and the value of the diffusion 

    # Reproducibility settings
    set_seed()


    # Create the folder for saving models
    folder_name = "2_step_models"
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Folder '{folder_name}' created." if not os.path.exists(folder_path) else f"Folder '{folder_name}' already exists.")


    # Main loop
    for m, discretization in enumerate(Discretizations):
        test_mse_HF_list, test_mse_LF_list = [], []
        r2_HF_list, r2_LF_list = [], []

        for d, diff in enumerate(diffusion):
            for nhf in n_HF:
                for nlf in Nlf:

                    print(f"********************  # Nb. nodes = {discretization}  ********************")
                    print(f"********************  # Diff: = {diff}  ********************")
                    print(f"********************  # NHF: = {nhf}  ********************")
                    print(f"********************  # NLF: = {nlf}  ********************")

                    # Import and prepare data
                    file_path_LF = f"test/DATA_reaction_diffusion_test_case/reaction_diffusion_LF_{discretization}_d{diff}.mat"
                    reaction_LF_test, U_LF_test = Diffusion_model_helpers.import_data(file_path_LF)

                    # Normalize data
                    U_LF_test = U_LF_test[:, -1, int(4 * (discretization - 1) / 9), int(4 * (discretization - 1) / 9)]
                    U_LF_test = Helpers_NN.normalization(U_LF_test)
                    reaction_LF_test = Helpers_NN.normalization(reaction_LF_test)

                    # Data transformation and augmentation
                    reaction_LF_test_original = Diffusion_model_helpers.augment_with_sin(reaction_LF_test)
                    U_LF_test_original = U_LF_test  # Keep original for later evaluation
                    noise_std1, noise_std2 = [0.02, 0.01, 0.02, 0.01], [0.01, 0.005, 0.01, 0.005]
                    U_LF_test, reaction_LF_test = Helpers_NN.add_noise(noise_std1, noise_std2, reaction_LF_test, U_LF_test.reshape(-1, 1))

                    # Randomization and selection of training set
                    reaction_HF, U_HF = Diffusion_model_helpers.select_random_data(reaction_HF_test, U_HF_test, nhf)
                    reaction_LF, U_train_LF = Diffusion_model_helpers.select_random_data(reaction_LF_test, U_LF_test, nlf)

                    # Further transformation
                    reaction_LF = Diffusion_model_helpers.augment_with_sin(reaction_LF)
                    reaction_LF_test = Diffusion_model_helpers.augment_with_sin(reaction_LF_test)

                    ########################## Neural Network ##########################
                    K.clear_session()

                    # Model parameters
                    bestLF_params = {
                        'kernel_init': 'glorot_uniform',
                        'l2weight': 0.03711750915883216,
                        'lr': 0.024244197598433916,
                        'opt': 'Adamax'
                    }

                    best_params = {
                        'kernel_init': 'glorot_uniform',
                        'l2weight': 0.025443311601261794,
                        'lr': 0.0033275229144994574,
                        'nodes': 17,
                        'opt': 'Adamax'
                    }

                    # Build and train the model
                    definition_2steps = {
                        "network_type": "2step",
                        "names": ["LF", "HF"],
                        "network_parameters": [bestLF_params, best_params],
                        "dataset_train": [reaction_LF, reaction_HF],
                        "dataset_validation": [reaction_LF, reaction_HF],
                        "output_validation": [U_train_LF, U_HF],
                        "output_train": [U_train_LF, U_HF],
                        "epochs_number": Nepo,
                        "batch_size": batch_size,
                        "train": True,
                        "do_HPO": False,
                        "verbose": False
                    }
                    model = NetworkFactory.build_network(NetworkConfig(**definition_2steps),Diffusion_model_helpers.getModel)

                    # Low fidelity predictions and evaluation
                    ULF = model.model_list[0].prediction(reaction_LF_test_original)
                    print("Low fidelity NN")
                    test_mse_LF, r2_LF = model.model_list[0].performance(reaction_LF_test_original, U_LF_test_original)
                    test_mse_LF_list.append(test_mse_LF)
                    r2_LF_list.append(r2_LF)

                    # High fidelity input preparation and evaluation
                    input_HF = np.concatenate((reaction_HF_test_original, model.model_list[0].prediction(reaction_HF_test_original).reshape(-1, 1)), axis=1)
                    input_HF_train = np.concatenate((reaction_HF, model.model_list[0].prediction(reaction_HF).reshape(-1, 1)), axis=1)
                    print("Final model")
                    test_mse_HF, r2_HF = model.model_list[1].performance(input_HF, U_HF_test_original)
                    test_mse_HF_list.append(test_mse_HF)
                    r2_HF_list.append(r2_HF)

                    # Plotting results
                    Diffusion_model_helpers.plot_results(reaction_LF_test_original,U_LF_test_original, ULF, "LF",  reaction_LF, U_train_LF, "#1F77B4", "#2CA02C", "LF model", "Predicted LF model",filename=f"First_Net_{diffusion[d]}_{Discretizations[m]}_2steps")
                    Diffusion_model_helpers.plot_results(reaction_HF_test_original,U_HF_test_original, model.model_list[1].prediction(input_HF), "HF", reaction_HF,  U_HF,"#FF7F0E", "#2CA02C", "HF model", "Predicted HF model", filename=f"Sec_Net_{diffusion[d]}_{Discretizations[m]}_2steps")

                    # Save the model if it improves on the previous best R2 score
                    if not r2_HF_list[:-1] or (r2_HF_list[:-1] and r2_HF > max(r2_HF_list[:-1])):
                        # Create a list of file paths for each model in model_list
                        file_paths = [os.path.join(folder_path, f"{str(discretization)}_model_{i}.keras") for i in range(len(model.model_list))]
                        model.save(file_paths)

    # Assuming r2_HF_df and mse_HF_df are defined somewhere in your code
    print(r2_HF_df.round(5))
    print(mse_HF_df.round(5))

 
if __name__ == "__main__":
    main_function()