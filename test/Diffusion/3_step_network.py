# # EXAMPLE I Three Steps MultiFidelity Neural Network

# Code to show the performance of the Neural Network when dealing with different LF datasets 

######### LIBRARIES ############
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt

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
    set_seed()
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
    U_HF_test= Helpers_NN.normalization(U_HF_test)

    # transformation of the dataseet
    reaction_HF_test_original=np.c_[reaction_HF_test, np.abs(np.sin(5*np.pi*reaction_HF_test[:, 0]-5*np.pi/6))]
    U_HF_test_original=U_HF_test

    # data augmentation introducing a noise
    noise_stddev1=[0.01,0.005,0.02,0.005]
    noise_stddev2=[0.005,0.003,0.01,0.003]

    (U_HF_test,reaction_HF_test)=Helpers_NN.add_noise(noise_stddev1,noise_stddev2,reaction_HF_test,U_HF_test.reshape(-1,1))
    reaction_HF_test=np.c_[reaction_HF_test, np.abs(np.sin(5*np.pi*reaction_HF_test[:, 0]-5*np.pi/6))]

    # plot of the new dataset
    plt.plot(reaction_HF_test[:,0], U_HF_test, "ro", linewidth=1, label="HF model with noise")
    plt.plot(reaction_HF_test_original[:,0], U_HF_test_original, "r-", linewidth=1, label="HF model")
    plt.show()

    # Number of data of High and Low Fidelity components of the dataset
    n_HF = np.array([100])
    Nlf = np.array([200])
    n_per=np.array([150])
    batch_size=[200,150, 100]



    # number of epochs 
    Nepo=[7000,1500,8000]

    # NepoLF = 7000  # number of epochs for first NN: NN_LF
    # NepoHF = 8000  # number of epochs for second NN: NN_HF

    r2_HF_df = pd.DataFrame(columns=['Discretization','Diffusion','R2'])  # dataframe which stores HF R^2
    r2_LF_df = pd.DataFrame(columns=['Discretization','Diffusion','R2'])  # dataframe which stores LF R^2
    r2_Lin_df = pd.DataFrame(columns=['Discretization','Diffusion','R2'])
    mse_HF_df = pd.DataFrame(columns=['Discretization','Diffusion','MSE'])  # dataframe which stores HF MSE
    mse_LF_df = pd.DataFrame(columns=['Discretization','Diffusion','MSE'])  # dataframe which stores LF MSE





    # Define parameters
    folder_name = "3_step_models"
    folder_path = Diffusion_model_helpers.create_folder(folder_name)

    # Iterate through discretizations, diffusion values, and model parameters
    for m in range(len(Discretizations)):
        test_mse_HF_list = []
        test_mse_LF_list = []
        test_mse_per_list = []

        r2_HF_list = []
        r2_LF_list = []
        r2_per_list = []

        for d in range(len(diffusion)):
            for nhf in n_HF:
                for nlf in Nlf:
                    for nper in n_per:

                        # Logging the current parameters
                        print(f"********************  # Nb. nodes = {Discretizations[m]}  ********************")
                        print(f"********************  # Diff: = {diffusion[d]}  ********************")
                        print(f"********************  # NHF: = {nhf}  ********************")
                        print(f"********************  # Nper: = {nper}  ********************")
                        print(f"********************  # NLF: = {nlf}  ********************")




                        # Import Low Fidelity data
                        file_path_LF = f"test/DATA_reaction_diffusion_test_case/reaction_diffusion_LF_{Discretizations[m]}_d{diffusion[d]}.mat"
                        reaction_LF_test, U_LF_test = Diffusion_model_helpers.import_data(file_path_LF)

                        # Normalize and select specific data points
                        U_LF_test = U_LF_test[:, -1, int(4*(Discretizations[m]-1)/9), int(4*(Discretizations[m]-1)/9)]
                        U_LF_test = Helpers_NN.normalization(U_LF_test)
                        reaction_LF_test = Helpers_NN.normalization(reaction_LF_test)
                        
                        # Transform Low Fidelity dataset
                        reaction_LF_test_original = np.c_[reaction_LF_test, np.abs(np.sin(5 * np.pi * reaction_LF_test[:, 0] - 5 * np.pi / 6))]
                        U_LF_test_original = U_LF_test  # Keep original for later use
                        
                        # Data augmentation with noise
                        noise_std1 = [0.02, 0.01, 0.02, 0.01]
                        noise_std2 = [0.01, 0.005, 0.01, 0.005]
                        U_LF_test, reaction_LF_test = Helpers_NN.add_noise(noise_std1, noise_std2, reaction_LF_test, U_LF_test.reshape(-1,1))
                        
                        # Prepare datasets for training and validation
                        reaction_LF, U_train_LF = Diffusion_model_helpers.select_random_data(reaction_LF_test, U_LF_test, nlf)
                        reaction_per, U_train_per = Diffusion_model_helpers.select_random_data(reaction_HF_test, U_HF_test, nper)
                        reaction_HF, U_HF = Diffusion_model_helpers.select_random_data(reaction_HF_test, U_HF_test, nhf)

                        # Transform datasets for training
                        reaction_LF =Diffusion_model_helpers. augment_with_sin(reaction_LF)
                        reaction_LF_test =Diffusion_model_helpers. augment_with_sin(reaction_LF_test)

                        # Prepare validation datasets
                        reaction_HF_val, U_HF_val = Diffusion_model_helpers.select_random_data(reaction_HF_test, U_HF_test, nhf)
                        reaction_per_val, U_per_val = Diffusion_model_helpers.select_random_data(reaction_HF_test, U_HF_test, nper)
                        reaction_LF_val, U_train_LF_val = Diffusion_model_helpers.select_random_data(reaction_LF_test, U_LF_test, nlf)

                        # Transform validation dataset
                        reaction_LF_val = Diffusion_model_helpers.augment_with_sin(reaction_LF_val)

                        ########################## Neural Network ##########################
                        K.clear_session()

                        bestLF_params = {'kernel_init': 'glorot_uniform',
                                        'l2weight': 0.055687407222784514,
                                        'lr': 0.013763036873234191,
                                        'nodes': 15,
                                        'opt': 'Adam'}
                        bestper_params = {'kernel_init': 'uniform',
                                            'l2weight': 0.00019723116196368882,
                                            'lr': 0.08155671235896393,
                                            'nodes': 7,
                                            'opt': 'Adamax'}
                        best_params = {'kernel_init': 'uniform',
                                        'l2weight': 0.00014981392074368653,
                                        'lr': 0.020280502294291045,
                                        'nodes': 14,
                                        'opt': 'Adamax'}

                        names = ["LF", "Hfper", "HF"]
                        params = [bestLF_params, bestper_params, best_params]
                        data_train = [reaction_LF, reaction_per, reaction_HF]
                        output_train = [U_train_LF, U_train_per, U_HF]

                        definition_3steps = {
                            "network_type": "3step",
                            "names": names,
                            "network_parameters": params,
                            "dataset_train": data_train,
                            "output_train": output_train,
                            "dataset_validation": [reaction_LF_val, reaction_per_val, reaction_HF_val],
                            "output_validation": [U_train_LF_val, U_per_val, U_HF_val],
                            "epochs_number": Nepo,
                            "batch_size": batch_size,
                            "train": True,
                            "do_HPO": False,
                            "verbose": False
                        }
                        model = NetworkFactory.build_network(NetworkConfig(**definition_3steps),Diffusion_model_helpers.getModel )

                        ### Network 1 (Low Fidelity Model)
                        ULF = model.model_list[0].prediction(reaction_LF_test_original)
                        print("Low Fidelity NN")

                        test_mse_LF, r2_LF = model.model_list[0].performance(reaction_LF_test_original, U_LF_test_original)
                        test_mse_LF_list.append(test_mse_LF)
                        r2_LF_list.append(r2_LF)

                        Diffusion_model_helpers.plot_results(reaction_LF_test_original, U_LF_test_original, ULF, "LF",reaction_LF,  U_train_LF,"#1F77B4", "#2CA02C", "LF model", "Predicted LF model",filename=f"First_Net_{diffusion[d]}_{Discretizations[m]}")

                        ### Network 2 (Intermediate Model)
                        input_per = np.concatenate((reaction_HF_test_original,  model.model_list[0].prediction(reaction_HF_test_original).reshape(-1,1)),axis=1 )
                        input_per_train = np.concatenate((reaction_HF,  model.model_list[0].prediction(reaction_HF).reshape(-1,1)),axis=1) 
                        print("Second model")
                        test_mse_per, r2_per = model.model_list[1].performance(input_per, U_HF_test_original)
                        test_mse_per_list.append(test_mse_per)
                        r2_per_list.append(r2_per)

                        Diffusion_model_helpers.plot_results(reaction_HF_test_original, U_HF_test_original, model.model_list[1].prediction(input_per),  "HF",  reaction_per,U_train_per, "#9467BD", "#2CA02C", "Periodic model", "Predicted Per model",filename=f"Second_Net_{diffusion[d]}_{Discretizations[m]}")

                        ### Network 3 (High Fidelity Model)
                        input_HF=np.concatenate((reaction_HF_test_original,  model.model_list[0].prediction(reaction_HF_test_original).reshape(-1,1)),axis=1) 

                        input_HF=np.concatenate((input_HF, model.model_list[1].prediction(input_HF).reshape(-1,1)),axis=1)


                        input_HF_train = np.concatenate((reaction_HF,  model.model_list[0].prediction(reaction_HF).reshape(-1,1)),axis=1) 

                        input_HF_train=np.concatenate((input_HF_train, model.model_list[1].prediction(input_HF_train).reshape(-1,1)),axis=1)

                        print("Third model")
                        test_mse_HF, r2_HF = model.performance(reaction_HF_test_original, U_HF_test_original)
                        test_mse_HF_list.append(test_mse_HF)
                        r2_HF_list.append(r2_HF)

                        Diffusion_model_helpers.plot_results(reaction_HF_test_original, U_HF_test_original, model.model_list[2].prediction(input_HF),"HF",reaction_HF,  U_HF, "#FF7F0E", "#2CA02C", "HF model", "Predicted HF model", filename=f"Third_Net_{diffusion[d]}_{Discretizations[m]}")

                        # Update DataFrames with results
                        r2_LF_df = Diffusion_model_helpers.update_results(r2_LF_df, Discretizations[m], diffusion[d], r2_LF_list[-1], "R2")
                        r2_HF_df = Diffusion_model_helpers.update_results(r2_HF_df, Discretizations[m], diffusion[d], r2_HF_list[-1], "R2")
                        mse_LF_df = Diffusion_model_helpers.update_results(mse_LF_df, Discretizations[m], diffusion[d], test_mse_LF_list[-1], "MSE")
                        mse_HF_df = Diffusion_model_helpers.update_results(mse_HF_df, Discretizations[m], diffusion[d], test_mse_HF_list[-1], "MSE")

                        # Save the model if it improves on the previous best R2 score
                        if not r2_HF_list[:-1] or (r2_HF_list[:-1] and r2_HF > max(r2_HF_list[:-1])):
                            # Create a list of file paths for each model in model_list
                            file_paths = [os.path.join(folder_path, f"{str(Discretizations[m])}_model_{i}.keras") for i in range(len(model.model_list))]
                            model.save(file_paths)


    # Print the results
    print(r2_HF_df.round(5))
    print(mse_HF_df.round(5))

 



if __name__ == "__main__":
    main_function()