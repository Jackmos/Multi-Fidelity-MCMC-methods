#########################     LIBRARIES     ##########################
import keras.backend as K

# import warnings
from keras.regularizers import l2
from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from keras.optimizers import Adam, Nadam, Adamax
from ann_functions import getModel, kCrossVal, transfBestparam, import_data
from time import perf_counter
import pandas
import pickle
import os

seed = 7
np.random.seed(seed)

file_path_LF = "../DATA/reaction_diffusion_LF.mat"
(reaction_LF, U_LF) = import_data(file_path_LF)
reaction_LF_test = reaction_LF
U_LF_test = U_LF
file_path_HF = "../DATA/reaction_diffusion_HF.mat"
(reaction_HF, U_HF) = import_data(file_path_HF)
reaction_HF_test = reaction_HF
U_HF_test = U_HF

# NORMALIZATION

permutation = np.random.permutation(len(reaction_LF))
Nlf = 10
reaction_LF = reaction_LF[permutation][0:Nlf]

reaction_max = np.max(reaction_LF_test)
reaction_min = np.min(reaction_LF_test)

# reaction_test_norm = (reaction_test - reaction_min) / (reaction_max - reaction_min)
reaction_LF = (reaction_LF - reaction_min) / (reaction_max - reaction_min)
reaction_HF = (reaction_HF - reaction_min) / (reaction_max - reaction_min)
reaction_LF_test = (reaction_LF_test - reaction_min) / (
    reaction_max - reaction_min
)
reaction_HF_test = (reaction_HF_test - reaction_min) / (
    reaction_max - reaction_min
)

# Output
# We consider the central value of U
U_LF = U_LF[
    :,
    int(np.shape(U_LF)[1] / 2) - 1,
    int(np.shape(U_LF)[2] / 2),
    int(np.shape(U_LF)[3] / 2),
]
U_LF = U_LF[permutation][0:Nlf]
U_LF_test = U_LF_test[
    :,
    int(np.shape(U_LF_test)[1] / 2) - 1,
    int(np.shape(U_LF_test)[2] / 2),
    int(np.shape(U_LF_test)[3] / 2),
]

permutation = np.random.permutation(len(reaction_HF))
n_HF = 10
reaction_HF = reaction_HF[permutation][0:n_HF]

U_HF = U_HF[:, -1, int(np.shape(U_HF)[2] / 2), int(np.shape(U_HF)[3] / 2)]
U_HF = U_HF[permutation][0:n_HF]
U_HF_test = U_HF_test[
    :, -1, int(np.shape(U_HF_test)[2] / 2), int(np.shape(U_HF_test)[3] / 2)
]

U_h_max_test = np.max(U_HF_test)
U_h_min_test = np.min(U_HF_test)
U_t_max_test = np.max(U_LF_test)
U_t_min_test = np.min(U_LF_test)

# U_LF = (U_LF - U_t_min_train) / (U_t_max_train - U_t_min_train)
U_LF = (U_LF - U_t_min_test) / (U_t_max_test - U_t_min_test)
U_LF_test = (U_LF_test - U_t_min_test) / (U_t_max_test - U_t_min_test)
U_HF = (U_HF - U_h_min_test) / (U_h_max_test - U_h_min_test)
U_HF_test = (U_HF_test - U_h_min_test) / (U_h_max_test - U_h_min_test)


NepoLF = 3000  # number of epochs for first NN: NN_LF
NepoLin = 1500
NepoHF = 3000  # number of epochs for second NN: NN_HF

U_LF_list = []
U_Lin_list = []
U_HF_list = []
# <------------------------------here
HF_data = [10]
HF_data_str = [str(num) for num in HF_data]
r2_HF_df = pandas.DataFrame(index=HF_data_str)  # dataframe which stores HF R^2
r2_LF_df = pandas.DataFrame()  # dataframe which stores LF R^2
r2_Lin_df = pandas.DataFrame()
mse_HF_df = pandas.DataFrame(
    index=HF_data_str
)  # dataframe which stores HF MSE
mse_LF_df = pandas.DataFrame()  # dataframe which stores LF MSE
mse_Lin_df = pandas.DataFrame()
###
Nlf_models = [1]
###

for m in range(len(Nlf_models)):
    # Loop over the range of the number of basis functions

    print(
        f"********************  #basis functions = {Nlf_models[m]}  ********************"
    )
    test_mse_HF_list = []
    test_mse_LF_list = []
    test_mse_Lin_list = []

    r2_HF_list = []
    r2_LF_list = []
    r2_Lin_list = []

    #########################     TRAIN SET      ##########################
    U_train_LF = U_LF

    #########################     TEST SET      ##########################
    U_test_LF = U_LF_test  # [:, m]

    ##########################       FIRST NN: NN_LF     ##########################
    K.clear_session()
    bestLF_params = {
        "lr": 0.0255,
        "kernel_init": "glorot_uniform",
        "opt": "Adam",
    }

    modelLF = getModel(bestLF_params, "LF")

    histLF = modelLF.fit(
        reaction_LF, U_train_LF, epochs=NepoLF, batch_size=Nlf, verbose=0
    )
    print("LF NN done")

    ULF = modelLF.predict(reaction_LF_test)
    U_LF_list.append(ULF)
    print("\nLF Model:")

    test_mse = np.mean(np.square(U_test_LF - ULF[:, 0]))
    test_mse_LF_list.append(test_mse)
    print(f"Test MSE: {test_mse}")

    r_2 = 1 - np.sum(np.square(U_test_LF - ULF[:, 0])) / np.sum(
        np.square(U_test_LF - np.mean(U_test_LF))
    )
    r2_LF_list.append(r_2)
    print(f"R^2: {r_2}")

    for n_HF in HF_data:
        # loop over the possible numbers of high-fidelity data

        print(f"\n-------  #HF data = {n_HF}  -------")
        start = perf_counter()

        n_HF_txt = str(n_HF) + ".txt"

        ##########################    SECOND NN: NN_Lin    ##########################
        reaction_test_help = modelLF.predict(reaction_HF_test)[:, 0]
        reaction_test_in = np.vstack(
            (reaction_HF_test[:, 0], reaction_test_help)
        ).transpose()  # <- TEST INPUT for the second NN: NN_Lin

        reaction_train_help = modelLF.predict(reaction_HF)[
            :, 0
        ]  # f_LF(mu_hf_train)
        reaction_lin = np.vstack(
            (reaction_HF[:, 0], reaction_train_help)
        ).transpose()  # <- TRAINING INPUT for the second NN: NN_Lin

        bestLin_params = {
            "lr": 0.001,
            "kernel_init": "glorot_uniform",
            "opt": "Adam",
            "l2weight": 0.01,
        }
        modelLin = getModel(bestLin_params, "Hflin")
        histLin = modelLin.fit(
            reaction_lin,
            U_HF,
            validation_data=(reaction_test_in, U_HF_test),
            epochs=NepoLin,
            batch_size=n_HF,
            verbose=0,
            validation_freq=50,
        )

        ULin = modelLin.predict(
            np.vstack((reaction_HF_test[:, 0], reaction_test_help)).transpose()
        )

        ##########################    THIRD NN: NN_HF    ##########################
        # Input for training NN_HF
        reaction_help1 = modelLin.predict(reaction_lin)[:, 0]
        reaction_final = np.vstack(
            (reaction_HF[:, 0], reaction_train_help, reaction_help1)
        ).transpose()

        # Input for testing NN_HF
        reaction_test_help1 = modelLin.predict(reaction_test_in)[:, 0]
        reaction_test_final = np.vstack(
            (reaction_HF_test[:, 0], reaction_test_help, reaction_test_help1)
        ).transpose()

        name = "3step"
        MAX_EVAL = 15

        K.clear_session()
        # best paramters obtained by HPO:
        best_params = {
            "kernel_init": "glorot_uniform",
            "l2weight": 0.0001738393292451542,
            "lr": 0.0003670853655509693,
            "nodes": 48.0,
            "opt": "Adam",
        }
        # best_params = {'kernel_init': 'glorot_uniform', 'l2weight': 0.00015338174139229794, 'lr': 0.010752960934369659, 'nodes': 28.0, 'opt': 'Adam'}
        finalModel = getModel(
            best_params, name
        )  # final model chosen according to the best paramters
        hist = finalModel.fit(
            reaction_final,
            U_HF,
            validation_data=(reaction_test_final, U_HF_test),
            epochs=NepoHF,
            batch_size=n_HF,
            validation_freq=50,
            verbose=0,
        )

        UHF = finalModel.predict(reaction_test_final)
        U_HF_list.append(UHF)

        stop = perf_counter()
        elapsed = stop - start
        print("Elapsed time: ", elapsed)

        ULin = modelLin.predict(
            np.vstack((reaction_HF_test[:, 0], reaction_test_help)).transpose()
        )
        U_Lin_list.append(ULin)
        print("\nLin Model:")

        test_mse = np.mean(np.square(U_HF_test - ULin[:, 0]))
        test_mse_Lin_list.append(test_mse)
        print(f"Test MSE: {test_mse}")

        r2_Lin = 1 - np.sum(np.square(U_HF_test - ULin[:, 0])) / np.sum(
            np.square(U_HF_test - np.mean(U_HF_test))
        )
        r2_Lin_list.append(r2_Lin)
        print(f"R^2: {r2_Lin}")

        print("\nHF Model:")
        test_mse = np.mean(np.square(U_HF_test - UHF[:, 0]))
        test_mse_HF_list.append(test_mse)
        print(f"Test MSE: {test_mse}")
        r2_HF = 1 - np.sum(np.square(U_HF_test - UHF[:, 0])) / np.sum(
            np.square(U_HF_test - np.mean(U_HF_test))
        )
        r2_HF_list.append(r2_HF)
        print(f"R^2: {r2_HF}")

    r2_LF_df[str(Nlf_models[m])] = r2_LF_list
    r2_HF_df[str(Nlf_models[m])] = r2_HF_list
    r2_Lin_df[str(Nlf_models[m])] = r2_Lin_list
    mse_LF_df[str(Nlf_models[m])] = test_mse_LF_list
    mse_HF_df[str(Nlf_models[m])] = test_mse_HF_list
    mse_Lin_df[str(Nlf_models[m])] = test_mse_Lin_list

print(r2_HF_df.round(5))
print(mse_HF_df.round(5))


#########################     SAVE the OUTPUT      ##########################
os.makedirs("Output_4")

r2_HF_df.to_csv(
    "./Output_4/r2_HF_lhs.txt", header=True, index=False, sep="\t", mode="a"
)
mse_HF_df.to_csv(
    "./Output_4/mse_HF_lhs.txt", header=True, index=False, sep="\t", mode="a"
)
r2_LF_df.to_csv(
    "./Output_4/r2_LF_lhs.txt", header=True, index=False, sep="\t", mode="a"
)
mse_LF_df.to_csv(
    "./Output_4/mse_LF_lhs.txt", header=True, index=False, sep="\t", mode="a"
)
r2_Lin_df.to_csv(
    "./Output_4/r2_Lin_lhs.txt", header=True, index=False, sep="\t", mode="a"
)
mse_Lin_df.to_csv(
    "./Output_4/mse_Lin_lhs.txt", header=True, index=False, sep="\t", mode="a"
)

with open("estimated_UHF.txt", "w") as file:
    for item in UHF[:, 0]:
        file.write(f"{UHF}\n")
