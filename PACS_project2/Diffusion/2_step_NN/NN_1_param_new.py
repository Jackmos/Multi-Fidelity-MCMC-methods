######### LIBRARIES ############
# RICORDA DI ELIMINARE TUTTE LIBRERIE CHE NON UTILIZZI
import keras.backend as K
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

file_path_LF = "../DATA/reaction_diffusion_LF_46_big.mat"
(reaction_LF, U_LF) = import_data(file_path_LF)
reaction_LF_test = reaction_LF
U_LF_test = U_LF
file_path_HF = "../DATA/reaction_diffusion_HF.mat"
(reaction_HF, U_HF) = import_data(file_path_HF)
reaction_HF_test = reaction_HF
U_HF_test = U_HF

# NORMALIZATION

permutation = np.random.permutation(len(reaction_LF))
Nlf = 20
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
     - 1,
    20,
    20,
]
U_LF = U_LF[permutation][0:Nlf]
U_LF_test = U_LF_test[
    :,
    - 1,
    20,
    20
]

permutation = np.random.permutation(len(reaction_HF))
n_HF = 10
reaction_HF = reaction_HF[permutation][0:n_HF]

U_HF = U_HF[:, -1, 44,44]
U_HF = U_HF[permutation][0:n_HF]
U_HF_test = U_HF_test[
    :, -1, 44,44
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
NepoHF = 3000  # number of epochs for second NN: NN_HF

U_LF_list = []
U_HF_list = []

HF_data = [10]
HF_data_str = [str(num) for num in HF_data]
r2_HF_df = pandas.DataFrame(index=HF_data_str)  # dataframe which stores HF R^2
r2_LF_df = pandas.DataFrame()  # dataframe which stores LF R^2
mse_HF_df = pandas.DataFrame(
    index=HF_data_str
)  # dataframe which stores HF MSE
mse_LF_df = pandas.DataFrame()  # dataframe which stores LF MSE
###
Nlf_models = [1]
###
for m in range(len(Nlf_models)):
    # loop over the basis
    print(
        f"********************  #basis functions = {Nlf_models[m]}  ********************"
    )

    test_mse_HF_list = []
    test_mse_LF_list = []

    r2_HF_list = []
    r2_LF_list = []

    #########################     TRAIN SET      ##########################
    U_train_LF = U_LF  # [:, m]

    #########################     TRAIN SET      ##########################
    U_test_LF = U_LF_test  # [:, m]

    ##########################       FIRST NN: NN_LF     ##########################
    K.clear_session()
    bestLF_params = {
        "lr": 0.0255,
        "kernel_init": "glorot_uniform",
        "opt": "Adam",
    }  # <------------------

    modelLF = getModel(bestLF_params, "LF")  # ann_functions
    histLF = modelLF.fit(
        reaction_LF, U_train_LF, epochs=NepoLF, batch_size=Nlf, verbose=0
    )
    #        histLF = modelLF.fit(Young_LF_norm[:,0], U_train_LF ,epochs=NepoLF,batch_size=Nlf, verbose = 0)
    # attenzione -> QUANDO LO FARAI IN PIU DIMENSIONI DI INPUT QUESTA PARTE E' DA SCRIVERE TENENDO CONTO DELLE DIMNSIONI EFFETTIVE DELLA MATRICE DI INPUT, DATO CHE NON POTRAI TAGLIARE ALL'INIZIO
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

        #########################     TRAIN SET      #########################
        # Young_train_HF = np.loadtxt(
        #    "../DATA_shear_cube/Data_Train_bases_young_new/mu_train_HF_young_"
        #    + n_HF_txt
        # )
        # U_hf_train = np.abs(
        #    np.loadtxt(
        #        "../DATA_shear_cube/Data_Train_bases_young_new/Uhf_train_young_"
        #        + n_HF_txt
        #    )
        # )

        #########################     TEST SET      ##########################
        # U_hf_test = np.abs(
        #    np.loadtxt(
        #        "../DATA_shear_cube/Data_Test_bases_young_new/Uhf_test_young.txt"
        #    )
        # )

        ##########################     NORMALIZATION  ##########################
        # Input
        # Young_train_HF_norm = (Young_train_HF - Young_min) / (
        #    Young_max - Young_min
        # )

        # U_t_min_test = np.min(U_hf_test, axis=0)
        # U_t_max_test = np.max(U_hf_test, axis=0)

        # U_t_min_train = np.min(U_hf_train, axis=0)
        # U_t_max_train = np.max(U_hf_train, axis=0)

        # U_hf_train = (U_hf_train - U_t_min_train) / (
        #    U_t_max_train - U_t_min_train
        # )
        # U_hf_test = (U_hf_test - U_t_min_test) / (U_t_max_test - U_t_min_test)

        ##########################    SECOND NN: NN_HF    ##########################
        reaction_test_help = modelLF.predict(reaction_HF_test)[:, 0]
        reaction_test_in = np.vstack(
            (reaction_HF_test[:, 0], reaction_test_help)
        ).transpose()  # <- TEST INPUT for the second NN: NN_HF

        reaction_train_help = modelLF.predict(reaction_HF)[
            :, 0
        ]  # f_LF(mu_hf_train)
        reaction_final = np.vstack(
            (reaction_HF[:, 0], reaction_train_help)
        ).transpose()  # <- TRAINING INPUT for the second NN: NN_HF

        name = "2step"
        K.clear_session()
        # best parameters obtained by HPO:

        best_params = {
            "kernel_init": "uniform",
            "l2weight": 0.0002588943618918075,
            "lr": 0.008365144388542751,
            "nodes": 6.0,
            "opt": "Adam",
        }
        
        
        finalModel = getModel(
            best_params, name
        )  # final model chosen according to the best paramters
        hist = finalModel.fit(
            reaction_final,
            U_HF,
            validation_data=(reaction_test_in, U_HF_test),
            epochs=NepoHF,
            batch_size=n_HF,
            verbose=0,
            validation_freq=20,
        )

        UHF = finalModel.predict(reaction_test_in)
        U_HF_list.append(UHF)

        stop = perf_counter()
        elapsed = stop - start
        print("Elapsed time: ", elapsed)
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

    mse_LF_df[str(Nlf_models[m])] = test_mse_LF_list
    mse_HF_df[str(Nlf_models[m])] = test_mse_HF_list

print(r2_HF_df.round(5))
print(mse_HF_df.round(5))

plt.figure()
plt.plot(
    reaction_test_in[:, 0], U_HF_test, "ro", linewidth=1.5, label="HF model"
)
plt.plot(
    reaction_final[:, 0],
    U_HF,
    "r*",
    markersize=5,
    label="HF training points",
)
plt.plot(
    reaction_LF,
    U_LF,
    "y*",
    markersize=5,
    label="LF training points",
)
plt.plot(
    reaction_test_in[:, 0],
    finalModel.predict(reaction_test_in),
    "g*",
    linewidth=3,
    label="Predicted HF model",
)
plt.legend(prop={"size": 8.3})
plt.show()

#########################     SAVE the OUTPUT      ##########################
os.makedirs("Output_MF_Nlf10_10")

r2_HF_df.to_csv(
    "./Output_MF_Nlf10_10/r2_HF_lhs.txt",
    header=True,
    index=False,
    sep="\t",
    mode="a",
)
mse_HF_df.to_csv(
    "./Output_MF_Nlf10_10/mse_HF_lhs.txt",
    header=True,
    index=False,
    sep="\t",
    mode="a",
)
r2_LF_df.to_csv(
    "./Output_MF_Nlf10_10/r2_LF_lhs.txt",
    header=True,
    index=False,
    sep="\t",
    mode="a",
)
mse_LF_df.to_csv(
    "./Output_MF_Nlf10_10/mse_LF_lhs.txt",
    header=True,
    index=False,
    sep="\t",
    mode="a",
)

with open("./Output_MF_Nlf10_10/U_HF_list.data", "wb") as filehandle:
    # store the data as binary data stream
    pickle.dump(U_HF_list, filehandle)

with open("./Output_MF_Nlf10_10/U_LF_list.data", "wb") as filehandle:
    pickle.dump(U_LF_list, filehandle)
