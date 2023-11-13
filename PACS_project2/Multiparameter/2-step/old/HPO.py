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
from ann_functions3D import getModel, kCrossVal, transfBestparam
from time import perf_counter
import pandas
import pickle
import os

seed = 7
np.random.seed(seed)

########################     PREPARATION      ##########################
# introduction of the data
# {'kernel_init': 'glorot_uniform', 'l2weight': 0.0005163753759788365, 'lr': 0.013859376098036235, 'nodes': 16.0, 'opt': 'Adamax'}
HF_data = np.loadtxt(
    "../DATA_shear_cube/3p/Data_Train_bases_new/HF_num_data.txt"
).astype(
    int
)  # list of number of HF data
HF_data_str = [str(num) for num in HF_data]
Nlf_models = np.loadtxt(
    "../DATA_shear_cube/3p/Data_Train_bases_new/N_bases.txt"
).astype(
    int
)  # list of number of bases
U_train_LF_full = np.abs(
    np.loadtxt(
        "../DATA_shear_cube/3p/Data_Train_bases_new/Ulf_train.txt"
    )
)

n_bases = U_train_LF_full.shape[1]
# m = 0
m = list(Nlf_models).index(n_bases)
n_HF = HF_data_str[-1]  # <---- uncomment
# n_HF = 5
n_HF_txt = str(n_HF) + ".txt"

#########################     TRAIN SET      ##########################
mu_train_LF = np.loadtxt(
    "../DATA_shear_cube/3p/Data_Train_bases_new/mu_train_LF.txt"
)
Nlf = np.size(mu_train_LF)  # number of low-fidelity data to TRAIN the NN

permutation = np.random.permutation(len(mu_train_LF))
mu_train_LF = mu_train_LF[permutation][0:10]

U_train_LF_full = U_train_LF_full[permutation][0:10]

# ADD NOISE
# noise_stddev = np.mean(U_train_LF_full) * 0.025
# noise = np.random.normal(0, noise_stddev, np.shape(U_train_LF_full))
# U_train_LF_full = U_train_LF_full + noise

noise_stddev = np.mean(mu_train_LF) * 0.025
noise = np.random.normal(0, noise_stddev, len(mu_train_LF))
mu_train_LF = mu_train_LF + noise

# U_lf_train_full = np.loadtxt("../DATA_shear_cube/Data_Train_bases_young_new/Ulf_train_young.txt")
U_lf_train = U_train_LF_full[:, m]

mu_train_HF = np.loadtxt(
    "../DATA_shear_cube/3p/Data_Train_bases_new/mu_train_HF_"
    + n_HF_txt
)
Nhf = np.size(mu_train_HF)  # number of high-fidelity data to TRAIN the NN

U_hf_train = np.abs(
    np.loadtxt(
        "../DATA_shear_cube/3p/Data_Train_bases_new/Uhf_train_"
        + n_HF_txt
    )
)

# permutation = np.random.permutation(len(mu_train_HF))
# mu_train_HF=mu_train_HF[permutation][0:5]
# U_hf_train=U_hf_train[permutation][0:5]

NepoLF = 3000  # number of epochs for first NN: NN_LF
NepoHF = 3000  # number of epochs for second NN: NN_HF

#########################     TEST SET      ##########################
mu_test = np.loadtxt(
    "../DATA_shear_cube/3p/Data_Test_bases_new/mu_test.txt"
)
N_test = np.size(mu_test)

U_lf_test_full = np.abs(
    np.loadtxt(
        "../DATA_shear_cube/3p/Data_Test_bases_new/Ulf_test.txt"
    )
)
U_lf_test = U_lf_test_full[:, m]

U_hf_test = np.abs(
    np.loadtxt(
        "../DATA_shear_cube/3p/Data_Test_bases_new/Uhf_test.txt"
    )
)

########################     NORMALIZATION  #########################
# Input
mu_max = np.max(mu_test)
mu_min = np.min(mu_test)

mu_test_norm = (mu_test - mu_min) / (mu_max - mu_min)
mu_train_LF_norm = (mu_train_LF - mu_min) / (mu_max - mu_min)
mu_train_HF_norm = (mu_train_HF - mu_min) / (mu_max - mu_min)


# Output
# TRANSFORMATION
# in order to reduce the linear relationship between the Young modulus and the displacement
U_lf_train = np.exp(U_lf_train * 100) * 1e-6
U_lf_test = np.exp(U_lf_test * 100) * 1e-6
U_hf_train = np.exp(U_hf_train * 100) * 1e-6
U_hf_test = np.exp(U_hf_test * 100) * 1e-6

###
# lfmean = np.mean(U_lf_test)
# hfmean = np.mean(U_hf_test)

# U_lf_train = U_lf_train - lfmean
# U_lf_test = U_lf_test - lfmean
# U_hf_train = U_hf_train - hfmean
# U_hf_test = U_hf_test - hfmean
U_t_max_train = np.max(U_lf_train)
U_t_min_train = np.min(U_lf_train)

U_t_max_test = np.max(U_lf_test)
U_t_min_test = np.min(U_lf_test)

U_lf_train = (U_lf_train - U_t_min_train) / (U_t_max_train - U_t_min_train)
U_lf_test = (U_lf_test - U_t_min_test) / (U_t_max_test - U_t_min_test)

U_h_max_train = np.max(U_hf_train)
U_h_min_train = np.min(U_hf_train)

U_h_max_test = np.max(U_hf_test)
U_h_min_test = np.min(U_hf_test)

U_hf_train = (U_hf_train - U_h_min_train) / (U_h_max_train - U_h_min_train)
U_hf_test = (U_hf_test - U_h_min_test) / (U_h_max_test - U_h_min_test)


##########################       FIRST NN: NN_LF     ##########################
K.clear_session()
bestLF_params = {
    "lr": 0.0255,
    "kernel_init": "glorot_uniform",
    "opt": "Adam",
}  # obtained

modelLF = getModel(bestLF_params, "LF")
histLF = modelLF.fit(
    mu_train_LF_norm, U_lf_train, epochs=NepoLF, batch_size=Nlf, verbose=0
)
print("LF NN done")

ULF = modelLF.predict(mu_test_norm)
print("\nLF Model:")

test_mse = np.mean(np.square(U_lf_test - ULF[:, 0]))
print(f"Test MSE: {test_mse}")

r_2 = 1 - np.sum(np.square(U_lf_test - ULF[:, 0])) / np.sum(
    np.square(U_lf_test - np.mean(U_lf_test))
)
print(f"R^2: {r_2}")

##################     Print LF, HF and LF-predicted models      ###################
plt.figure()
plt.plot(mu_test_norm, U_hf_test, "bo", linewidth=1.5, label="HF model")
plt.plot(
    mu_train_HF_norm,
    U_hf_train,
    "b*",
    markersize=7,
    label="HF training points",
)
plt.plot(mu_test_norm, U_lf_test, "ro", linewidth=1.5, label="LF model")
plt.plot(
    mu_train_LF_norm,
    U_lf_train,
    "r*",
    markersize=5,
    label="LF training points",
)
plt.plot(
    mu_test_norm,
    modelLF.predict(mu_test_norm),
    "g*",
    label="Predicted LF model",
)
plt.legend(prop={"size": 8.3})
plt.show()


print(f"\n-------  #HF data = {n_HF}  -------")
start = perf_counter()

##########################    SECOND NN: NN_HF    ##########################
mu_test_help = modelLF.predict(mu_test_norm)[:, 0]
mu_test_in = np.vstack(
    (mu_test_norm, mu_test_help)
).transpose()  # <- TEST INPUT for the second NN: NN_HF

mu_train_help = modelLF.predict(mu_train_HF_norm)[:, 0]  # f_LF(mu_hf_train)
mu_final = np.vstack(
    (mu_train_HF_norm, mu_train_help)
).transpose()  # <- TRAINING INPUT for the second NN: NN_HF

name = "2step"

####################    HYPERPARAMETER OPTIMIZATION    #######################
MAX_EVAL = 15

K.clear_session()
bayes_trials = Trials()
opt_list = ["Adam", "Adamax"]
kernel_list = ["uniform", "glorot_uniform"]
aux_dic = {"opt": opt_list, "kernel_init": kernel_list}
space = {
    "nodes": hp.qloguniform("nodes", np.log(4), np.log(64), 2),
    "l2weight": hp.loguniform("l2weight", np.log(0.0001), np.log(100)),
    "lr": hp.loguniform("lr", np.log(0.0001), np.log(0.1)),
    "kernel_init": hp.choice("kernel_init", kernel_list),
    "opt": hp.choice("opt", opt_list),
}


def objective(params):
    K.clear_session()
    CVres = kCrossVal(Nhf, NepoHF, mu_final, U_hf_train, params, name)

    # mse, r_squared = calculate_metrics(Nhf, NepoHF, mu_final, U_hf_train, params, name)   # ADDED

    # return {'loss': CVres, 'mse': mse, 'r_squared': r_squared, 'params': params, 'status': STATUS_OK}
    return {"loss": CVres, "params": params, "status": STATUS_OK}


best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=MAX_EVAL,
    trials=bayes_trials,
)

# for trial in bayes_trials.trials:
#    print("Parameters:", trial['result']['params'])
#    print("Loss:", trial['result']['loss'])
#    print("MSE:", trial['result']['mse'])
#    print("R-squared:", trial['result']['r_squared'])
#    print("---")

transfBestparam(best_params, aux_dic)
print(best_params)

####################    NN_HF training and PREDICTION    #######################
finalModel = getModel(
    best_params, name
)  # final model chosen according to the best paramters
hist = finalModel.fit(
    mu_final,
    U_hf_train,
    validation_data=(mu_test_in, U_hf_test),
    epochs=NepoHF,
    batch_size=Nhf,
    verbose=0,
    validation_freq=20,
)

UHF = finalModel.predict(mu_test_in)

stop = perf_counter()
elapsed = stop - start
print("Elapsed time: ", elapsed)
print("\nHF Model:")

test_mse = np.mean(np.square(U_hf_test - UHF[:, 0]))
print(f"Test MSE: {test_mse}")

r2_HF = 1 - np.sum(np.square(U_hf_test - UHF[:, 0])) / np.sum(
    np.square(U_hf_test - np.mean(U_hf_test))
)
print(f"R^2: {r2_HF}")

print("Number of basis functions: ", int(Nlf_models[m]))
print("Number of HF data: ", n_HF)

####################    HF GRAPHS    #######################
plt.figure()
plt.plot(
    mu_test_norm,
    UHF,
    "ko",
    label="estimated $U_{HF}$ by $NN_{HF}$",
    linewidth=3,
)  # estimated values of u_HF by third NN
plt.plot(
    mu_train_HF_norm, U_hf_train, "ro", label="HF data"
)  # high fidelity data
plt.plot(
    mu_test_norm, U_hf_test, "k*", label="exact solution"
)  # exact solution
plt.plot(
    mu_test_norm,
    modelLF.predict(mu_test_norm),
    "bo",
    label="estimated $U_{LF}$ by $NN_{LF}$",
    linewidth=1,
)
plt.title("High fidelity model ")
plt.legend(loc=1, prop={"size": 8})
plt.show()

####################    TRAINING INSIGHTS    #######################
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(hist.history["mse"], color="red", label="High fidelity train mse")
plt.plot(histLF.history["mse"], color="black", label="Low fidelity")
plt.legend()
plt.yscale("log")
plt.subplot(2, 1, 2)
plt.plot(hist.history["val_mse"], color="red")
plt.yscale("log")
plt.show()
