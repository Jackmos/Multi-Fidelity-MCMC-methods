#########################     LIBRARIES     ##########################
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
from ann_functions import (
    getModel,
    kCrossVal,
    transfBestparam,
    kCrossValSingle,
    import_data,
)
from time import perf_counter
import pandas
import os

seed = 7
np.random.seed(seed)
# Nlf=15
#{'kernel_init': 'uniform', 'l2weight': 0.02672483771802389, 'lr': 0.0021557686739560574, 'nodes': 32.0, 'opt': 'Adamax'}
#1/1 [==============================] - 0s 62ms/step
#Elapsed time:  2490.439029100002

#LF Model:
#Elapsed time:  16.3247005000012

#LF Model:
#Test MSE: 0.18578364760206803
#R^2: -0.6315290167326626
#1/1 [==============================] - 0s 78ms/step
#Elapsed time:  2519.741864099982

#LF Model:
#Elapsed time:  45.627535499981605

#LF Model:
#Test MSE: 0.21627322394283716
#R^2: -0.725987060875829
#1/1 [==============================] - 0s 16ms/step
#----------------------------------
# new Nlf=20
#{'kernel_init': 'uniform', 'l2weight': 0.0021453159318891644, 'lr': 0.00038420659577907094, 'nodes': 40.0, 'opt': 'Adam'}
#1/1 [==============================] - 0s 67ms/step
#Elapsed time:  3404.7170924000093

#LF Model:
#Elapsed time:  15.824968300003093

#LF Model:
#Test MSE: 0.11635362907708972
#R^2: -0.0218031806977399
#1/1 [==============================] - 0s 68ms/step
#Elapsed time:  3435.4436560000177

#LF Model:
#Elapsed time:  46.55153190001147

#LF Model:
#Test MSE: 0.1272319940057293
#R^2: -0.01538586876275616


# 1/1 [==============================] - 0s 78ms/step
# Elapsed time:  1783.1395402999988

# LF Model:
# Elapsed time:  16.36429810000118

# LF Model:
# Test MSE: 0.11387091776554122
# R^2: -3.1700524827371623e-07

# 1/1 [==============================] - 0s 109ms/step
# Elapsed time:  28684.207620799993

# LF Model on HF data:
# Elapsed time:  26917.432378599995

# LF Model on HF data:
# Test MSE: 0.29078675534444665
# R^2: -4.525715881773088
# 1/1 [==============================] - 0s 16ms/step

########################     PREPARATION      ##########################

file_path_LF = "../DATA/reaction_diffusion_LF_46_big.mat"
(reaction_LF, U_LF) = import_data(file_path_LF)
reaction_LF_test = reaction_LF
U_LF_test = U_LF

NepoLF = 3000  # number of epochs for second NN: NN_HF
permutation = np.random.permutation(len(reaction_LF))
Nlf = 20
reaction_LF = reaction_LF[permutation][0:Nlf]

#########################     TRAIN SET      ##########################
reaction_max = np.max(reaction_LF_test)
reaction_min = np.min(reaction_LF_test)

# reaction_test_norm = (reaction_test - reaction_min) / (reaction_max - reaction_min)
reaction_LF = (reaction_LF - reaction_min) / (reaction_max - reaction_min)
reaction_LF_test = (reaction_LF_test - reaction_min) / (
    reaction_max - reaction_min
)

# Output
# We consider the central value of U
permutation = np.random.permutation(len(reaction_LF))
reaction_LF = reaction_LF[permutation][0:Nlf]

#U_LF = U_LF[:, -1, int(np.shape(U_LF)[2] / 2), int(np.shape(U_LF)[3] / 2)]
U_LF = U_LF[:, -1, 20,20]
U_LF = U_LF[permutation][0:Nlf]
#U_LF_test = U_LF_test[
#    :, -1, int(np.shape(U_LF_test)[2] / 2), int(np.shape(U_LF_test)[3] / 2)
#]
U_LF_test = U_LF_test[
    :, -1, 20,20
]
# TRANSFORMATION
U_h_max_test = np.max(U_LF_test)
U_h_min_test = np.min(U_LF_test)

U_LF = (U_LF - U_h_min_test) / (U_h_max_test - U_h_min_test)
U_LF_test = (U_LF_test - U_h_min_test) / (U_h_max_test - U_h_min_test)


start = perf_counter()
#########
reaction_final = np.vstack(
    (reaction_LF)
)  # .transpose() # <- TRAINING INPUT for the second NN: NN_HF
########
##########################       NN_LF     ##########################
####################    HYPERPARAMETER OPTIMIZATION    #######################
MAX_EVAL = 5
name = "LF"
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
    CVres = kCrossValSingle(Nlf, NepoLF, reaction_final, U_LF, params, name)
    return {"loss": CVres, "params": params, "status": STATUS_OK}


best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=MAX_EVAL,
    trials=bayes_trials,
)

transfBestparam(best_params, aux_dic)
print(best_params)

####################    NN training and PREDICTION    #######################
start2 = perf_counter()
finalModel = getModel(
    best_params, name
)  # final model chosen according to the best paramters
hist = finalModel.fit(
    reaction_LF,
    U_LF,
    validation_data=(reaction_LF_test, U_LF_test),
    epochs=NepoLF,
    batch_size=Nlf,
    verbose=0,
    validation_freq=20,
)

ULF = finalModel.predict(reaction_LF_test)

stop = perf_counter()
elapsed = stop - start
print("Elapsed time: ", elapsed)
print("\nLF Model:")

elapsed2 = stop - start2
print("Elapsed time: ", elapsed2)
print("\nLF Model:")

test_mse = np.mean(np.square(U_LF_test - ULF[:, 0]))
print(f"Test MSE: {test_mse}")

r2_LF = 1 - np.sum(np.square(U_LF_test - ULF[:, 0])) / np.sum(
    np.square(U_LF_test - np.mean(U_LF_test))
)
print(f"R^2: {r2_LF}")

# print("Number of basis functions: ", int(Nlf_models[m]))
# print('Number of LF data: ', n_LF)

####################    NN training and PREDICTION HF   #######################
# start2 = perf_counter()
finalModel = getModel(
    best_params, name
)  # final model chosen according to the best paramters

file_path_HF = "../DATA/reaction_diffusion_HF.mat"
(reaction_HF_test, U_HF_test) = import_data(file_path_HF)
#U_HF_test = U_HF_test[
#    :, -1, int(np.shape(U_HF_test)[2] / 2), int(np.shape(U_HF_test)[3] / 2)
#]
U_HF_test = U_HF_test[
    :, -1, 44,44
]

reaction_HF_test = (reaction_HF_test - np.min(reaction_HF_test)) / (
    np.max(reaction_HF_test) - np.min(reaction_HF_test)
)
U_HF_test = (U_HF_test - np.min(U_HF_test)) / (
    np.max(U_HF_test) - np.min(U_HF_test)
)

hist = finalModel.fit(
    reaction_LF,
    U_LF,
    validation_data=(reaction_HF_test, U_HF_test),
    epochs=NepoLF,
    batch_size=Nlf,
    verbose=0,
    validation_freq=20,
)

UHF = finalModel.predict(reaction_HF_test)

stop = perf_counter()
elapsed = stop - start
print("Elapsed time: ", elapsed)
print("\nLF Model:")

elapsed2 = stop - start2
print("Elapsed time: ", elapsed2)
print("\nLF Model:")

test_mse_HF = np.mean(np.square(U_HF_test - UHF[:, 0]))
print(f"Test MSE: {test_mse_HF}")

r2_HF = 1 - np.sum(np.square(U_HF_test - UHF[:, 0])) / np.sum(
    np.square(U_HF_test - np.mean(U_HF_test))
)
print(f"R^2: {r2_HF}")

# print("Number of basis functions: ", int(Nlf_models[m]))

##################     Print LF, HF and LF-predicted models      ###################
plt.figure()
plt.plot(reaction_HF_test, U_HF_test, "b*", linewidth=1.5, label="HF model")
# plt.plot(mu_train_HF, U_hf_train,'b*', markersize = 7, label = 'HF training points')
plt.plot(reaction_LF_test, U_LF_test, "ro", linewidth=1.5, label="LF model")
plt.plot(
    reaction_LF,
    U_LF,
    "r*",
    markersize=5,
    label="LF training points",
)
plt.plot(
    reaction_HF_test,
    finalModel.predict(reaction_HF_test),
    "g*",
    linewidth=3,
    label="Predicted HF model",
)
plt.legend(prop={"size": 8.3})
plt.show()

#########################     SAVE the OUTPUT      ##########################
os.makedirs("Output_NNLF_end")

R2_HF = pandas.DataFrame({"R2_HF": [r2_HF]})
R2_LF = pandas.DataFrame({"R2_LF": [r2_LF]})
MSE_test_LF = pandas.DataFrame({"MSE_test_LF": [test_mse]})
MSE_test_HF = pandas.DataFrame({"MSE_test_HF": [test_mse_HF]})


R2_LF.to_csv(
    "./Output_NNLF_end/r2_HF.txt",
    header=True,
    index=False,
    sep="\t",
    mode="a",
)
MSE_test_HF.to_csv(
    "./Output_NNLF_end/test_mse.txt",
    header=True,
    index=False,
    sep="\t",
    mode="a",
)
R2_HF.to_csv(
    "./Output_NNLF_end/r2_LF_lhs.txt",
    header=True,
    index=False,
    sep="\t",
    mode="a",
)
MSE_test_LF.to_csv(
    "./Output_NNLF_end/mse_LF_lhs.txt",
    header=True,
    index=False,
    sep="\t",
    mode="a",
)
