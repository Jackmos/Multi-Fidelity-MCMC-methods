# The following NN architectures are the ones introduced by Maurice Amendt.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Dense, Input

# from keras.layers.merge import concatenate
from tensorflow.keras.layers import (
    concatenate,
)  # PROBLEMA SEMBRA RISOLTO COSI !!!
from keras.regularizers import l2
from sklearn.model_selection import KFold
import numpy as np
from keras.optimizers import Adam, Nadam, Adamax, RMSprop
import keras.backend as K
import tensorflow as tf
import h5py

def custom_activation(x):
    return x + K.square(K.sin(x))

def  normalization(x):
    return (x - np.min(x)) / (
    np.max(x) - np.min(x)
)

def import_data(name):
    with h5py.File(name, "r") as file:
        
        t = file["t"][()]

        U = file["U"][()]

        # V=file['V']
        # V=V[()]

    return (t.T, U)


def custom_loss(y_pred, y_true):
    goodind = K.not_equal(y_pred, -10)
    # -10 is used as special value to indicate NaN

    # goodind = tf.math.logical_not(tf.math.is_nan(y_pred))
    y_pred_loss = tf.boolean_mask(y_pred, goodind)
    y_pred_true = tf.boolean_mask(y_true, goodind)
    return K.mean(K.square(y_pred_loss - y_pred_true))  # MSE


def getOpti(name, lr):
    if name == "Adam":
        return Adam(learning_rate=lr, amsgrad=True)
    elif name == "Nadam":
        return Nadam(learning_rate=lr)
    elif name == "Adamax":
        return Adamax(learning_rate=lr)
    elif name == "RMSprop":
        return RMSprop(learning_rate=lr)
    elif name == "standardadam":
        return "adam"


def getModel(params, name):
    if name == "2step":
        inputs = Input(shape=(2,))  #
        # NN_HF is a  shallow neural network consisting of a single layer:
        hidden1 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(inputs)
        # hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        output = Dense(1, activation="linear", name="HF")(hidden1)
        # y_HF is the output

    elif name == "LF":
        inputs = Input(shape=(2,))
        hidden1 = Dense(
            64, activation=custom_activation, kernel_initializer=params["kernel_init"]
        )(inputs)
        hidden2 = Dense(
            64, activation=custom_activation, kernel_initializer=params["kernel_init"]
        )(hidden1)
        hidden3 = Dense(
            64, activation=custom_activation, kernel_initializer=params["kernel_init"]
        )(hidden2)
        hidden4 = Dense(
            64, activation=custom_activation, kernel_initializer=params["kernel_init"]
        )(hidden3)
        output = Dense(1, activation="sigmoid", name="LF")(hidden4)

    elif name == "Single":
        inputs = Input(shape=(1,))
        hidden1 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"]),
        )(inputs)
        hidden2 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"]),
        )(hidden1)
        hidden3 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"]),
        )(hidden2)
        hidden4 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"]),
        )(hidden3)
        output = Dense(1, activation="sigmoid", name="Single")(hidden2)

    elif name == "Hflin":
        inputs = Input(
            shape=(2,)
        )  # second NN (NN_Lin) in the 3-steps architecture
        # Linear activation function: it approximates the high-fidelity data by a linear combiantion of the inputs
        # and is thus responsible for capturing the linear correlations between the datasets
        hiddenlin = Dense(
            64,
            activation="linear",
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(inputs)
        output = Dense(1, activation="sigmoid", name="HFlin")(hiddenlin)

    elif name == "3step":
        inputs = Input(
            shape=(4,)
        )  # third NN (NN_HF) in the 3-steps architecture
        hidden1 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(inputs)
        output = Dense(1, activation="sigmoid", name="HF")(hidden1)

    elif name == "GP":
        # architecture which is supposed to mimic the action of a GP
        inputs = Input(shape=(1,))
        hidden1 = Dense(
            int(params["nodes"]),
            activation="tanh",
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(inputs)
        hidden2 = Dense(
            int(params["nodes"]),
            activation="tanh",
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden1)
        hidden3 = Dense(
            int(params["nodes"]),
            activation="tanh",
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden2)
        hidden4 = Dense(
            int(params["nodes"]),
            activation="tanh",
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden3)

        GPlayer = Dense(
            2,
            activation="linear",
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden4)

        outputLF = Dense(1, activation="linear", name="LF")(GPlayer)
        outputHF = Dense(1, activation="linear", name="HF")(GPlayer)

        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params["opt"], params["lr"])
        model.compile(
            loss=custom_loss,
            loss_weights=[params["alpha"], 1 - params["alpha"]],
            optimizer=opti,
        )
        return model

    elif name == "Inter":
        inputs = Input(shape=(1,))

        hidden1 = Dense(
            64, activation="tanh", kernel_initializer=params["kernel_init"]
        )(
            inputs
        )  # The same input layer is used for high and low fidelity data
        hidden2 = Dense(
            64, activation="tanh", kernel_initializer=params["kernel_init"]
        )(hidden1)
        outputLF = Dense(1, activation="linear", name="LF")(
            hidden2
        )  # low-fidelity output is situate at the third hidden layer.
        outputadd = Dense(
            int(params["nodes"]),
            activation="tanh",
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden2)
        merge = concatenate([outputLF, outputadd])

        hidden3 = Dense(
            int(params["nodes"]),
            activation="tanh",
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(merge)

        hidden4 = Dense(
            int(params["nodes"]),
            activation="tanh",
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden3)

        # Possible additional layers:

        # hidden5 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
        # hidden6 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden5)

        # lincorr = Dense(int(params['nodes']),activation='linear',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(outputLF)
        # merge2 = concatenate([hidden3,lincorr])

        outputHF = Dense(1, activation="sigmoid", name="HF")(hidden4)
        output = [outputHF, outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params["opt"], params["lr"])
        model.compile(
            loss=custom_loss,
            loss_weights=[params["alpha"], 1 - params["alpha"]],
            optimizer=opti,
        )
        # alpha weighs the different fidelity level components of the error
        return model

    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params["opt"], params["lr"])
    model.compile(loss="mse", optimizer=opti, metrics=["mse"])
    return model


def kCrossVal(N, Nepo, x, y, params, name):
    # K.clear_session()
    model = getModel(params, name)
    score = np.zeros([N])
    cv = KFold(N)
    splits = cv.split(x)
    i = 0
    for train_index, test_index in splits:
        x_train = x[train_index, :]
        y_train = y[train_index]
        x_val = x[test_index, :]
        y_val = y[test_index]
        # model = getModel(params,name)
        model.fit(x_train, y_train, epochs=Nepo, batch_size=N - 1, verbose=0)
        score[i] = np.square(y_val - model.predict(x_val))
        i = i + 1
    return np.mean(score)


def kCrossValSingle(N, Nepo, x, y, params, name):
    score = np.zeros([N])
    cv = KFold(N)
    splits = cv.split(x)
    model = getModel(params, name)
    i = 0
    for train_index, test_index in splits:
        x_train = x[train_index]
        y_train = y[train_index]
        x_val = x[test_index]
        y_val = y[test_index]
        model.fit(x_train, y_train, epochs=Nepo, batch_size=N - 1, verbose=0)
        score[i] = np.square(y_val - model.predict(x_val))
        i = i + 1
    return np.mean(score)


def kCrossValGP(Nhf, Nlf, Nepo, xhf, yhf, xlf, ylf, params, name):
    score = np.zeros([Nhf])
    cv = KFold(Nhf)
    splits = cv.split(xhf)
    i = 0
    N = Nhf + Nlf
    model = getModel(params, name)
    for train_index, test_index in splits:
        xhf_train = xhf[train_index]
        yhf_train = yhf[train_index]
        xhf_val = xhf[test_index]
        yhf_val = yhf[test_index]
        x_train = np.concatenate((xhf_train, xlf))
        yhf_train = np.concatenate((yhf_train, np.full(Nlf, -10)))
        ylf_train = np.concatenate((np.full(Nhf - 1, -10), ylf))
        model.fit(
            x_train,
            [yhf_train, ylf_train],
            epochs=int(params["epochs"]) * Nepo,
            batch_size=N - 1,
            verbose=0,
        )
        score[i] = np.square(yhf_val - model.predict(xhf_val)[0])
        i = i + 1
    return np.mean(score)


def transfBestparam(bestparam, dic):
    for key in bestparam:
        if key in ["kernel_init", "opt"]:
            bestparam[key] = dic[key][bestparam[key]]
    return
