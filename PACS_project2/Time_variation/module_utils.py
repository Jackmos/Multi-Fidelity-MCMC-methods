from keras.models import Model
from keras.layers import Dense, Input, Dropout, Layer
from tensorflow.keras.layers import concatenate
from keras.regularizers import l2, l1
from sklearn.model_selection import KFold
import numpy as np
from keras.optimizers import Adam, Nadam, Adamax, RMSprop
import keras.backend as K
import tensorflow as tf
import keras as kr
import h5py
from typing import Tuple

class FourierLayer(Layer):
    """Custom Keras layer for Fourier Transform-based activations."""
    
    def __init__(self, output_dim: int, **kwargs):
        """Initialize FourierLayer with the dimension of the output."""
        self.output_dim = output_dim
        super(FourierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Create trainable weights for the sine and cosine functions."""
        self.kernel_sin = self.add_weight(name='kernel_sin',
                                          shape=(self.output_dim,),  
                                          initializer='glorot_uniform',
                                          trainable=True)
        self.kernel_cos = self.add_weight(name='kernel_cos',
                                          shape=(self.output_dim,),   
                                          initializer='glorot_uniform',
                                          trainable=True)
        super(FourierLayer, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply the sine and cosine transformations to the input tensor."""
        result = tf.sin(tf.multiply(x, self.kernel_sin)) + tf.cos(tf.multiply(x, self.kernel_cos))
        return result

    def compute_output_shape(self, input_shape) -> Tuple[int, ...]:
        """Compute output shape to be same as input shape."""
        return input_shape

def custom_activation(x: tf.Tensor) -> tf.Tensor:
    """Custom activation function: adds square of sine to input tensor."""
    return x + K.square(K.sin(x))

def normalization(x: np.array) -> np.array:
    """Normalize a NumPy array to range [0, 1]."""
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def import_data(name: str) -> Tuple[np.array, np.array]:
    """
    Imports data from a .mat file.

    Args:
        name (str): Name of the .mat file.

    Returns:
        Tuple[np.array, np.array]: Input and output data from the file.
    """
    with h5py.File(name, "r") as file:
        t = file["t"][()]  # Extract 't' dataset and transpose
        U = file["U"][()]  # Extract 'U' dataset
    return t.T, U

def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom loss function ignoring specific invalid values.

    Args:
        y_true (tf.Tensor): True values tensor.
        y_pred (tf.Tensor): Predicted values tensor.

    Returns:
        tf.Tensor: Mean squared error considering only valid values.
    """
    goodind = K.not_equal(y_pred, -10)  # Consider only values not equal to -10
    y_pred_valid = tf.boolean_mask(y_pred, goodind)
    y_true_valid = tf.boolean_mask(y_true, goodind)
    return K.mean(K.square(y_pred_valid - y_true_valid))  # MSE of valid values

def get_optimizer(name: str, lr: float) -> tf.keras.optimizers.Optimizer:
    """
    Select and return an optimizer based on name and learning rate.

    Args:
        name (str): Name of the optimizer ('Adam', 'Nadam', 'Adamax', 'RMSprop', etc.)
        lr (float): Learning rate for the optimizer.

    Returns:
        tf.keras.optimizers.Optimizer: The selected optimizer.
    """
    if name == "Adam":
        return Adam(learning_rate=lr, amsgrad=True)
    elif name == "Nadam":
        return Nadam(learning_rate=lr)
    elif name == "Adamax":
        return Adamax(learning_rate=lr)
    elif name == "RMSprop":
        return RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def getModel(params, name):
    if name == "2step":
        inputs = Input(shape=(3,))  # 4
        # NN_HF is a  shallow neural network consisting of a single layer:
        hidden1 = Dense(
            int(params["nodes"]),
            kernel_regularizer=l2(params["l2weight"]),
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            #kernel_constraint=clip_norm(1.0)
        )(
            inputs
        )  # kernel_regularizer=l2(params['l2weight']),
        # hidden1=Dropout(0.5)(hidden1)
        # hidden2 = Dense(int(params['nodes']),kernel_regularizer=l2(params['l2weight']),activation='tanh',kernel_initializer=params['kernel_init'])(hidden1)     # kernel_regularizer=l2(params['l2weight']),
        hidden1=Dropout(0.05)(hidden1)

        fourier_layer = FourierLayer(output_dim=int(params["nodes"]))(hidden1)

        output = Dense(1, activation="sigmoid", name="HF")(fourier_layer)

        # y_HF is the output

    elif name == "LF":
        inputs = Input(shape=(2,))
        hidden1 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
            #kernel_constraint=clip_norm(1.0)
        )(inputs)
        hidden2 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
            #kernel_constraint=clip_norm(1.0)
        )(hidden1)
        fourier_layer1 = FourierLayer(output_dim=64)(hidden2)

        hidden3 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
           # kernel_constraint=clip_norm(1.0)
        )(fourier_layer1)
        hidden3=Dropout(0.03)(hidden3)

        hidden4 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
            #kernel_constraint=clip_norm(1.0)
        )(hidden3)
        hidden4=Dropout(0.03)(hidden4)
        output = Dense(1, activation="linear", name="LF")(hidden4)

    elif name == "HF":
        inputs = Input(shape=(2,))  # 4
        # NN_HF is a  shallow neural network consisting of a single layer:
        hidden1 = Dense(
            int(params["nodes"]),
            kernel_regularizer=l2(params["l2weight"]),
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            #kernel_constraint=clip_norm(1.0)
        )(
            inputs
        )  # kernel_regularizer=l2(params['l2weight']),
        # hidden1=Dropout(0.5)(hidden1)
        # hidden2 = Dense(int(params['nodes']),kernel_regularizer=l2(params['l2weight']),activation='tanh',kernel_initializer=params['kernel_init'])(hidden1)     # kernel_regularizer=l2(params['l2weight']),
        hidden1=Dropout(0.05)(hidden1)

        fourier_layer = FourierLayer(output_dim=int(params["nodes"]))(hidden1)

        output = Dense(1, activation="sigmoid", name="HF")(fourier_layer)

    elif name == "Single":
        inputs = Input(shape=(2,))
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
            activation="sigmoid",
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(params["l2weight"]),
        )(hidden3)
        output = Dense(1, activation="sigmoid", name="Single")(hidden2)

    elif name == "Hflin":
        inputs = Input(
            shape=(3,)
        )  # second NN (NN_Lin) in the 3-steps architecture
        # Linear activation function: it approximates the high-fidelity data by a linear combiantion of the inputs
        # and is thus responsible for capturing the linear correlations between the datasets
        hiddenlin = Dense(
            64,
            activation=custom_activation,
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
        inputs = Input(shape=(2,))
        hidden1 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(inputs)
        hidden2 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden1)
        hidden3 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden2)
        hidden4 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden3)

        GPlayer = Dense(
            2,
            activation=custom_activation,
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
        inputs = Input(shape=(2,))

        hidden1 = Dense(
            64, activation=custom_activation, kernel_initializer=params["kernel_init"]
        )(
            inputs
        )  # The same input layer is used for high and low fidelity data
        hidden2 = Dense(
            64, activation=custom_activation, kernel_initializer=params["kernel_init"]
        )(hidden1)
        outputLF = Dense(1, activation=custom_activation, name="LF")(
            hidden2
        )  # low-fidelity output is situate at the third hidden layer.
        outputadd = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(hidden2)
        merge = kr.merge.concatenate([outputLF, outputadd])

        hidden3 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2((1 - params["alpha"]) * params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(merge)

        hidden4 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
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

