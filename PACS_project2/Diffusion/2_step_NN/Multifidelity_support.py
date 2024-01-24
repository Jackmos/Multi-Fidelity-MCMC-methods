from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
from keras.models import save_model
import numpy as np
from matplotlib import pyplot as plt
from ann_functions import getModel, kCrossVal, transfBestparam
from time import perf_counter
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers import Layer
from tensorflow.keras.layers import (
    concatenate,
)  
from keras.regularizers import l2, l1
from sklearn.model_selection import KFold
import numpy as np
from keras.optimizers import Adam, Nadam, Adamax, RMSprop
import keras.backend as K
import tensorflow as tf
import keras as kr
import h5py
import sys
import os
import warnings

from cuqi.distribution import Uniform, Gaussian,JointDistribution
from cuqi.sampler import MH
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Discrete

class Suppressor:
    # suppress the printed message
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._stdout


class Neural_network:
    
    def __init__(self,name,params=None,data_train=None,output_train=None,N=1000,n=10,train=True,do_HPO=False,verbose=False):
        K.clear_session()
        self.name=name
        self.params=params
        self.N=N    # epochs
        self.n=n    # batch size    
        self.verbose=verbose
        self.hist=None
        self.data_train=data_train
        self.output_train=output_train
        
        if data_train is not None and len(data_train.shape) > 1:
            self.shape_value = data_train.shape[1]
        else:
            self.shape_value = 1  

        if (do_HPO or params is None):
            if (output_train is None or data_train is None):
                warning_message = "Not enough data given!"
                warnings.warn(warning_message, UserWarning)
            self.params=self.HPO(data_train,output_train)
        self.model = getModel(self.params,self.shape_value,self.name)

        if(train):
            self.hist=self.model.fit(data_train,output_train,epochs=self.N,batch_size=self.n,verbose=0) 
            plt.plot(self.model.history.history['loss'][100:], label='Training Loss')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()
    
    def training(self,x,y,epoch,batch):
        # DUBBIO : va definita variabile hist?
        self.hist=self.model.fit(x,y,epochs=epoch,batch_size=batch,verbose=0) 
        return self.hist
    
    def prediction(self,x_test):
        if(self.verbose):
            y_pred=self.model.predict(x_test)[:,0]
        else:
            with Suppressor():
                y_pred = self.model.predict(x_test)[:,0]
        return y_pred    

    def save(self):
        print("saving the model ...")
        save_model(self.model,f"{self.name}_model.h5")


    def performance(self,data_test,output_test):
        pred=self.prediction(data_test)#[:,0]
        #print(pred.shape)
        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2= 1 - np.sum(np.square(output_test - pred)) / np.sum(
            np.square(output_test - np.mean(output_test))
        )
        print(f"R^2: {r2}")
        
        return (test_mse,r2)

    def inverse(self, y_obs, N=1000, burn_in=500, proposal_sd=0.3,x_init=None,diagnostic=True):
        dim=y_obs.shape[0]
        if (N<=burn_in):
                warning_message = "number of steps insufficient, smaller or equal than burn-in"
                warnings.warn(warning_message, UserWarning)        
        if (x_init is None):
            x_init=np.zeros(dim)
        elif isinstance(x_init, (int, float)):
            x_init=x_init*np.ones(dim)
        A=CuqiModel(forward=self.prediction,range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
        x=Uniform(np.zeros(dim),np.ones(dim))
        y=Gaussian(mean=A(x),cov=proposal_sd)
        # y_obs=y(x=real_x).sample()
        posterior=JointDistribution(y,x)(y=y_obs)
        
        sampler=MH(posterior,x0=x_init)   # rendere variabile per altri sampler
        samples=sampler.sample_adapt(N-burn_in,burn_in)
        
        if(diagnostic is True):
            samples.plot_trace()
            mean=samples.mean()
            print(f"Mean values= {mean}")
            ESS=samples.compute_ess()
            print(f"ESS= {ESS}")
            samples.plot_autocorrelation()    
    # def mse_plot(self,val):
    #     plt.plot(self.model.history.history['loss'][val:], label='Training Loss')
    #     plt.title('Mean Squared Error (MSE) over Epochs')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('MSE')
    #     plt.legend()
    #     plt.show()
    def objective(self,par): 
        #print(self.name)
        K.clear_session()
        CVres = kCrossVal(self.n,self.N,self.data_train,self.output_train,par,self.name,self.shape_value)  # ATTENZIONE!! NEL CASO LF, data_train E output_train dovrebbero essere HF (vedere codice Nlf di esempio), correggere
        return {"loss": CVres, "params": par, "status": STATUS_OK}   


    def HPO(self,data_train,output_train):

        MAX_EVAL = 5

        bayes_trials = Trials()
        opt_list = ["Adam", "Adamax"]
        kernel_list = ["uniform", "glorot_uniform"]
        aux_dic = {"opt": opt_list, "kernel_init": kernel_list}
        space = {
        "nodes": hp.qloguniform("nodes", np.log(4), np.log(64), 2),
        "l2weight": hp.loguniform("l2weight", np.log(0.0001), np.log(1)),
        "lr": hp.loguniform("lr", np.log(0.0001), np.log(0.1)),
        "kernel_init": hp.choice("kernel_init", kernel_list),
        "opt": hp.choice("opt", opt_list),
        }

        best_params = fmin(fn = self.objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = MAX_EVAL,
                        trials = bayes_trials)
        transfBestparam(best_params, aux_dic)
        #self.params=best_params
        
        print("Best parameters from the HPO:")
        print(best_params)
        return best_params

class MultiFidelity():
   
    def __init__(self,names,params=None,data_train_LF=None,output_train_LF=None,data_train_HF=None, output_train_HF=None,N=None,n=None,do_HPO=False,verbose=False):
        # names: list of strings
        # params: list of dictionaries
        # N: list of epochs
        # n: list of batch_sizes
        # data_train: list of inputs
        # output_train: list of outputs 
        self.names=names
        self.Ns=N
        self.ns=n
        self.model_list = []
        self.outputs = np.empty((0,0))
        
        K.clear_session()
        if len(params)<len(self.names):
            diff = len(self.names) - len(params)
            params += [None] * diff
                        
        for index, name in enumerate(self.names):  
            if(name=='LF'):
                model=Neural_network(name,params=params[index],data_train=data_train_LF,output_train=output_train_LF,N=self.Ns[index],n=self.ns[index],train=True,do_HPO=do_HPO,verbose=verbose)
                self.model_list.append(model)
            else:
                model=Neural_network(name,params=params[index],data_train=data_train_HF,output_train=output_train_HF,N=self.Ns[index],n=self.ns[index],train=True,do_HPO=do_HPO,verbose=verbose)
                self.model_list.append(model)
            # self.outputs.append(model.prediction(self.outputs)) 
            data_train_HF=np.c_[data_train_HF,model.prediction(data_train_HF)]     # caso con LF di seguito non è mai capitato finora    LINEA VERA
            #data_train_HF=np.c_[data_train_HF,output_train_HF]     # caso con LF di seguito non è mai capitato finora
        
    def prediction(self,data_test):
        self.outputs = data_test
        if(len(self.outputs.shape)==1):
            self.outputs=self.outputs.reshape(-1,1)
        for index, _ in enumerate(self.names):
            #print("check")
            self.outputs=np.c_[self.outputs,self.model_list[index].prediction(self.outputs)]
        return self.outputs[:,-1]
 
    def inverse(self, y_obs, N=1000, burn_in=500, proposal_sd=0.3,x_init=None,diagnostic=True):
        dim=y_obs.shape[0]
        if (N<=burn_in):
                warning_message = "number of steps insufficient, smaller or equal than burn-in"
                warnings.warn(warning_message, UserWarning)        
        if (x_init is None):
            x_init=np.zeros(dim)
        elif isinstance(x_init, (int, float)):
            x_init=x_init*np.ones(dim)
        A=CuqiModel(forward=self.prediction,range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
        x=Uniform(np.zeros(dim),np.ones(dim))
        y=Gaussian(mean=A(x),cov=proposal_sd)
        # y_obs=y(x=real_x).sample()
        posterior=JointDistribution(y,x)(y=y_obs)
        
        sampler=MH(posterior,x0=x_init)   # rendere variabile per altri sampler
        samples=sampler.sample_adapt(N-burn_in,burn_in)
        
        if(diagnostic is True):
            samples.plot_trace()
            mean=samples.mean()
            print(f"Mean values= {mean}")
            ESS=samples.compute_ess()
            print(f"ESS= {ESS}")
            samples.plot_autocorrelation() 
    
    def save(self):
        print("Saving ...")
        for index, _ in enumerate(self.model_list):
            save_model(self.model_list[index].model,f"{self.names[index]}_model.h5")
            
    def get_output(self):
        return self.outputs[-1]

class FourierLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FourierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel_sin = self.add_weight(name='kernel_sin',
                                          shape=(self.output_dim,),  
                                          initializer='glorot_uniform',
                                          trainable=True)
        self.kernel_cos = self.add_weight(name='kernel_cos',
                                          shape=(self.output_dim,),   
                                          initializer='glorot_uniform',
                                          trainable=True)
        super(FourierLayer, self).build(input_shape)

    def call(self, x):
        result = tf.sin(tf.multiply(x, self.kernel_sin)) + tf.cos(tf.multiply(x, self.kernel_cos))
        return result

    def compute_output_shape(self, input_shape):
        return input_shape

    
def custom_activation(x):
    return x + K.square(K.sin(x))

def  normalization(x):
    return (x - np.min(x)) / (
    np.max(x) - np.min(x)
)

def import_data(name):#-> Tuple[np.array, np.array]:
    """imports data defined in a .mat file

    Args:
        name : name of the file

    Returns:
        Tuple[np.array, np.array]: input and output of the NN
    """
    with h5py.File(name, "r") as file:
        
        R = file["betas"][()]

        U = file["U"][()]

        # V=file['V']
        # V=V[()]

    return (R, U)

# NON USATA
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

def add_noise(noise_std_data, noise_sta_output, data, output):
    output_flag=output
    data_flag=data
    for std1,std2 in zip(noise_std_data,noise_sta_output):
        noise_1 = np.random.normal(0, std1, output.shape[0])
        noise_2 = np.random.normal(0, std2, data.shape)
        temp1=output+noise_1
        temp2=data+noise_2
        output_flag=np.concatenate((output_flag,temp1),axis=0)
        #print(data_flag.shape)
        #print(temp2.shape)
        data_flag=np.concatenate((data_flag,temp2))
    return (output_flag,data_flag)

def getModel(params,num_inputs, name):
    if name == "2step":
        inputs = Input(shape=(num_inputs,))  
        
        hidden1 = Dense(
            int(params["nodes"]),
            kernel_regularizer=l2(params["l2weight"]),
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
        )(
            inputs
        )  
        hidden1=Dropout(0.05)(hidden1)

        fourier_layer = FourierLayer(output_dim=int(params["nodes"]))(hidden1)
        
        hidden2 = Dense(
            int(params["nodes"]),
            kernel_regularizer=l2(params["l2weight"]),
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            
        )(
            fourier_layer
        ) 
                
        output = Dense(1, activation="linear", name="HF")(hidden2)


    elif name == "LF":
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
            
        )(inputs)
        hidden2 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
            
        )(hidden1)
        fourier_layer1 = FourierLayer(output_dim=64)(hidden2)

        hidden3 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
           
        )(fourier_layer1)
        
        hidden3=Dropout(0.05)(hidden3)

        hidden4 = Dense(
            64,
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            kernel_regularizer=l2(0.001),
            #kernel_constraint=clip_norm(1.0)
        )(hidden3)

        
        output = Dense(1, activation="linear", name="LF")(hidden4)

    elif name == "HF":
        inputs = Input(shape=(num_inputs,))  
        hidden1 = Dense(
            int(params["nodes"]),
            kernel_regularizer=l2(params["l2weight"]),
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
           
        )(
            inputs
        )  
        hidden1=Dropout(0.05)(hidden1)

        fourier_layer = FourierLayer(output_dim=int(params["nodes"]))(hidden1)
        
        hidden2 = Dense(
            int(params["nodes"]),
            kernel_regularizer=l2(params["l2weight"]),
            activation=custom_activation,
            kernel_initializer=params["kernel_init"],
            #kernel_constraint=clip_norm(1.0)
        )(
            fourier_layer
        ) 
                
        output = Dense(1, activation="linear", name="HF")(hidden2)

    elif name == "Single":
        inputs = Input(shape=(num_inputs,))
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
            shape=(num_inputs,)
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
            shape=(num_inputs,)
        )  
        hidden1 = Dense(
            int(params["nodes"]),
            activation=custom_activation,
            kernel_regularizer=l2(params["l2weight"]),
            kernel_initializer=params["kernel_init"],
        )(inputs)
        output = Dense(1, activation="sigmoid", name="HF")(hidden1)

    elif name == "GP":
        # architecture which is supposed to mimic the action of a GP
        inputs = Input(shape=(num_inputs,))
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
        inputs = Input(shape=(num_inputs,))

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


def kCrossVal(N, Nepo, x, y, params, name,shape_value):
    # K.clear_session()
    #print(name)
    model = getModel(params,shape_value, name)
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




def kCrossValSingle(N, Nepo, x, y, params, name,shape_value):
    score = np.zeros([N])
    cv = KFold(N)
    splits = cv.split(x)
    model = getModel(params,shape_value, name)
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


def kCrossValGP(Nhf, Nlf, Nepo, xhf, yhf, xlf, ylf, params, name,shape_value):
    score = np.zeros([Nhf])
    cv = KFold(Nhf)
    splits = cv.split(xhf)
    i = 0
    N = Nhf + Nlf
    model = getModel(params,shape_value, name)
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
    # trasforma kernel e opt da parametri numerici (indicatori booleani) a valore stringa
    for key in bestparam:
        if key in ["kernel_init", "opt"]:
            bestparam[key] = dic[key][bestparam[key]]
    return


