import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
from ann_functions import getModel, kCrossVal, transfBestparam
from time import perf_counter
import tensorflow as tf
import sys
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.optimizers import Adam,Nadam,Adamax, RMSprop
import tensorflow.keras.backend as K
import tensorflow as tf

#######################################################################
# rendere variabile il numero di input della rete in modo da evitare problemi in def classe
######################################################################

class Suppressor:
    # suppress the printed message
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._stdout


class Neural_network:
    
    #def __init__(self, data, labels, params, num_classes=10, verbose=False):
    def __init__(self,params,name,data_train=None,output_train=None,N=1000,n=10,train=True,do_HPO=False,verbose=False):
        K.clear_session()
        self.verbose=verbose
        self.model = getModel(params,name)
        if (do_HPO):
            params=HPO()
        if(train):
            self.model.fit(data_train,output_train,epochs=N,batch_size=n,verbose=0) 
    
    def training(self,x,y,epoch,batch):
        # DUBBIO : va definita variabile hist?
        self.model.fit(x,y,epochs=epoch,batch_size=batch,verbose=0) 

    def prediction(self,x_test):
        if(self.verbose):
            y_pred=self.model.predict(x_test)[:,0]
        else:
            with Suppressor():
                y_pred = self.model.predict(x_test)[:,0]
        return y_pred    
    
    def predict(self, x):
        if (self.verbose):
            y = self.model.predict(np.array([x]))[:,0]
        else:
            with Suppressor():
                y = self.model.predict(np.array([x]))[:,0]
        return y

    def objective(self,params):
        # wrapper
        K.clear_session()
        CVres = kCrossVal(n,N,data_train,output_train,params,name)  # ATTENZIONE!! NEL CASO LF, data_train E output_train dovrebbero essere HF (vedere codice Nlf di esempio), correggere
        return {"loss": CVres, "params": params, "status": STATUS_OK}

    def HPO():
        # wrapper
        MAX_EVAL = 15

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

        best_params = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = MAX_EVAL,
                        trials = bayes_trials)
        transfBestparam(best_params, aux_dic)
        print(best_params)
        return best_params

class MultiFidelity(Neural_network):
    # idea: se rendi input in numeri variabili, puoi mettere assieme piu modelli in modo diverso e generico
    # fai lista di nomi degli elementi che entrano (per evitare siano solo 2, cosi generalizzi alla 3 steps) 
    # in ogni riga aggiungi colonna per prosecutio
    
    
    
    
def custom_loss(y_pred,y_true):
    goodind = K.not_equal(y_pred,-10)
    #goodind = tf.math.logical_not(tf.math.is_nan(y_pred))
    y_pred_loss = tf.boolean_mask(y_pred,goodind)
    y_pred_true = tf.boolean_mask(y_true,goodind)
    return K.mean(K.square(y_pred_loss - y_pred_true))

def getOpti(name,lr):
    if name == 'Adam':
        return Adam(learning_rate=lr,amsgrad=True)
    elif name == 'Nadam':
        return Nadam(learning_rate=lr)
    elif name == 'Adamax':
        return Adamax(learning_rate=lr)
    elif name == 'RMSprop':
        return RMSprop(learning_rate=lr)
    elif name == 'standardadam':
        return 'adam'

def getModel(params,name):
    if(name == '2step'):
        inputs = Input(shape=(2,))
        hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        #hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        output = Dense(1,activation='linear',name='HF')(hidden1)   
    elif (name == 'LF'):
        inputs = Input(shape=(1,))
        hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden3)
        output = Dense(1,activation='linear',name='LF')(hidden4)
        
    elif (name == 'Single'):
        inputs = Input(shape=(1,))
        hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(inputs)
        hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden1)
        hidden3 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden2)
        hidden4 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden3)
        output = Dense(1,activation='linear',name='Single')(hidden2)        
        
    elif (name == 'Hflin'):
        inputs = Input(shape=(2,))
        hiddenlin = Dense(64,activation='linear',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(1,activation='linear',name='HFlin')(hiddenlin)
        
    elif(name == '3step'):
        inputs = Input(shape=(3,))
        hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(1,activation='linear',name='HF')(hidden1)   
        
    elif (name == 'GP'):
        inputs = Input(shape=(1,))
        hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden3)
        GPlayer = Dense(2,activation='linear',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
        outputLF = Dense(1,activation='linear',name='LF')(GPlayer)
        outputHF = Dense(1,activation='linear',name='HF')(GPlayer)   
        output = [outputHF,outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'],params['lr'])
        model.compile(loss=custom_loss,loss_weights=[params['alpha'],1-params['alpha']],optimizer=opti)    
        return model
    
    elif (name == 'Inter'):
        inputs = Input(shape=(1,))
        hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden1)
        outputLF = Dense(1,activation='linear',name='LF')(hidden2)
        outputadd = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden2)
        merge = concatenate([outputLF,outputadd])
        hidden3 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(merge)
        hidden4 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden3)
        #hidden5 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden4)
        #hidden6 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2((1-params['alpha'])*params['l2weight']),kernel_initializer=params['kernel_init'])(hidden5)
  
        #lincorr = Dense(int(params['nodes']),activation='linear',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(outputLF)
        #merge2 = concatenate([hidden3,lincorr])
        outputHF = Dense(1,activation='linear',name='HF')(hidden4)
        output = [outputHF,outputLF]
        model = Model(inputs=inputs, outputs=output)
        opti = getOpti(params['opt'],params['lr'])
        model.compile(loss=custom_loss,loss_weights=[params['alpha'],1-params['alpha']],optimizer=opti)
        return model
 
        
    model = Model(inputs=inputs, outputs=output)
    opti = getOpti(params['opt'],params['lr'])
    model.compile(loss='mse',optimizer=opti,metrics=['mse'])
    return model


def kCrossVal(N,Nepo,x,y,params,name):
    #K.clear_session()
    model = getModel(params,name)
    score = np.zeros([N])
    cv = KFold(N)
    splits = cv.split(x)
    i=0
    for train_index, test_index in splits:
      x_train = x[train_index,:]
      y_train = y[train_index]
      x_val = x[test_index,:]
      y_val = y[test_index]
      #model = getModel(params,name)
      model.fit(x_train,y_train,epochs=Nepo,batch_size=N-1,verbose=0)
      score[i] = np.square(y_val - model.predict(x_val))     
      i = i+1
    return np.mean(score)

def kCrossValSingle(N,Nepo,x,y,params,name):
    score = np.zeros([N])
    cv = KFold(N)
    splits = cv.split(x)
    model = getModel(params,name)
    i=0
    for train_index, test_index in splits:
      x_train = x[train_index]
      y_train = y[train_index]
      x_val = x[test_index]
      y_val = y[test_index]    
      model.fit(x_train,y_train,epochs=Nepo,batch_size=N-1,verbose=0)
      print(y_val)
      print(model.predict(x_val))
      score[i] = np.square(y_val - model.predict(x_val))     
      i = i+1
    return np.mean(score)

def kCrossValGP(Nhf,Nlf,Nepo,xhf,yhf,xlf,ylf,params,name):
    score = np.zeros([Nhf])
    cv = KFold(Nhf)
    splits = cv.split(xhf)
    i=0
    N = Nhf + Nlf
    model = getModel(params,name)
    for train_index, test_index in splits:
      xhf_train = xhf[train_index]
      yhf_train = yhf[train_index]
      xhf_val = xhf[test_index]
      yhf_val = yhf[test_index]
      x_train = np.concatenate((xhf_train,xlf))
      yhf_train = np.concatenate((yhf_train,np.full(Nlf,-10)))
      ylf_train = np.concatenate((np.full(Nhf-1,-10),ylf))    
      model.fit(x_train,[yhf_train,ylf_train],epochs=int(params['epochs'])*Nepo,batch_size=N-1,verbose=0)
      score[i] = np.square(yhf_val - model.predict(xhf_val)[0])
      i = i+1
    return np.mean(score)

def transfBestparam(bestparam,dic):
    for key in bestparam:
        if key in ['kernel_init','opt']:
            bestparam[key] = dic[key][bestparam[key]]
    return 
