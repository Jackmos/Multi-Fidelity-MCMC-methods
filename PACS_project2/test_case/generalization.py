import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
import tensorflow as tf
import arviz

from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from ann_functions import getModel, kCrossVal, transfBestparam
from time import perf_counter
import sys
import os
import warnings

from cuqi.distribution import Uniform, Gaussian,JointDistribution, Beta
from cuqi.sampler import MH,NUTS
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Discrete
#from cuqi.diagnostics import Geweke
import tinyDA as tda
from scipy.stats import multivariate_normal
import arviz as az


class Suppressor:
    # suppress the printed message
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._stdout

class Function:
    def __init__(self, func):
        """
        Initialize the class with a given function.

        Parameters:
        - func: The function for which we want to compute the Jacobian.
        """
        self.func = func

    def compute_jacobian(self, x):
        """
        Compute the Jacobian matrix for the given point x.

        Parameters:
        - x: The point at which to compute the Jacobian.

        Returns:
        - The Jacobian matrix.
        """
        x = np.atleast_1d(x).astype(float)
        n = len(x)
        m = len(self.func(x))

        jacobian = np.zeros((m, n))

        for i in range(n):
            x_copy1 = x.copy()
            x_copy2 = x.copy()

            # Perturb the i-th element
            h = 1e-8
            x_copy1[i] += h
            x_copy2[i] -= h

            # Compute partial derivative using finite difference
            jacobian[:, i] = (self.func(x_copy1) - self.func(x_copy2)) / (2 * h)

        return jacobian

class Neural_network:
    
    def __init__(self,name,params=None,data_train=None,output_train=None,N=1000,n=10,train=True,do_HPO=False,verbose=False):
        K.clear_session()
        self.name=name
        self.params=params
        self.N=N    # epochs
        self.n=n    # batch size    
        self.verbose=verbose
        self.hist=None
        
        if data_train is not None and len(data_train.shape) > 1:
            shape_value = data_train.shape[1]
        else:
            shape_value = 1  

        self.model = getModel(self.params,shape_value,self.name)
        if (do_HPO or params is None):
            if (output_train is None or data_train is None):
                warning_message = "Not enough data given!"
                warnings.warn(warning_message, UserWarning)
            self.params=self.HPO(data_train,output_train)
        if(train):
            self.hist=self.model.fit(data_train,output_train,epochs=self.N,batch_size=self.n,verbose=0) 
    
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


    def performance(self,data_test,output_test):   # to correct in case of inntermediate
        pred=self.prediction(data_test)#[:,0]
        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2= 1 - np.sum(np.square(output_test - pred)) / np.sum(
            np.square(output_test - np.mean(output_test))
        )
        print(f"R^2: {r2}")
        
        return (test_mse,r2)



    def HPO(self,data_train,output_train):

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
        
        def objective(self,params): 

            K.clear_session()
            CVres = kCrossVal(self.n,self.N,data_train,output_train,params,self.name)  # ATTENZIONE!! NEL CASO LF, data_train E output_train dovrebbero essere HF (vedere codice Nlf di esempio), correggere
            return {"loss": CVres, "params": params, "status": STATUS_OK}

        best_params = fmin(fn = self.objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = MAX_EVAL,
                        trials = bayes_trials)
        transfBestparam(best_params, aux_dic)
        print(best_params)
        return best_params
    
    def inverse(self, mean_prior, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True):
    
        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    
        if (N<=burn_in):
                warning_message = "number of steps insufficient, smaller or equal than burn-in"
                warnings.warn(warning_message, UserWarning)        
        
        if(cov_prior is None):
            cov_prior=mean_prior*0.2
        if(cov_likelihood is None):
            cov_likelihood=cov_noise*np.eye(x_real.shape[0])
            
        my_prior = multivariate_normal(mean_prior, cov_prior) # modo per settare uniforme?
        
        if(y_obs is None):
            y_obs=self.prediction(x_real)+np.random.normal(loc=0., scale=0.5,size=y_obs.shape)
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=0.5,size=y_obs.shape)    # 
        
        my_loglike = tda.GaussianLogLike(y_obs+np.random.normal(scale=cov_noise, size=x_real.shape[0]), cov_likelihood)
        my_posterior = tda.Posterior(my_prior, my_loglike, self.prediction)
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.prediction(estimates), self.prediction(x_real))
        
        return estimates

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
            data_train_HF=np.c_[data_train_HF,model.prediction(data_train_HF)]     # caso con LF di seguito non è mai capitato finora
        
    def prediction(self,data_test):
        self.outputs = data_test
        if(len(self.outputs.shape)==1):
            self.outputs=self.outputs.reshape(-1,1)
        for index, _ in enumerate(self.names):
            #print("check")
            self.outputs=np.c_[self.outputs,self.model_list[index].prediction(self.outputs)]
        return self.outputs[:,-1]
            
    #---------------------------------------------
    def inverse(self, mean_prior, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True):
        
        if x_real is not None:
            dim = x_real.shape[0]
        elif y_obs is not None:
            dim=y_obs.shape[0]
        else: 
            warning_message = "No observation nor data given"
            warnings.warn(warning_message, UserWarning)    
        if (N<=burn_in):
                warning_message = "number of steps insufficient, smaller or equal than burn-in"
                warnings.warn(warning_message, UserWarning)        
        
        if(cov_prior is None):
            cov_prior=mean_prior*0.2
        if(cov_likelihood is None):
            cov_likelihood=cov_noise*np.eye(x_real.shape[0])
            
        my_prior = multivariate_normal(mean_prior, cov_prior) # modo per settare uniforme?
        
        if(y_obs is None):
            y_obs=self.prediction(x_real)+np.random.normal(loc=0., scale=0.5,size=y_obs.shape)
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=0.5,size=y_obs.shape)    # 
        
        my_loglike = tda.GaussianLogLike(y_obs+np.random.normal(scale=cov_noise, size=dim), cov_likelihood)
        my_posterior = tda.Posterior(my_prior, my_loglike, self.prediction)
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.prediction(estimates), self.prediction(x_real))
        
        return estimates
    #------------------------------------------     
            
    def get_output(self):
        return self.outputs[-1]
    
    
def MCMC(my_posterior,N, burnin, n=1, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=False):
    
    
    my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive)
    my_chains = tda.sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True)
    idata = tda.to_inference_data(my_chains, burnin=burnin)
    estimates=np.array(az.summary(idata)['mean'])
    print(f"estimated values are {estimates}")
    if (diagnostic is True):
        print(az.summary(idata))
        az.plot_trace(idata)
        print("Autocorrelation...")
        az.plot_autocorr(idata)
        #az.plot_violin(idata)
        
    
    return estimates

def plot_hist(estimates, real_x, output1,output2):   
    values2=estimates
    values1=real_x
    values = np.vstack((values1, values2))

    # Creare categorie in base alla lunghezza di values
    categories = np.arange(1, values.shape[1] + 1)

    # Larghezza delle colonne
    bar_width = 0.35

    # Posizioni delle colonne
    bar_positions = [categories - bar_width/2 + i*bar_width for i in range(values.shape[0])]
    plt.figure()
    # Creazione del plot
    for i in range(values.shape[0]):
        plt.bar(bar_positions[i], values[i, :], width=bar_width)


    plt.ylabel('Value')
    plt.title('Input')
    plt.xticks(categories)
    plt.legend(["Real value", "Estimate"])


    # Mostra il plot
    plt.show()
    
    values2 = output2
    values1=output1

    values = np.vstack((values1, values2))

    # Creazione del plot
    for i in range(values.shape[0]):
        plt.bar(bar_positions[i], values[i, :], width=bar_width)


    plt.ylabel('Value')
    plt.title('Output')
    plt.xticks(categories)
    plt.legend(["Real value", "Estimate"])
    # Mostra il plot
    plt.show()
    return
    
    
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

def getModel(params,num_inputs,name):
    if(name == '2step'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        #hidden2 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(hidden1)
        output = Dense(1,activation='linear',name='HF')(hidden1)   
    elif (name == 'LF'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(inputs)
        hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden1)
        hidden3 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden2)
        hidden4 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'])(hidden3)
        output = Dense(1,activation='linear',name='LF')(hidden4)
        
    elif (name == 'Single'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(inputs)
        hidden2 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden1)
        hidden3 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden2)
        hidden4 = Dense(64,activation='tanh',kernel_initializer=params['kernel_init'],kernel_regularizer=l2(params['l2weight']))(hidden3)
        output = Dense(1,activation='linear',name='Single')(hidden2)        
        
    elif (name == 'Hflin'):
        inputs = Input(shape=(num_inputs,))
        hiddenlin = Dense(64,activation='linear',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(1,activation='linear',name='HFlin')(hiddenlin)
        
    elif(name == '3step'):
        inputs = Input(shape=(num_inputs,))
        hidden1 = Dense(int(params['nodes']),activation='tanh',kernel_regularizer=l2(params['l2weight']),kernel_initializer=params['kernel_init'])(inputs)
        output = Dense(1,activation='linear',name='HF')(hidden1)   
        
    elif (name == 'GP'):
        inputs = Input(shape=(num_inputs,))
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
        inputs = Input(shape=(num_inputs,))
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
 
        
    model = Model(inputs=inputs,  outputs=output)
    opti = getOpti(params['opt'],params['lr'])
    model.compile(loss='mse',optimizer=opti,metrics=['mse'])
    return model


def kCrossVal(N,Nepo,x,y,params,num_inputs,name):
    #K.clear_session()
    model = getModel(params,num_inputs,name)
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

def kCrossValSingle(N,Nepo,x,y,params,num_inputs,name):
    score = np.zeros([N])
    cv = KFold(N)
    splits = cv.split(x)
    model = getModel(params,num_inputs,name)
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

def kCrossValGP(Nhf,Nlf,Nepo,xhf,yhf,xlf,ylf,params,num_inputs,name):
    score = np.zeros([Nhf])
    cv = KFold(Nhf)
    splits = cv.split(xhf)
    i=0
    N = Nhf + Nlf
    model = getModel(params,num_inputs,name)
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
