from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
from keras.models import save_model
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers import Layer
from tensorflow.keras.layers import (
    concatenate,
)  
from keras.regularizers import l2, l1
from sklearn.model_selection import KFold
from keras.optimizers import Adam, Nadam, Adamax, RMSprop
import keras.backend as K
import tensorflow as tf
import keras as kr
import h5py
import sys
import os
import warnings
import copy
from cuqi.distribution import Uniform, Gaussian,JointDistribution
from cuqi.sampler import MH
from cuqi.model import Model as CuqiModel
from cuqi.geometry import Continuous1D, Discrete
import time
import tinyDA as tda
from scipy.stats import multivariate_normal, beta   
import arviz as az

from abc import ABCMeta, abstractstaticmethod, abstractmethod
from functools import reduce

import importlib
from pathlib import Path
import sys

# connect to specific definition
def load_context_functions(context_folder):
    #print(context_folder)
    try:
        sys.path.append('context_folder')
        return True
    except ImportError:
        print(f"folder path {context_folder} not found")
        return False
from module_utils import *


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

        h = 1e-8
        for i in range(n):
            x_copy = x.copy()

            # Perturb the i-th element positively
            x_copy[i] += h
            func_pos = self.func(x_copy)

            # Perturb the i-th element negatively
            x_copy[i] -= 2 * h
            func_neg = self.func(x_copy)

            # Compute partial derivative using central difference
            jacobian[:, i] = (func_pos - func_neg) / (2 * h)

        return jacobian
    
def compute_time(function):
    def wrapper(*args, **kwargs):
        init = time.time()
        res = function(*args, **kwargs)
        end = time.time()
        timespam = end - init
        print(f"The function {function.__name__} took {timespam} seconds.")
        return res
    return wrapper


class Suppressor:
    # suppress the printed message
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._stdout


class INetwork(metaclass=ABCMeta):
    
    @abstractstaticmethod
    def prediction(self):
        return
    
    @abstractstaticmethod
    def performance(self):
        return

    @abstractmethod
    def inverse(self):
        return
    
    @abstractmethod
    def inverse_cuqi(self):
        return
    
    @abstractstaticmethod
    def save(self):
        return

class Neural_Network(INetwork):
    
    def __init__(self,name,params=None,data_train=None,output_train=None,N=1000,n=10,train=True,do_HPO=False,transformations=[],verbose=False):
        K.clear_session()
        self.name=name
        self.params=params
        self.N=N    # epochs
        self.n=n    # batch size    
        self.verbose=verbose
        self.hist=None
        self.data_train=data_train
        # NON VALE LA PENA SALVARSI SOLO LE DIMENSIONI?
        self.output_train=output_train
        self.transformations=transformations
        self.input_shape=1
        self.output_shape=1

        if data_train is not None and len(data_train.shape) > 1:
            self.input_shape = data_train.shape[1]
      
        if output_train is not None and len(output_train.shape) > 1:
            self.output_shape = output_train.shape[1] 

        if (do_HPO or params is None):
            if (output_train is None or data_train is None):
                warning_message = "Not enough data given!"
                warnings.warn(warning_message, UserWarning)
            self.params=self.HPO(data_train,output_train)
        self.model = getModel(self.params,self.input_shape,self.name,self.output_shape)

        if(train):
            self.hist=self.training(data_train,output_train,epoch=self.N,batch=self.n) 
            plt.plot(self.model.history.history['loss'][100:], label='Training Loss')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()
          
    @compute_time
    def training(self,x,y,epoch,batch):
        # DUBBIO : va definita variabile hist?
        self.hist=self.model.fit(x,y,epochs=epoch,batch_size=batch,verbose=0) 
        return self.hist

    def prediction(self,x_test):
        if(self.verbose):
            y_pred=self.model.predict(x_test)#[:,0]             # <-------------  commentato il 6/03
        else:
            with Suppressor():
                y_pred = self.model.predict(x_test)#[:,0]       # <--------------- commentato il 6/03
        return y_pred  
    
    def wrapper_prediction(self,x_test):
        # ricontrolla
        # ---- caso shear cube ----
        # if (x_test.ndim==1 and self.data_train.shape[1]>1 and len(self.transformations)==0):   # e.g. Shear cube case
        #     x_test=x_test.reshape(-1,1).T
        # elif (x_test.ndim==1):
        #     x_test=x_test.reshape(-1,1)
        # -----------
        if (x_test.ndim==1):
            x_test=x_test.reshape(-1,1)
            # ----- sostituito 
        #print(x_test.shape)
        if self.transformations:
            x_final = reduce(lambda acc, trasf: np.hstack([acc, trasf(acc)]), self.transformations, x_test)
        else:
            x_final = x_test
            
        x_final=np.tile(x_final,(self.inputs.shape[0],1))
        #print(x_final.shape)
        #print(np.concatenate((self.inputs,x_final),axis=1).shape)
        rep=self.prediction(np.concatenate((self.inputs,x_final),axis=1)).flatten()
        #print(rep.shape)
        return rep
    
    def save(self,discr="_",place=""):
        print("saving the model ...")
        folder=os.path.join(place,f"{self.name}_model{discr}.h5")
        save_model(self.model,folder)

    
    def performance(self,data_test,output_test):
        pred=self.wrapper_prediction(data_test)#[:,0]
        #print(pred.shape)
        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2= 1 - np.sum(np.square(output_test - pred)) / np.sum(
            np.square(output_test - np.mean(output_test))
        )
        print(f"R^2: {r2}")
        
        return (test_mse,r2)
    
    @compute_time
    def inverse(self, mean_prior, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, cov_search=False,transformations=[], algo="MH"):
        
        self.transformations=transformations
        
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
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
            
            
        my_prior = multivariate_normal(mean_prior, cov_prior) # modo per settare uniforme?
        
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=0.5,size=y_obs.shape)
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)    # 
        
        my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
        my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
    
    
    # CUQI CASE
    @compute_time
    def inverse_cuqi(self, mean_prior, x_real=None,y_obs=None, N=1000, burn_in=500, cov_prior=0.5,sd_noise=0.1,proposal_sd=0.3,adapt=False,scale=0.3,x_init=None,diagnostic=True, number_chains=1, algo="MH"):

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
              
        if (x_init is None):
            x_init=np.random.rand(dim)
        elif isinstance(x_init, (int, float)):
            x_init=x_init*np.ones(dim)

        if algo=="NUTS":
            fun=Function(self.wrapper_prediction)
            A=CuqiModel(forward=self.wrapper_prediction,jacobian=fun.compute_jacobian, range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
            x=Gaussian(mean=mean_prior,cov=cov_prior)
        else:    
            A=CuqiModel(forward=self.wrapper_prediction,range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
            x=Uniform(np.zeros(dim),np.ones(dim)*mean_prior*2)  # prior  # mettere if con diverse prior?
        y=Gaussian(mean=A(x),cov=proposal_sd)
        
        if(y_obs is None):
            y_obs=y(x=x_real).sample()
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=sd_noise,size=y_obs.shape)

        # posterior=JointDistribution(y,x)(y=y_obs)
        
        # sampler=MH(posterior,x0=x_init)   # rendere variabile per altri sampler
        # samples=sampler.sample_adapt(N-burn_in,burn_in)
        
        # if(diagnostic is True):
        #     samples.plot_trace()
        #     mean=samples.mean()
        #     print(f"Mean values= {mean}")
        #     ESS=samples.compute_ess()
        #     print(f"ESS= {ESS}")
        #     samples.plot_autocorrelation()
        #     plot_hist(estimates,x_real, self.wrapper_prediction(mean), self.wrapper_prediction(x_real))
        
        estimates= MCMC_cuqi(y,x,y_obs,N,burn_in,number_chains,diagnostic=diagnostic,algo=algo, adapt=adapt, scale=scale)
        estimates=np.mean(estimates,axis=1)
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
    


    def objective(self,par): 
        #print(self.name)
        K.clear_session()
        CVres = kCrossVal(self.n,self.N,self.data_train,self.output_train,par,self.name,self.input_shape)  # ATTENZIONE!! NEL CASO LF, data_train E output_train dovrebbero essere HF (vedere codice Nlf di esempio), correggere
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
    
        
    
class MultiFidelity(INetwork):
   
    def __init__(self,names,params=None,data_train=None,output_train=None,N=None,n=None,do_HPO=False,verbose=False):
        # names: list of strings, names of the networks (if the discretizations layer that we consider are 2, len(names)==nb_steps)
        # params: list of dictionaries
        # N: list of epochs
        # n: list of batch_sizes
        # data_train: list of inputs
        # output_train: list of outputs 
        self.names=names
        self.Ns=N
        self.ns=n
        self.data_train=data_train
        self.steps=int((len(names)-1)/(len(data_train)-1))+1
        self.model_list = []
        self.outputs = np.empty((0,0))
        self.transformations=[]
        self.input_shape=1
        self.output_shape=1

        if len(data_train)!=len(output_train) or data_train is None or output_train is None or len(data_train)==0 or len(output_train)==0:
            raise ValueError('The data are incoherent or insufficient')

        if data_train is not None and len(data_train[0].shape) > 1:
            self.input_shape = data_train[0].shape[1] 

        if output_train is not None and len(output_train[0].shape) > 1:
            self.output_shape = output_train[0].shape[1] 

        K.clear_session()
        if len(params)<len(self.names):
            diff = len(self.names) - len(params)
            params += [None] * diff
                        
        count=1             
        for index, name in enumerate(names):  # number networks
            # ATTENTION to the fact that the "number of dataset" is denoted by the number of discretizations considered 
        
            #print(data_train[count-1].shape)
            model=NetworkFactory.build_network(name,params=params[index],data_train=data_train[count-1],output_train=output_train[count-1],N=self.Ns[index],n=self.ns[count-1],train=True,do_HPO=do_HPO,verbose=verbose)
            self.model_list.append(model)
            # self.outputs.append(model.prediction(self.outputs)) 
            if (index+1)==(self.steps-1)*(count-1)+1:
                count=count+1
           # print(data_train[count:].shape)
            data_train[count-1:] = [np.c_[matrix, model.prediction(matrix)] for matrix in data_train[count-1:]]
            #for i in range(len(data_train)):
            #    print(data_train[i].shape)

            # data_train[index+1]=np.c_[data_train_HF,model.prediction(data_train_HF)]     # caso con LF di seguito non è mai capitato finora    LINEA VERA
            # debugging purpose
            # indici_piu_vicini = trova_indici_piu_vicini(data_train_LF[:,0], data_train_HF[:,0])
            # print(indici_piu_vicini)
            # data_train_HF=np.c_[data_train_HF,output_train_LF[indici_piu_vicini]] 
            
            #data_train_HF=np.c_[data_train_HF,output_train_HF]     # caso con LF di seguito non è mai capitato finora
    
    def prediction(self,data_test):
        self.outputs = data_test
        if(len(self.outputs.shape)==1):
            self.outputs=self.outputs.reshape(-1,1)
        for index, _ in enumerate(self.names):
            #print("check")
            self.outputs=np.c_[self.outputs,self.model_list[index].prediction(self.outputs)]
        return self.outputs[:,-self.output_shape:]


    def performance(self,data_test,output_test,position=None):
        
        data=copy.copy(data_test)
        if(position is None):
            position=len(self.model_list)
        elif(not isinstance(position, int) or position>len(self.model_list)):
            raise ValueError('the required NN is not existent')
        
        for i in range(position-1):
            data = np.concatenate(
                    (data,  self.model_list[i].wrapper_prediction(data).reshape(-1,1)),axis=1
                ) 
        
        pred=self.model_list[position-1].wrapper_prediction(data)#[:,0]
    
        #print(pred.shape)
        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2= 1 - np.sum(np.square(output_test - pred)) / np.sum(
            np.square(output_test - np.mean(output_test))
        )
        print(f"R^2: {r2}")
        
        return (test_mse,r2)
    
    @compute_time
    def param_inverse(self, mean_prior, x_data, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
            # I should insert the 
        self.transformations=transformation
        self.inputs=x_data
        # rivedere la questione dimensioni (x_real!=y_obs)
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
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
        # prior
        if(mean_prior.shape[0]==1):
            my_prior=beta(1.,1.)
        else:
            my_prior = multivariate_normal(mean_prior, cov_prior) 

        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape) # check
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape) 
            y_obs=y_obs.flatten()
        # likelihood
        my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
        # posterior
        my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
    


    def wrapper_prediction(self,x_test,multi_input=False):
        # ricontrolla, se usata solo per param inv si potra mettere private
    
        #if (x_test.ndim==1 and self.data_train[0].shape[1]>1 and len(self.transformations)==0):   # e.g. Shear cube case
        #    x_test=x_test.reshape(-1,1).T
        if (x_test.ndim==1):
            x_test=x_test.reshape(-1,1)
        #print(x_test.shape)
        x_test=x_test.T # VEDI
        if self.transformations:
            x_final = reduce(lambda acc, trasf: np.hstack([acc, trasf(acc)]), self.transformations, x_test)
        else:
            x_final = x_test
            # mettere messaggio di errore quando inputs non riempiton 
        #print(self.inputs.shape)
        x_final=np.tile(x_final,(self.inputs.shape[0],1))
        #print(x_final.shape)
        #print(np.concatenate((self.inputs,x_final),axis=1).shape)
        rep=self.prediction(np.concatenate((self.inputs,x_final),axis=1)).flatten()
        #print(rep.shape)
        return rep


    @compute_time
    def inverse(self, mean_prior, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, levels=1,diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
        # transformations is a list of lambda functions
        self.transformations=transformation

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
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
        if(mean_prior.shape[0]==1):
            my_prior=beta(1.,1.)
        else:
            my_prior = multivariate_normal(mean_prior, cov_prior) # modo per settare uniforme?
        
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)    # 

        if levels>1:
            my_loglike=[]
            my_posterior=[]
            if(levels>=len(self.model_list)):
                warning_message = "number of levels is exceeding the number of models"
                warnings.warn(warning_message, UserWarning)
            else:    
                for i in range(levels):    
                    my_loglike = [my_loglike, tda.GaussianLogLike(y_obs, cov_likelihood)]
                    my_posterior = [my_posterior, tda.Posterior(my_prior, my_loglike[-1], self.model_list[i].wrapper_prediction)]   # attention, you should put ML model at the end
        else:
            my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
            my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
    
    @compute_time
    def inverse_ML(self, mean_prior, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
        # transformations is a list of lambda functions
        self.transformations=transformation

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
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
        if(mean_prior.shape[0]==1):
            my_prior=beta(1.,1.)
        else:
            my_prior = multivariate_normal(mean_prior, cov_prior) # modo per settare uniforme?
        
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)    # 
        
        my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
        my_posterior=[]

        for i in range(len(self.model_list)):
            my_posterior.append(tda.Posterior(my_prior, my_loglike, self.model_list[i].wrapper_prediction))
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates







    # CUQIPY
    @compute_time
    def inverse_cuqi(self, mean_prior, x_real=None,y_obs=None, N=1000, burn_in=500, cov_prior=0.5, sd_noise=0.1,adapt=False,scale=0.3,proposal_sd=0.3,x_init=None,diagnostic=True, number_chains=1, algo="MH"):

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
              
        if (x_init is None):
            x_init=np.random.rand(dim)
        elif isinstance(x_init, (int, float)):
            x_init=x_init*np.ones(dim)

        if algo=="NUTS":
            fun=Function(self.wrapper_prediction)
            A=CuqiModel(forward=self.wrapper_prediction,jacobian=fun.compute_jacobian, range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
            x=Gaussian(mean=mean_prior,cov=cov_prior)
        else:    
            A=CuqiModel(forward=self.wrapper_prediction,range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
            x=Uniform(np.zeros(dim),np.ones(dim)*mean_prior*2)  # prior  # mettere if con diverse prior?
        y=Gaussian(mean=A(x),cov=proposal_sd)
        
        if(y_obs is None):
            y_obs=y(x=x_real).sample()
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=sd_noise,size=y_obs.shape)

        


        # posterior=JointDistribution(y,x)(y=y_obs)
        
        # sampler=MH(posterior,x0=x_init)   # rendere variabile per altri sampler
        # samples=sampler.sample_adapt(N-burn_in,burn_in)
        
        # if(diagnostic is True):
        #     samples.plot_trace()
        #     mean=samples.mean()
        #     print(f"Mean values= {mean}")
        #     ESS=samples.compute_ess()
        #     print(f"ESS= {ESS}")
        #     samples.plot_autocorrelation()
        #     plot_hist(estimates,x_real, self.wrapper_prediction(mean), self.wrapper_prediction(x_real))
        
        estimates= MCMC_cuqi(y,x,y_obs,N,burn_in,number_chains,diagnostic=diagnostic,algo=algo,adapt=adapt,scale=scale)
        estimates=np.mean(estimates,axis=1)
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
            
            
    def save(self,discr="_",place=""):
        print("Saving ...")
        for index, _ in enumerate(self.model_list):
            folder=os.path.join(place,f"{self.names[index]}_model_{discr}.h5")
            save_model(self.model_list[index].model,folder)
            
    def get_output(self):
        return self.outputs[-1]
    

class Intermediate(INetwork):
    
    def __init__(self,params=None,data_train=None,output_train=None,N=None,n=None,train=True,do_HPO=False,verbose=False):
        # names: list of strings, names of the networks (if the discretizations layer that we consider are 2, len(names)==nb_steps)
        # params: list of dictionaries
        # N: list of epochs
        # n: list of batch_sizes
        # data_train: list of inputs
        # output_train: list of outputs
        self.params=params
        self.name="Inter" 
        self.N=N
        self.n=n
        self.verbose=verbose
        self.hist=None
        self.data_train=data_train
        self.output_train=output_train
        self.transformations=[]
        
        self.input_shape=1
        self.output_shape=1


        if len(data_train)!=2 or len(output_train)!=2:
            raise ValueError('The data are incoherent or insufficient')
                
        # in this way, the dataset should be only made of 1 matrix (np.array)         
        data_train=np.concatenate((data_train[1],data_train[0]),axis=0)
        output_train=np.concatenate((output_train[1],output_train[0]),axis=0)
        
        # number of inputs of the matrix (columns)  (DUBBIO: non si confonde con lista del caso multifidelity?)
        if data_train is not None and len(data_train.shape) > 1:
            self.input_shape=data_train.shape[1]

        if output_train is not None and len(output_train.shape) > 1:
            self.output_shape=output_train.shape[1]

        if (do_HPO or params is None):
            if (output_train is None or data_train is None):
                warning_message = "Not enough data given!"
                warnings.warn(warning_message, UserWarning)
            self.params=self.HPO(data_train,output_train)
        self.model = getModel(self.params,self.input_shape,self.name,self.output_shape)

        if(train):
        
            self.hist=self.model.fit(data_train,output_train,epochs=self.N,batch_size=self.n,verbose=0) 
            plt.plot(self.model.history.history['loss'][100:], label='Training Loss')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()

    @compute_time
    def training(self,x,y,epoch,batch):
        # DUBBIO : va definita variabile hist?
        self.hist=self.model.fit(x,y,epochs=epoch,batch_size=batch,verbose=0) 
        return self.hist
    
    def prediction(self,x_test):
        if len(x_test)!=2:
            raise ValueError("Not enough data given")
        x_test=np.concatenate((x_test[1],x_test[0]),axis=0)

        if(self.verbose):
            y_pred=self.model.predict(x_test)#[:,0]
        else:
            with Suppressor():
                y_pred = self.model.predict(x_test)
                #print(y_pred)
                #y_pred = self.model.predict(x_test)[:,0]
        return y_pred  
    
    def wrapper_prediction(self,x_test):
           # ricontrolla
        
        if (x_test.ndim==1 and self.data_train.shape[1]>1 and len(self.transformations)==0):   # e.g. Shear cube case
            x_test=x_test.reshape(-1,1).T
        elif (x_test.ndim==1):
            x_test=x_test.reshape(-1,1)
        #print(x_test.shape)
        if self.transformations:
            x_final = reduce(lambda acc, trasf: np.hstack([acc, trasf(acc)]), self.transformations, x_test)
        else:
            x_final = x_test
        return self.prediction(x_final) 
    
    def save(self,discr="_",place=""):
        print("saving the model ...")
        folder=os.path.join(place,f"Inter_model{discr}.h5")
        save_model(self.model,folder)

    def performance(self,data_test,output_test):
        
        data_test=np.concatenate((data_test[1],data_test[0]),axis=0)
        output_test=np.concatenate((output_test[1],output_test[0]),axis=0)
        
        pred=self.prediction(data_test)#[:,0]
        #print(pred.shape)
        test_mse = np.mean(np.square(output_test - pred))
        print(f"Test MSE: {test_mse}")

        r2= 1 - np.sum(np.square(output_test - pred)) / np.sum(
            np.square(output_test - np.mean(output_test))
        )
        print(f"R^2: {r2}")
        
        return (test_mse,r2)
    
    
    # TO BE MODIFIED!!
    @compute_time
    def inverse(self, mean_prior, cov_prior=None, cov_noise=0.1, cov_likelihood=None, y_obs=None, x_real=None, number_chains=1, N=1000, burn_in=500, diagnostic=True,rwmh_cov=None,rmwh_scaling=0.1,rwmh_adaptive=True, algo="MH",transformation=[]):
        # transformations is a list of lambda functions
        self.transformations=transformation

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
            cov_likelihood=cov_noise**2*np.eye(x_real.shape[0])
        
        if(mean_prior.shape[0]==1):
            my_prior=beta(1.,1.)
        else:
            my_prior = multivariate_normal(mean_prior, cov_prior) # modo per settare uniforme?
        
        if(y_obs is None):
            y_obs=self.wrapper_prediction(x_real)+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=cov_noise,size=y_obs.shape)    # 
        
        my_loglike = tda.GaussianLogLike(y_obs, cov_likelihood)
        my_posterior = tda.Posterior(my_prior, my_loglike, self.wrapper_prediction)
        
        print(f"real values are {x_real}")
        
        if(rwmh_cov is None):
            rwmh_cov = np.eye(len(x_real))
            
        estimates=MCMC(my_posterior,N,burn_in,number_chains,diagnostic=diagnostic,rwmh_cov=rwmh_cov,rmwh_scaling=rmwh_scaling,rwmh_adaptive=rwmh_adaptive,algo=algo,dim=dim)
        
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
    
    @compute_time
    def inverse_cuqi(self, mean_prior, x_real=None,y_obs=None, N=1000, burn_in=500, cov_prior=0.5,sd_noise=0.1,adapt=False,scale=0.3,proposal_sd=0.3,x_init=None,diagnostic=True, number_chains=1, algo="MH"):

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
              
        if (x_init is None):
            x_init=np.random.rand(dim)
        elif isinstance(x_init, (int, float)):
            x_init=x_init*np.ones(dim)

        if algo=="NUTS":
            fun=Function(self.wrapper_prediction)
            A=CuqiModel(forward=self.wrapper_prediction,jacobian=fun.compute_jacobian, range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
            x=Gaussian(mean=mean_prior,cov=cov_prior)
        else:    
            A=CuqiModel(forward=self.wrapper_prediction,range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))
            x=Uniform(np.zeros(dim),np.ones(dim)*mean_prior*2)  # prior  # mettere if con diverse prior?

        y=Gaussian(mean=A(x),cov=proposal_sd)
        
        if(y_obs is None):
            y_obs=y(x=x_real).sample()
        else:
            y_obs=y_obs+np.random.normal(loc=0., scale=sd_noise,size=y_obs.shape)

        


        # posterior=JointDistribution(y,x)(y=y_obs)
        
        # sampler=MH(posterior,x0=x_init)   # rendere variabile per altri sampler
        # samples=sampler.sample_adapt(N-burn_in,burn_in)
        
        # if(diagnostic is True):
        #     samples.plot_trace()
        #     mean=samples.mean()
        #     print(f"Mean values= {mean}")
        #     ESS=samples.compute_ess()
        #     print(f"ESS= {ESS}")
        #     samples.plot_autocorrelation()
        #     plot_hist(estimates,x_real, self.wrapper_prediction(mean), self.wrapper_prediction(x_real))
        
        estimates= MCMC_cuqi(y,x,y_obs,N,burn_in,number_chains,diagnostic=diagnostic,algo=algo, adapt=adapt, scale=scale)
        estimates=np.mean(estimates,axis=1)
        if diagnostic is True:
            plot_hist(estimates,x_real, self.wrapper_prediction(estimates), self.wrapper_prediction(x_real))
        
        return estimates
            
    def objective(self,par): 
        #print(self.name)
        K.clear_session()
        CVres = kCrossVal(self.n,self.N,self.data_train,self.output_train,par,self.name,self.input_shape)  # ATTENZIONE!! NEL CASO LF, data_train E output_train dovrebbero essere HF (vedere codice Nlf di esempio), correggere
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
    


    
class NetworkFactory:
    # CHIAMA ANCHE DA MF
    @staticmethod
    def build_network(network_type,names=[],params=None,data_train=None,output_train=None,N=1000,n=10,train=True,do_HPO=False,verbose=False):    
        
        if (network_type=="LF" or network_type=="HF" or network_type=="Hflin" or network_type=="Hfper" ):
            return Neural_Network(network_type,params,data_train,output_train,N,n,train,do_HPO,verbose)
        
        if (network_type[:-4].isdigit() and network_type.endswith("step") ):#and len(names)==int(network_type[:-4])):
            # in this case, data_train, output_train are lists of matrixes containing the data
            return MultiFidelity(names,params,data_train,output_train,N,n,do_HPO,verbose)
        
        if (network_type=="Inter"):
            return  Intermediate(params,data_train,output_train,N,n,train,do_HPO,verbose)
        
        print("Invalid Network")
        return -1

def add_noise(noise_std_data, noise_sta_output, data, output):
    output_flag=output
    data_flag=data
    for std1,std2 in zip(noise_std_data,noise_sta_output):
        noise_1 = np.random.normal(0, std1, output.shape)     
        noise_2 = np.random.normal(0, std2, data.shape)
        temp1=output+noise_1
        temp2=data+noise_2
        output_flag=np.concatenate((output_flag,temp1),axis=0)
        #print(data_flag.shape)
        #print(temp2.shape)
        data_flag=np.concatenate((data_flag,temp2))
    return (output_flag,data_flag)


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



def kCrossVal(N,Nepo,x,y,params,name,input_shape,p=1):
    #K.clear_session()
    model = getModel(params,input_shape,name,y.shape[1]) # check che y.shape[1] sia corretto 
    Nfold = int(N/p)
    score = np.zeros([Nfold])
    cv = KFold(n_splits=Nfold,shuffle=True)
    splits = cv.split(x)
    i=0
    for train_index, test_index in splits:
      x_train = x[train_index,:]
      y_train = y[train_index]
      x_val = x[test_index,:]
      y_val = y[test_index]
      #model = getModel(params,name)
      model.fit(x_train,y_train,epochs=Nepo,batch_size=N-p,verbose=0)
      score[i] = np.mean(np.square(y_val - model.predict(x_val)[:,0]))
      i = i+1
    return np.mean(score)

def kCrossValSingle(N, Nepo, x, y, params, name,input_shape):
    score = np.zeros([N])
    cv = KFold(N)
    splits = cv.split(x)
    model = getModel(params,input_shape, name,y.shape[1]) # check che y.shape[1] sia corretto 
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


def kCrossValGP(Nhf,Nlf,Nepo,xhf,yhf,xlf,ylf,params,name,input_shape,p=1):
    Nfolds = int(Nhf/p)
    score = np.zeros([Nfolds])
    cv = KFold(n_splits = Nfolds, shuffle = True)
    splits = cv.split(xhf)
    i=0
    N = Nhf + Nlf

    model = getModel(params,input_shape,name,yhf.shape[1]) # check che y.shape[1] sia corretto 
    for train_index, test_index in splits:
      xhf_train = xhf[train_index,:]
      yhf_train = yhf[train_index]
      xhf_val = xhf[test_index,:]
      yhf_val = yhf[test_index]
      x_train = np.concatenate((xhf_train,xlf))
      yhf_train = np.concatenate((yhf_train,np.full(Nlf,-10)))
      ylf_train = np.concatenate((np.full(Nhf-p,-10),ylf))
      model.fit(x_train,[yhf_train,ylf_train],epochs=int(params['epochs'])*Nepo,batch_size=N-p,verbose=0)
      score[i] = np.mean(np.square(yhf_val - model.predict(xhf_val)[0][:,0]))
      i = i+1
    return np.mean(score)




def transfBestparam(bestparam, dic):
    # trasforma kernel e opt da parametri numerici (indicatori booleani) a valore stringa
    for key in bestparam:
        if key in ["kernel_init", "opt"]:
            bestparam[key] = dic[key][bestparam[key]]
    return


