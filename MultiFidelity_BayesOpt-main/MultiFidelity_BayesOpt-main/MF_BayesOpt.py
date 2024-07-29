
#%%
"""
Created on Fri Jul 21 12:50:49 2023

@author: paolo
"""

import tensorflow.keras.backend as K

from tensorflow.keras.regularizers import l2
from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
from time import perf_counter

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Add, Lambda, BatchNormalization
from tensorflow.keras.layers import concatenate
import tensorflow as tf
#if not GPU, just comment the following line
#tf.config.list_physical_devices('GPU')

import sys
sys.path.insert(0, './progressive_network')

from progressive_network import MultifidelityNetwork
# %%

#######################     CONFIGURATIONS     ##########################
seed = 0

#Choose the benchmark case
case = 'discontinuous'#'discontinuous' #'linear'
example = './1D_Benchmark_' + case

scaling = True
train = True
save = False
original_order = True

if case == 'linear':
    Nhf = 5
elif case == 'discontinuous':
    Nhf = 8
N_test = 100


#%% Model definitions

#1D Benchmark: Linear correlation
if case == 'linear':
    lowfid0 = lambda x: 30*x -10
    lowfid1 = lambda x: np.sin(12*x-4)
    lowfid2 = lambda x: (6*x -2.)**2 * lowfid1(x)
    lowfid3 = lambda x: np.cos(7*x) + 3*np.sin(30*x)
    highfid = lambda x: 0.5*lowfid2(x) + 10*(x-0.5) + 5. 

if case == 'discontinuous':
    #1D Benchmark: Discontinuous functions
    lowfid0 = lambda x: 30*x -10
    lowfid1 = lambda x: ((6.*x-2.)*np.sin(12.*x-4))*(x<0.5) + (3+0.5*(6.*x-2)*np.sin(12.*x-4))*(x>0.5)
    lowfid2 = lambda x: (0.5*(6.*x-2.)**2*np.sin(12.*x-4) + 10*(x-0.5) -5)*(x<0.5) + (3+0.5*(6.*x-2)**2*np.sin(12.*x-4)+10*(x-0.5)-5)*(x>0.5)
    lowfid3 = lambda x: np.cos(7*x) + 3*np.sin(30*x)
    highfid = lambda x: (2*lowfid2(x)- 20*x+20)*(x<0.5) + (4+2*lowfid2(x)- 20*x+20)*(x>0.5) 



#%% Create dataset 

#Input
xhf = np.linspace(0,1,Nhf)
xhf_test = np.linspace(0,1,N_test)

#Plot
plt.figure()
plt.plot(xhf_test, lowfid0(xhf_test), 'k--', label = "$f_{LF^{(0)}}$")
plt.plot(xhf_test, lowfid1(xhf_test), 'g--', label = "$f_{LF^{(1)}}$")
plt.plot(xhf_test, lowfid2(xhf_test), 'b--', label = "$f_{LF^{(2)}}$")
plt.plot(xhf_test, lowfid3(xhf_test), 'r--', label = "$f_{LF^{(3)}}$")
plt.plot(xhf_test, highfid(xhf_test), 'r-', label = "$f_{HF}$")
plt.xlabel('x')
plt.legend(ncol = 2)
plt.show()


#%% Define fidelity order:
original_fidelity_order = [lowfid0, lowfid1, lowfid2, lowfid3]
original_fidelity_labels = [0, 1, 2, 3]

#Do a permutation of the labels and sort fidelity orders accordingly
#np.random.seed(30)
#Select a random numer of models from 1 to 4
n_models = np.random.randint(1,5)
fidelity_order = np.random.permutation(original_fidelity_labels)[:n_models]
print('Number of models: ', n_models, '\nFidelity order', fidelity_order)

fidelity_order = [original_fidelity_order[i] for i in fidelity_order]
#fidelity_order = original_fidelity_order

#%%

#Scaling
if scaling:
    if case == 'discontinuous':
        scale = 27.
    elif case == 'linear':
        scale = 18.
    scale_param = 1.
    scale_time = 1.
else:
    scale =  1.
    scale_param = 1.
    scale_time = 1.

y = highfid(xhf) / scale
y = y.reshape(-1,1)

y_test = highfid(xhf_test) / scale
y_test = y_test.reshape(-1,1)

#Hyperparameters
n_sim = 3
Nepo = 2500
patience = 50
#if case == 'disco':
params = {'lr' :1e-3, 
                'kernel_init' : 'glorot_uniform', 
                'opt' : 'Adam', 
                'activation' : 'tanh',
                'layers_encoder' : [],
                'layers_decoder' : [8, 8, 8], 
                'l2weight' : 1e-4, 
                'Nepo' : Nepo,
                'concatenate' : False}

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)

params['model_type'] = model_type = 'Dense'

input_dim = 1
output_dim = 1
latent_dim = 1


full_input_data = []
full_input_data_test = []
full_pred_sim = []
full_model_list = []

for n_model in range(n_models):
    print('######### model ' + str(n_model+1) + ' of ' + str(n_models)+ ' #########')

    pred_sim = []
    model_list = []

    #Preprocess data
    x = fidelity_order[n_model](xhf) / scale
    x = x.reshape(-1,1)

    x_test = fidelity_order[n_model](xhf_test) / scale
    x_test = x_test.reshape(-1,1)

    full_input_data.append(x)
    full_input_data_test.append(x_test)
    
    for i in range(n_sim):
        print('sim ' + str(i+1) + ' of ' + str(n_sim))
        prev_models = [full_model_list[j][i] for j in range(n_model)]
        prev_inputs = [full_input_data[j] for j in range(n_model)]
        model = MultifidelityNetwork(params, input_dim = input_dim, latent_dim = latent_dim, output_dim = output_dim, prev_models = prev_models, prev_inputs = prev_inputs)
        name = example + '/models/model' + str(n_model) + '_traj_' + model_type + str(i) + '_HPO'
        if scaling:
                name = name + '_scaled'   

        if train:
            model.autoencoder.compile(loss='mse',optimizer=params['opt'],metrics=['mse'])
            tf.random.set_seed(seed + i)
            np.random.seed(seed + i)
            input_data = [full_input_data[j] for j in range(n_model+1)]
            hist = model.autoencoder.fit(input_data,y,epochs=Nepo,batch_size=Nhf,verbose=0,callbacks=[early_stopping])
        else:
            model.load_weights(name)
        model_list.append(model)
        if save:
            model.save_weights(name)

        
        #Predict
        input_data_test = [full_input_data_test[j] for j in range(n_model+1)]
        y_pred_test = model.predict(input_data_test)[:,0] * scale
        pred_sim.append(y_pred_test)

        #plt.plot(xhf_test, y_pred1_test, label = 'HF', color = 'blue',  linewidth = 0.3)

    fig = plt.figure(figsize=(6,4))

    mean_sim = np.mean(np.array(pred_sim), axis = 0)
    std_sim = np.std(np.array(pred_sim), axis = 0)
    plt.plot(xhf_test, y_test * scale, label = 'HF', color = 'red')
    plt.plot(xhf_test, mean_sim, label = 'mean', color = 'blue')
    plt.fill_between(xhf_test, mean_sim - std_sim, mean_sim + std_sim, color='lightblue', alpha=0.5, label='mean $\pm$ std')
    plt.plot(xhf,y * scale, 'r*')
    plt.xlabel('x')
    plt.legend()
    plt.show()

    full_pred_sim.append(pred_sim)
    full_model_list.append(model_list)
