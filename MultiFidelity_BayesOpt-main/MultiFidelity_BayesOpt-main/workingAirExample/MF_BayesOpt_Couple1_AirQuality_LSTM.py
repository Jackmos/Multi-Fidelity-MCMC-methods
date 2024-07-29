#!C:\Users\Lips\Anaconda3\envs\310
#%%
"""
Created on Fri Jul 21 12:50:49 2023

@author: Paolo Conti, Johannes Lips

Inner loop for combining ANN Multi-Fidelity Regression with Bayesian Optimization in MATLAB
MATLAB provides variables defined directly "above" the python script, therefore introduce flag MATLAB. 
If true then the code is used together with MATLAB for BayesOpt, when False all variables are defined here

"""

if 'externalInput' in globals():
    MATLAB = True
else:
    MATLAB = False
    
    
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

from progressive_network import MultifidelityNetwork, sliding_windows
import pandas as pd
#%%
#######################     CONFIGURATIONS     ##########################
seed = 0

#Choose the benchmark case
case = 'PM2.5'
example = './AirQuality' + case


scaling = False
train = True
validation = False
save = False
original_order = True

Nhf = 64
# %%
#######################     DATASET     ##########################
seed = 0
#Load the AirQuality data s
df = pd.read_csv('dfTrain_AirQuality.csv', index_col=0)[:11995] #cut off last data for which not everything is available
df.index = pd.to_datetime(df.index)

#Extract array values of grimm_PM10 data
lowfid0 = df.index.hour.values*60 + df.index.minute.values
lowfid1 = df['B03_RH HYT'].values  #external humidity of B03 sensor
lowfid2 = df['B03_RH OPC'].values  #internal humidity of B03 sensor
lowfid3 = df['B03_PM2.5'].values    #PM10 of B03 sensor
lowfid4 = df['SDS011_PM2.5'].values #PM10 of SDS011 sensor
lowfid5 = df['B03_Temp OPC'].values #
lowfid6 = df['B03_Temp HYT'].values #
lowfid7 = df['B03_Bin0'].values #
#[0, 3, 2, 6, 7]
highfid = df['grimm_PM2.5'].values   #PM10 of grimm sensor

assert len(lowfid1) == len(lowfid2) == len(lowfid3) == len(lowfid4) == len(highfid)

if not MATLAB:
    #Plot data
    plt.figure()
    plt.plot(df.index, lowfid1, label = 'B03_RH HYT')
    plt.plot(df.index, lowfid2, label = 'B03_RH OPC')
    plt.plot(df.index, lowfid3, label = 'B03_PM2.5')
    plt.plot(df.index, lowfid4, label = 'SDS011_PM2.5')
    plt.plot(df.index, highfid, label = 'grimm_PM2.5')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


#Divide in training, validation and test set
#Training data are up to June 2 at midnight
train_idx = df.index.get_loc('2020-05-31 00:00:00')
#train_idx = df.index.get_loc('2020-06-02 00:00:00')
time_train = df.index[:train_idx]
lowfid0_train = lowfid0[:train_idx]
lowfid1_train = lowfid1[:train_idx]
lowfid2_train = lowfid2[:train_idx]
lowfid3_train = lowfid3[:train_idx]
lowfid4_train = lowfid4[:train_idx]
lowfid5_train = lowfid5[:train_idx]
lowfid6_train = lowfid6[:train_idx]
lowfid7_train = lowfid7[:train_idx]
highfid_train = highfid[:train_idx]

#Validation data are from June 2 at midnight to June 3 at midnight
val_idx = df.index.get_loc('2020-06-01 00:00:00')
#val_idx = df.index.get_loc('2020-06-03 00:00:00')
time_val = df.index[train_idx:val_idx]
lowfid0_val = lowfid0[train_idx:val_idx]
lowfid1_val = lowfid1[train_idx:val_idx]
lowfid2_val = lowfid2[train_idx:val_idx]
lowfid3_val = lowfid3[train_idx:val_idx]
lowfid4_val = lowfid4[train_idx:val_idx]
lowfid5_val = lowfid5[train_idx:val_idx]
lowfid6_val = lowfid6[train_idx:val_idx]
lowfid7_val = lowfid7[train_idx:val_idx]
highfid_val = highfid[train_idx:val_idx]

#Test data are from June 3 at midnight to June 6 at midnight
test_idx = df.index.get_loc('2020-06-02 00:00:00')
time_test = df.index[val_idx:test_idx]
lowfid0_test = lowfid0[val_idx:test_idx]
lowfid1_test = lowfid1[val_idx:test_idx]
lowfid2_test = lowfid2[val_idx:test_idx]
lowfid3_test = lowfid3[val_idx:test_idx]
lowfid4_test = lowfid4[val_idx:test_idx]
lowfid5_test = lowfid5[val_idx:test_idx]
lowfid6_test = lowfid6[val_idx:test_idx]
lowfid7_test = lowfid7[val_idx:test_idx]
highfid_test = highfid[val_idx:test_idx]

#Rescale data with respect to their maximum value
if scaling:
    scale_lf0 = np.max(lowfid0_train)
    scale_lf1 = np.max(lowfid1_train)
    scale_lf2 = np.max(lowfid2_train)
    scale_lf3 = np.max(lowfid3_train)
    scale_lf4 = np.max(lowfid4_train)
    scale_lf5 = np.max(lowfid5_train)
    scale_lf6 = np.max(lowfid6_train)
    scale_lf7 = np.max(lowfid7_train)
    scale_hf =  50#np.max(highfid_train)
else:
    scale_lf0 = 60*24 #scaled wrt to the total number of minutes in a day
    scale_lf1 = 1
    scale_lf2 = 1
    scale_lf3 = 1
    scale_lf4 = 1
    scale_lf5 = 3
    scale_lf6 = 3
    scale_lf7 = 1000
    scale_hf = 1

lowfid0_train = lowfid0_train / scale_lf0
lowfid1_train = lowfid1_train / scale_lf1
lowfid2_train = lowfid2_train / scale_lf2
lowfid3_train = lowfid3_train / scale_lf3
lowfid4_train = lowfid4_train / scale_lf4
lowfid5_train = lowfid5_train / scale_lf5
lowfid6_train = lowfid6_train / scale_lf6
lowfid7_train = lowfid7_train / scale_lf7
highfid_train = highfid_train / scale_hf

scale = scale_hf

if not MATLAB:
    #Plot scaled training data
    plt.figure()
    plt.plot(time_train, lowfid0_train, label = '# minutes')
    plt.plot(time_train, lowfid1_train, label = 'B03_RH HYT')
    plt.plot(time_train, lowfid2_train, label = 'B03_RH OPC')
    plt.plot(time_train, lowfid3_train, label = 'B03_PM2.5')
    plt.plot(time_train, lowfid4_train, label = 'SDS011_PM2.5')
    plt.plot(time_train, lowfid5_train, label = '5')
    plt.plot(time_train, lowfid6_train, label = '6')
    plt.plot(time_train, lowfid7_train, label = '7')
    plt.plot(time_train, highfid_train, label = 'grimm_PM2.5')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

#%%
#######################     RESHAPE FOR LSTM     ##########################
seq_length = 100
freq = 10

lowfid0_train = lowfid0_train.reshape(1,-1,1)
lowfid1_train = lowfid1_train.reshape(1,-1,1)
lowfid2_train = lowfid2_train.reshape(1,-1,1)
lowfid3_train = lowfid3_train.reshape(1,-1,1)
lowfid4_train = lowfid4_train.reshape(1,-1,1)
lowfid5_train = lowfid5_train.reshape(1,-1,1)
lowfid6_train = lowfid6_train.reshape(1,-1,1)
lowfid7_train = lowfid7_train.reshape(1,-1,1)
highfid_train = highfid_train.reshape(1,-1,1)

lowfid0_val = lowfid0_val.reshape(1,-1,1)
lowfid1_val = lowfid1_val.reshape(1,-1,1)
lowfid2_val = lowfid2_val.reshape(1,-1,1)
lowfid3_val = lowfid3_val.reshape(1,-1,1)
lowfid4_val = lowfid4_val.reshape(1,-1,1)
lowfid5_val = lowfid5_val.reshape(1,-1,1)
lowfid6_val = lowfid6_val.reshape(1,-1,1)
lowfid7_val = lowfid7_val.reshape(1,-1,1)
highfid_val = highfid_val.reshape(1,-1,1)

lowfid0_test = lowfid0_test.reshape(1,-1,1)
lowfid1_test = lowfid1_test.reshape(1,-1,1)
lowfid2_test = lowfid2_test.reshape(1,-1,1)
lowfid3_test = lowfid3_test.reshape(1,-1,1)
lowfid4_test = lowfid4_test.reshape(1,-1,1)
lowfid5_test = lowfid5_test.reshape(1,-1,1)
lowfid6_test = lowfid6_test.reshape(1,-1,1)
lowfid7_test = lowfid7_test.reshape(1,-1,1)
highfid_test = highfid_test.reshape(1,-1,1)

### Reshape training in subsequences
lowfid0_train_seq = sliding_windows(lowfid0_train, seq_length, freq)
lowfid1_train_seq = sliding_windows(lowfid1_train, seq_length, freq)
lowfid2_train_seq = sliding_windows(lowfid2_train, seq_length, freq)
lowfid3_train_seq = sliding_windows(lowfid3_train, seq_length, freq)
lowfid4_train_seq = sliding_windows(lowfid4_train, seq_length, freq)
lowfid5_train_seq = sliding_windows(lowfid5_train, seq_length, freq)
lowfid6_train_seq = sliding_windows(lowfid6_train, seq_length, freq)
lowfid7_train_seq = sliding_windows(lowfid7_train, seq_length, freq)
highfid_train_seq = sliding_windows(highfid_train, seq_length, freq)

yhf = highfid_train.flatten()

#%% Define fidelity order:
    
original_fidelity_order_train_seq = [lowfid0_train_seq, lowfid1_train_seq, lowfid2_train_seq, lowfid3_train_seq, lowfid4_train_seq, lowfid5_train_seq, lowfid6_train_seq, lowfid7_train_seq]
original_fidelity_order_train = [lowfid0_train, lowfid1_train, lowfid2_train, lowfid3_train, lowfid4_train, lowfid5_train, lowfid6_train, lowfid7_train]
original_fidelity_order_val = [lowfid0_val, lowfid1_val, lowfid2_val, lowfid3_val, lowfid4_val, lowfid5_val, lowfid6_val, lowfid7_val]
original_fidelity_order_test = [lowfid0_test, lowfid1_test, lowfid2_test, lowfid3_test, lowfid4_test, lowfid5_test, lowfid6_test, lowfid7_test]
original_fidelity_labels = [0, 1, 2, 3, 4, 5, 6, 7]


#Do a permutation of the labels and sort fidelity orders accordingly
if not MATLAB:
    #Select a random numer of models 
    # n_models = np.random.randint(1,5)
    # np.random.seed(30)
    # fidelity_order_labels = np.random.permutation(original_fidelity_labels)[:n_models]
    # print(fidelity_order_labels)

    #fidelity_order_labels = [3,0,2,1][:n_models]

    ##Original order:
    fidelity_order_labels = [0, 3, 2, 6, 7]
else:
    fidelity_order_labels = np.array(fidelity_order_labels, dtype = int).flatten()
#fidelity_order = original_fidelity_order
#print(type(fidelity_order_labels))

n_models = len(fidelity_order_labels) #redundant if not MATLAB, else necessary to define variable   
fidelity_order_train_seq = [original_fidelity_order_train_seq[i] for i in fidelity_order_labels]
fidelity_order_train = [original_fidelity_order_train[i] for i in fidelity_order_labels]
fidelity_order_val = [original_fidelity_order_val[i] for i in fidelity_order_labels]
fidelity_order_test = [original_fidelity_order_test[i] for i in fidelity_order_labels]

print('Number of models: ', n_models, '\nFidelity order', fidelity_order_labels)
#%%

y = highfid_train_seq
y_val = highfid_val.flatten()
y_test = highfid_test.flatten()


#Hyperparameters
n_sim = 1
Nepo = 200
patience = 50
#if case == 'disco':
params = {'lr' :5e-3, 
                'kernel_init' : 'glorot_uniform', 
                'opt' : 'Adam', 
                'activation' : 'tanh',
                'layers_encoder' : [],
                'layers_decoder' : [16, 16], 
                'model_type_encoder' : 'LSTM',
                'model_type_decoder' : 'LSTM',
                'l2weight' : 1e-4, 
                'Nepo' : Nepo,
                'concatenate' : False}
if validation:
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
else:
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
params['model_type'] = model_type = 'Dense'

input_dim = 1
output_dim = 1
latent_dim = 1

full_input_data = []
full_input_data_train = []
full_input_data_val = []
full_input_data_test = []
full_pred_sim = []
full_pred_sim_train = []
full_pred_sim_val = []
full_model_list = []
all_goodness_train = [] # list of all goodness levels reached after adding different fidelity stages.
all_goodness_val = [] # list of all goodness levels reached after adding different fidelity stages.

for n_model in range(n_models):
    print('######### model ' + str(n_model+1) + ' of ' + str(n_models)+ ' #########')

    pred_sim = []
    pred_sim_val = []
    pred_sim_train = []
    model_list = []

    #Preprocess data
    x = fidelity_order_train_seq[n_model]
    x_train = fidelity_order_train[n_model]
    x_val = fidelity_order_val[n_model]
    x_test = fidelity_order_test[n_model]

    full_input_data.append(x)
    full_input_data_train.append(x_train)
    full_input_data_val.append(x_val)
    full_input_data_test.append(x_test)
    
    
    for i in range(n_sim):
        print('sim ' + str(i+1) + ' of ' + str(n_sim))
        prev_models = [full_model_list[j][i] for j in range(n_model)]
        prev_inputs = [full_input_data[j].reshape(-1,1) for j in range(n_model)]
        model = MultifidelityNetwork(params, input_dim = input_dim, latent_dim = latent_dim, output_dim = output_dim, prev_models = prev_models, prev_inputs = prev_inputs)
        name = example + '/models/model' + str(n_model) + '_traj_' + model_type + str(i) + '_HPO'
        if scaling:
                name = name + '_scaled'   

        if train:
            model.autoencoder.compile(loss='mse',optimizer=params['opt'],metrics=['mse'])
            tf.random.set_seed(seed + i)
            np.random.seed(seed + i)
            input_data = [full_input_data[j] for j in range(n_model+1)]
            input_data_train = [full_input_data_train[j] for j in range(n_model+1)]
            input_data_val = [full_input_data_val[j] for j in range(n_model+1)]
            if validation:
                hist = model.autoencoder.fit(input_data,y,epochs=Nepo,batch_size=Nhf,verbose=0,callbacks=[early_stopping],validation_data = (input_data_val, y_val))
            else:
                hist = model.autoencoder.fit(input_data,y,epochs=Nepo,batch_size=Nhf,verbose=1,callbacks=[early_stopping])

        else:
            model.load_weights(name)
        model_list.append(model)
        if save:
            model.save_weights(name)

        
        #Predict
        input_data_test = [full_input_data_test[j] for j in range(n_model+1)]
        y_pred_test = model.predict(input_data_test).numpy().flatten()
        pred_sim.append(y_pred_test)
        y_pred_train = model.predict(input_data_train).numpy().flatten()
        pred_sim_train.append(y_pred_train)
        y_pred_val = model.predict(input_data_val).numpy().flatten()
        pred_sim_val.append(y_pred_val)

        #plt.plot(xhf_test, y_pred_test, label = 'HF', color = 'blue',  linewidth = 0.3)

    if not MATLAB:
        mean_sim = np.mean(np.array(pred_sim), axis = 0)
        std_sim = np.std(np.array(pred_sim), axis = 0)
        fig = plt.figure(figsize=(6,4))
        plt.plot(time_test, y_test.flatten() * scale, label = 'HF', color = 'red')
        plt.plot(time_test, mean_sim* scale, label = 'mean', color = 'blue')
        plt.fill_between(time_test, (mean_sim - std_sim)* scale, (mean_sim + std_sim)* scale, color='lightblue', alpha=0.5, label='mean $\pm$ std')

        plt.plot(time_train,yhf * scale, 'r--')
        plt.plot(time_train, np.mean(np.array(pred_sim_train), axis = 0).flatten()*scale, 'b--')
        
        plt.xlabel('time')
        plt.ylim([0, 20])
        plt.legend()
        plt.show()

    all_goodness_train.append(np.mean(np.sum((yhf.flatten() - np.array(pred_sim_train))**2, axis = 1)))
    all_goodness_val.append(np.mean(np.sum((y_val.flatten() - np.array(pred_sim_val))**2, axis = 1)))
    full_pred_sim.append(pred_sim)
    full_pred_sim_train.append(pred_sim_train)
    full_pred_sim_val.append(pred_sim_val)
    full_model_list.append(model_list)


try:
    goodness = np.mean(np.sum((yhf.flatten() - np.array(full_pred_sim_train[-1]))**2, axis = 1))
except:
    goodness = 1e3 #if somehow landed in invalid point return very large objective

if not MATLAB:
    mean_sim = np.mean(np.array(full_pred_sim[-1]), axis = 0)
    std_sim = np.std(np.array(full_pred_sim[-1]), axis = 0)
    fig = plt.figure(figsize=(6,4))
    plt.plot(time_test, y_test * scale, label = 'HF', color = 'red')
    plt.plot(time_test, mean_sim * scale, label = 'mean', color = 'blue')
    plt.fill_between(time_test, (mean_sim - std_sim) * scale, (mean_sim + std_sim) * scale, color='lightblue', alpha=0.5, label='mean $\pm$ std')
    plt.plot(time_train,yhf * scale, 'r--')
    plt.plot(time_train, np.mean(np.array(pred_sim_train), axis = 0).flatten()*scale, 'b--')
    plt.ylim([0, 20])
    plt.xlabel('time')
    plt.legend()
    plt.show()
# %%
