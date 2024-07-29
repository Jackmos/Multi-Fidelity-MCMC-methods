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
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from datetime import datetime

import pickle
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
    plt.plot(df.index, lowfid5, label = 'B03_Temp OPC')
    plt.plot(df.index, lowfid6, label = 'B03_Temp HYT')
    plt.plot(df.index, lowfid7, label = 'B03_Bin0')
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
test_idx = df.index.get_loc('2020-06-05 14:00:00')
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
    scale_lf5 = 2
    scale_lf6 = 2
    scale_lf7 = 30
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

if not MATLAB:
    #Plot data
    plt.figure()
    plt.plot(df.index, lowfid1/ scale_lf1, label = 'B03_RH HYT')
    plt.plot(df.index, lowfid2/ scale_lf2, label = 'B03_RH OPC')
    plt.plot(df.index, lowfid3/ scale_lf3, label = 'B03_PM2.5')
    plt.plot(df.index, lowfid4/ scale_lf4, label = 'SDS011_PM2.5')
    plt.plot(df.index, lowfid5/ scale_lf5, label = 'B03_Temp OPC')
    plt.plot(df.index, lowfid6/ scale_lf6, label = 'B03_Temp HYT')
    plt.plot(df.index, lowfid7/ scale_lf7, label = 'B03_Bin0')
    plt.plot(df.index, highfid/ scale_hf, label = 'grimm_PM2.5')
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
    fidelity_order_labels = original_fidelity_labels
    #fidelity_order_labels = [0, 3, 2, 6, 7] #optimal solution first run
    fidelity_order_labels = [0, 3, 1, 4, 5] #optimal solution good run
else:
    fidelity_order_labels = np.array(fidelity_order_labels, dtype = int).flatten()+1
    fidelity_order_labels = np.concatenate(([0], fidelity_order_labels))
#fidelity_order = original_fidelity_order
#print(type(fidelity_order_labels))

n_models = len(fidelity_order_labels) #redundant if not MATLAB, else necessary to define variable   
fidelity_order_train_seq = [original_fidelity_order_train_seq[i] for i in fidelity_order_labels]
fidelity_order_train = [original_fidelity_order_train[i] for i in fidelity_order_labels]
fidelity_order_val = [original_fidelity_order_val[i] for i in fidelity_order_labels]
fidelity_order_test = [original_fidelity_order_test[i] for i in fidelity_order_labels]

print('Number of models: ', n_models, '\nFidelity order', fidelity_order_labels)
#%%
if False:
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
    all_goodness_test = [] # list of all goodness levels reached after adding different fidelity stages.
    
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
            if not MATLAB:
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
            # Function to format the x-axis labels
            def format_date(x, pos=None):
                # Format the datetime into two lines: hour:minute on the top line and year-month-day on the bottom line
                return f'{x.strftime("%H:%M")}\n{x.strftime("%Y-%m-%d")}'
            date_format = mdates.DateFormatter('%H:%M\n%d/%m')  # Example format: YYYY-MM-DD HH:MM:SS
            ymin = 1
            ymax = 1000
    
            plt.gca().xaxis.set_major_formatter(FuncFormatter(date_format))
    
            plt.plot(time_test, y_test.flatten() * scale, label = 'HF', color = 'red', zorder = 1)
            plt.plot(time_test, mean_sim* scale, label = 'mean', color = 'blue', zorder = 2)
            #plt.fill_between(time_test, (mean_sim - std_sim)* scale, (mean_sim + std_sim)* scale, color='lightblue', alpha=0.5, label='mean $\pm$ std')
    
            plt.plot(time_train,yhf * scale, 'r', zorder = 1)
            plt.plot(time_train, np.mean(np.array(pred_sim_train), axis = 0).flatten()*scale, 'b', zorder = 2)
            
            plt.plot(time_val,y_val.flatten() * scale, 'r', zorder = 1)
            plt.plot(time_val, np.mean(np.array(pred_sim_val), axis = 0).flatten()*scale, 'b', zorder = 2)
            
            x_plot = time_train
            x_plot = [x_plot[0],x_plot[-1]]
            y1_plot = [ymin-1,ymin-1]
            y2_plot = [ymax+1, ymax+1]
            plt.fill_between(x_plot, y1_plot, y2_plot, color = 'grey', alpha = 0.15, zorder = 0)
            
            x_plot = time_val
            x_plot = [x_plot[0],x_plot[-1]]
            y1_plot = [ymin-1,ymin-1]
            y2_plot = [ymax+1, ymax+1]
            plt.fill_between(x_plot, y1_plot, y2_plot, color = 'grey', alpha = 0.3, zorder = 0)
            
            plt.xlabel('Time')
            plt.yscale('log')
            plt.ylim([ymin, ymax])
            plt.ylabel('Mass concentration [$\mu$g/m$^3$]')
            plt.yticks([1,10,100,1000], labels = ['1','10','100','1000'])
        
            plt.legend()
            plt.show()
            
        all_goodness_val.append(np.mean(((y_val.flatten() - np.array(pred_sim_val).flatten())**2)))
    
        if not MATLAB:
            all_goodness_train.append(np.mean(((yhf.flatten() - np.array(pred_sim_train).flatten())**2)))
            full_pred_sim.append(pred_sim)
            full_pred_sim_train.append(pred_sim_train)
            all_goodness_test.append(np.mean(((y_test.flatten() - np.array(mean_sim).flatten())**2)))
    
            
        full_pred_sim_val.append(pred_sim_val)
        full_model_list.append(model_list)
        if not MATLAB:
            res = {
                'scale':scale,
                'order':fidelity_order_labels,
                'real_train':yhf.flatten(),
                'real_val':y_val.flatten(),
                'real_test':y_test.flatten(),
                'pred_train':np.array(pred_sim_train).flatten(),
                'pred_val':np.array(pred_sim_val).flatten(),
                'pred_test':np.array(mean_sim).flatten(),
                'all_goodness_train':all_goodness_train,
                'all_goodness_val':all_goodness_val,
                'all_goodness_test':all_goodness_test,
                }
            with open(f'resBIS_{n_model}.pkl', 'wb') as file:
                pickle.dump(res, file)
    
    
    
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
#%% get image and goodness
all_goodness_train = [] # list of all goodness levels reached after adding different fidelity stages.
all_goodness_val = [] # list of all goodness levels reached after adding different fidelity stages.
all_goodness_test = [] # list of all goodness levels reached after adding different fidelity stages.
MaxErr_train = []
MaxErr_val = []
MaxErr_test = []
ME_train = []
ME_val = []
ME_test = []
x_corr = pd.Timedelta(hours=-1-6/60); #put the time axis to the grimm time measurement
for n_model in range(n_models):
    print('######### RESULTS ' + str(n_model+1) + ' of ' + str(n_models)+ ' #########')
    with open(f'resBIS_{n_model}.pkl', 'rb') as file:
        # Load the data from the pickle file
        res = pickle.load(file)
    
    scale = res['scale']
    fidelity_order_labels = res['order']
    yhf = res['real_train'].reshape(-1, 1)
    y_val = res['real_val'].reshape(-1, 1)
    y_test = res['real_test'].reshape(-1, 1)
    pred_sim_train = res['pred_train'].reshape(-1, 1)
    pred_sim_val = res['pred_val'].reshape(-1, 1)
    mean_sim = res['pred_test'].reshape(-1, 1)
    all_goodness_train = res['all_goodness_train']
    all_goodness_val = res['all_goodness_val']
    all_goodness_test = res['all_goodness_test'] # all_goodness_test is not loaded, as we still need to cut off a part where the data is not available
    
    MaxErr_train.append(np.max(yhf-pred_sim_train))
    MaxErr_val.append(np.max(y_val-pred_sim_val))
    MaxErr_test.append(np.max(y_test-mean_sim))
    ME_train.append(np.mean(yhf-pred_sim_train))
    ME_val.append(np.mean(y_val-pred_sim_val))
    ME_test.append(np.mean(y_test-mean_sim))
    
    fig = plt.figure(figsize=(7.02625,3))
    plt.rcParams.update({'font.size': 9})
    plt.title('                   Training                  Validation                                            Test', fontsize=9, loc = 'left', pad = 0)

    date_format = mdates.DateFormatter('%H:%M\n%d/%m')  # Example format: YYYY-MM-DD HH:MM:SS
    ymin = 1
    ymax = 200

    plt.gca().xaxis.set_major_formatter(FuncFormatter(date_format))

    plt.plot(time_test+x_corr, y_test.flatten() * scale, label = '${y}_\\text{HF}$: HF Signal', color = 'red', zorder = 1)
    plt.plot(time_test+x_corr, mean_sim* scale, label = '$\hat{y}_\\text{HF}^{('+str(n_model+1)+')}$: MF Soft Sensor', color = 'blue', zorder = 2)
    #plt.fill_between(time_test, (mean_sim - std_sim)* scale, (mean_sim + std_sim)* scale, color='lightblue', alpha=0.5, label='mean $\pm$ std')

    plt.plot(time_train+x_corr,yhf * scale, 'r', zorder = 1)
    plt.plot(time_train+x_corr, pred_sim_train*scale, 'b', zorder = 2)
    
    plt.plot(time_val+x_corr,y_val.flatten() * scale, 'r', zorder = 1)
    plt.plot(time_val+x_corr, pred_sim_val*scale, 'b', zorder = 2)
    
    x_plot = time_train+x_corr
    x_plot = [x_plot[0],x_plot[-1]]
    y1_plot = [ymin-1,ymin-1]
    y2_plot = [ymax+1, ymax+1]
    plt.fill_between(x_plot, y1_plot, y2_plot, color = 'grey', alpha = 0.15, zorder = 0)
    
    x_plot = time_val+x_corr
    x_plot = [x_plot[0],x_plot[-1]]
    y1_plot = [ymin-1,ymin-1]
    y2_plot = [ymax+1, ymax+1]
    plt.fill_between(x_plot, y1_plot, y2_plot, color = 'grey', alpha = 0.3, zorder = 0)
    
    plt.xlabel('Time [hh:mm] of the day [dd/mm]')
    plt.yscale('log')
    plt.ylim([ymin, ymax])
    plt.ylabel('Mass concentration [µg/m$^3$]',labelpad=0)
    plt.yticks([1,10,100], labels = ['1','10','100'])

    plt.legend(loc='upper left')
    plt.show()
    plt.grid(axis='y', which='major', zorder = 0)

    plt.grid(axis='x', which='both', zorder = 0)
    minor_locator = MultipleLocator(0.5)  # Set the minor tick spacing here
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.xlim([time_train[0]+x_corr+pd.Timedelta(hours=-3),time_test[-1]+x_corr+pd.Timedelta(hours=3)])
    
    
    
    # define small range
    small_startIdx = np.where(time_test+x_corr>=datetime.strptime('2020-06-04 17:00:00', '%Y-%m-%d %H:%M:%S'))[0][0]
    small_endIdx = np.where(time_test+x_corr>=datetime.strptime('2020-06-05 1:01:00', '%Y-%m-%d %H:%M:%S'))[0][0]
    time_small = time_test[small_startIdx:small_endIdx]
    y_test_small = y_test[small_startIdx:small_endIdx]
    pred_small = mean_sim[small_startIdx:small_endIdx]
    
    plt.plot(np.array([time_small[0],time_small[0],time_small[-1],time_small[-1],time_small[0]])+x_corr, [1.1,20,20,1.1,1.1], 'k', lw = '0.8')
    plt.plot(np.array([time_small[0], time_test[4920]])+x_corr, [1.1,23], 'k', lw = '0.8')
    plt.plot(np.array([time_small[0], time_test[4920]])+x_corr, [20,150], 'k', lw = '0.8')
        
    plt.tight_layout()

    a = plt.axes([.51, .64, .33, .24], facecolor = (1,1,1,0.9))
    
    plt.plot(time_small+x_corr, y_test_small* scale, label = 'HF', color = 'red', zorder = 1)
    plt.plot(time_small+x_corr, pred_small* scale, label = 'Soft Sensor', color = 'blue', zorder = 2, alpha=0.8)
    plt.yscale('log')
    plt.ylim([1,20])
    plt.yticks([1,10], labels = ['1','10'])
    date_format = mdates.DateFormatter('%H:%M')  # Example format: YYYY-MM-DD HH:MM:SS
    plt.gca().xaxis.set_major_formatter(FuncFormatter(date_format))
    plt.xlim(np.array([time_small[0],time_small[-1]])+x_corr)
    xtck = time_small + x_corr
    plt.xticks(xtck[::120], fontsize=8, minor = True)
    plt.xticks(xtck[60::120], fontsize=7)
    plt.tick_params(axis='x', pad=1)
    plt.tick_params(axis='y', pad=1)
    plt.yticks(fontsize=8)
    plt.grid(axis='y', which='major', zorder = 0)
    plt.grid(axis='x', which='both', zorder = 0)
    plt.tight_layout()
    
    
    plt.savefig(f'AirResultsBIS_model{n_model+1}of{n_models}.pdf')
    



#%%
              
x_corr = pd.Timedelta(hours=-1-6/60); #put the time axis to the grimm time measurement

small_startIdx = np.where(time_test+x_corr>=datetime.strptime('2020-06-04 17:00:00', '%Y-%m-%d %H:%M:%S'))[0][0]
small_endIdx = np.where(time_test+x_corr>=datetime.strptime('2020-06-05 1:01:00', '%Y-%m-%d %H:%M:%S'))[0][0]
time_small = time_test[small_startIdx:small_endIdx]
y_test_small = y_test[small_startIdx:small_endIdx]

xtck = time_small + x_corr
date_format = mdates.DateFormatter('%H:%M')  # Example format: YYYY-MM-DD HH:MM:SS

fig = plt.figure(figsize=(3.4127,2))
plt.rcParams.update({'font.size': 9})

plt.xticks(xtck[::120], minor = True)
plt.xticks(xtck[60::120])
plt.xlim(np.array([time_small[0],time_small[-1]])+x_corr)
plt.gca().xaxis.set_major_formatter(FuncFormatter(date_format))
plt.grid(axis='x', which='both', zorder = 0)

X = np.array([np.zeros_like(lowfid0[val_idx:test_idx]), # placeholder for 'grimm_PM2.5'
              lowfid0[val_idx:test_idx],   # time
              lowfid1[val_idx:test_idx],   # 'B03_RH HYT' (out)
              lowfid6[val_idx:test_idx],   # 'B03_Temp HYT'
              lowfid4[val_idx:test_idx],   # 'SDS011_PM2.5'
              lowfid3[val_idx:test_idx],   # 'B03_PM2.5'
              lowfid2[val_idx:test_idx],   # 'B03_RH OPC' (in)
              lowfid5[val_idx:test_idx],   # 'B03_Temp OPC'
              lowfid7[val_idx:test_idx]])   # 'B03_Bin0'
def min_max_normalize_matrix(matrix):
    min_vals = np.min(matrix, axis=1, keepdims=True)
    max_vals = np.max(matrix, axis=1, keepdims=True)
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)
    return normalized_matrix
X = min_max_normalize_matrix(X)
for i in [2,6,3,7,4,8,5]:
    ls = '--' if i>=6 else '-'
    plt.plot(time_small+x_corr,X[i,small_startIdx:small_endIdx], ls=ls, label = r'${y}_\mathrm{LF}^{('+str(i)+')}$')
plt.legend(loc='lower center', bbox_to_anchor=(0.5,1), ncol=4, handlelength=1.5, handletextpad=0.5,columnspacing=1.2, borderpad=0.5)
plt.ylim(0,1)
plt.yscale('linear')
plt.ylabel('Normalized LF [-]')
plt.tight_layout()
plt.savefig(f'normalizedLFinputs.pdf')
#%%
fig = plt.figure(figsize=(3.4127,2))
plt.rcParams.update({'font.size': 9})

plt.xticks(xtck[::120], minor = True)
plt.xticks(xtck[60::120])
plt.xlim(np.array([time_small[0],time_small[-1]])+x_corr)
plt.gca().xaxis.set_major_formatter(FuncFormatter(date_format))
plt.grid(axis='x', which='both', zorder = 0)
for n_model in [2,0,3,1,4]:
    plt.xticks(xtck[::120], minor = True)
    plt.xticks(xtck[60::120])
    plt.xlim(np.array([time_small[0],time_small[-1]])+x_corr)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(date_format))
    plt.grid(axis='x', which='both', zorder = 0)
    
    with open(f'resBIS_{n_model}.pkl', 'rb') as file:
        # Load the data from the pickle file
        res = pickle.load(file)
    scale = res['scale']
    fidelity_order_labels = res['order']
    yhf = res['real_train'].reshape(-1, 1)
    y_val = res['real_val'].reshape(-1, 1)
    y_test = res['real_test'].reshape(-1, 1)
    pred_sim_train = res['pred_train'].reshape(-1, 1)
    pred_sim_val = res['pred_val'].reshape(-1, 1)
    mean_sim = res['pred_test'].reshape(-1, 1)
    
    pred_small = mean_sim[small_startIdx:small_endIdx]
    if n_model == 2:
        plt.plot(time_small+x_corr, y_test_small* scale, label = 'HF', color = 'red', zorder = 1)
    alpha = 0.4
    if n_model == 0:
        linesty = '-'
        lc = 'orange'
    elif n_model == 1:
        linesty = '-'
        lc = 'blue'
    elif n_model == 2:
        linesty = '-.'
        lc = 'green'
    elif n_model == 3:
        linesty = '--'
        lc = 'orange'
        alpha = 0.8
    elif n_model == 4:
        linesty = '--'
        lc = 'blue'
        alpha = 0.8
    plt.plot(time_small+x_corr, pred_small* scale, label = '$\hat{y}_\\text{HF}^{('+str(n_model+1)+')}$', color = lc, zorder = 2, alpha=alpha,ls=linesty)
    plt.yscale('log')
    plt.ylim([1,20])
    plt.yticks([1,10], labels = ['1','10'])
    plt.tick_params(axis='x', pad=1)
    plt.tick_params(axis='y', pad=1)
    plt.yticks(fontsize=8)
    plt.grid(axis='y', which='major', zorder = 0)
    plt.ylabel('Mass Conc.\n[µg/m$^3$]',labelpad=0)
plt.legend(loc='lower center', bbox_to_anchor=(0.5,1), ncol=3, handlelength=1.5, handletextpad=0.5,columnspacing=1.2, borderpad=0.5)
    
plt.tight_layout()
plt.savefig(f'differentNNs.pdf')


    
#%% make a dataframe with the model goodness
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

X = np.array([np.concatenate((yhf.flatten(),y_val.flatten())).flatten(), #'grimm_PM2.5'
              lowfid0[:val_idx],   # time
              lowfid1[:val_idx],   # 'B03_RH HYT' (out)
              lowfid6[:val_idx],   # 'B03_Temp HYT'
              lowfid4[:val_idx],   # 'SDS011_PM2.5'
              lowfid3[:val_idx],   # 'B03_PM2.5'
              lowfid2[:val_idx],   # 'B03_RH OPC' (in)
              lowfid5[:val_idx],   # 'B03_Temp OPC'
              lowfid7[:val_idx],   # 'B03_Bin0'
              np.concatenate((pred_sim_train,pred_sim_val)).flatten()# 
              ])

# SKIP next part: introduces unrealistically large lags and in both directions! so shouldn't do this
# check best cross-correlation and roll the signals to accomodate for it (not the final one of course)
# for i in range(1,np.shape(X)[0]-1):
#     signal1 = X[0,:]
#     signal2 = np.copy(X[i,:])
#     cross_corr = np.correlate(signal1, signal2, mode='full')
#     max_corr_index = np.argmax(cross_corr)
#     time_lag = max_corr_index - len(signal1) + 1
#     print("Time lag:", time_lag)
#     X[i,:] = np.roll(signal2, time_lag)
#     plt.figure()
#     plt.plot(signal2)
#     plt.plot(X[i,:])

# make a rolling average using a convolution
window_size = 30  # Size of the rolling window
axis = 1  # Axis along which to compute the rolling mean (1 for rows, 0 for columns)
# Define the kernel for the rolling mean
kernel = np.ones(window_size) / window_size
# Calculate the rolling mean along the specified axis
X = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), axis, X)

#%%
CC = np.corrcoef(X)[0,:]
mutual_info = [mutual_info_regression(X[i].reshape(-1, 1), X[j])[0] for i in range(X.shape[0]) for j in range(X.shape[0])]
mutual_info = np.array(mutual_info).reshape(X.shape[0], X.shape[0])
MI = mutual_info[0,:]

#%%
dfDescr = pd.DataFrame({'Var.': [r'${y}_\mathrm{HF}$', r'${y}_\mathrm{LF}^{(1)}$', r'${y}_\mathrm{LF}^{(2)}$', 
                        r'${y}_\mathrm{LF}^{(3)}$', r'${y}_\mathrm{LF}^{(4)}$', r'${y}_\mathrm{LF}^{(5)}$', 
                        r'${y}_\mathrm{LF}^{(6)}$', r'${y}_\mathrm{LF}^{(7)}$', r'${y}_\mathrm{LF}^{(8)}$'],
                        'Description':[r'HF PM2.5 [µg/m$^3$]', r'Time [s]', r'LF RH [\%]', r'LF Temp. [\degC]',
                                       r'LF PM2.5 Unit 1 [µg/m$^3$]', r'LF PM2.5 Unit 2 [µg/m$^3$]', 
                                       r'LF RH Unit 2 (Internal) [\%]', r'LF Temp. Unit 2 (Internal) [\degC]', 
                                       r'LF PM2.5 Bias Unit 2 [-]']
                        })


#%%
latex_code = dfDescr.to_latex(escape = False, column_format='ll', index = False)
with open('air_descr.tex', 'w') as file:
    file.write(latex_code)


#%% create a dataframe with RMSE of the different results
dfRMSE = pd.DataFrame({'Model': ['$\hat{y}_\\text{HF}^{('+str(i+1)+')}$' for i in range(n_models)],
                       'Training':np.round(np.sqrt(np.array(all_goodness_train)),3).astype(str),
                       'Validation':np.round(np.sqrt(np.array(all_goodness_val)),3).astype(str),
                       'Test':np.round(np.sqrt(np.array(all_goodness_test)),3).astype(str)})
#%%
latex_code = dfRMSE.to_latex(escape = False, column_format='cccc', index = False)
with open('air_RMSE.tex', 'w') as file:
    file.write(latex_code)

#%% goodness of individual LF PM sensors
#sensor LF3= B03_PM2.5, now sensor LF(5)
dfLF5RMSE = {'Model':[r'${y}_\mathrm{LF}^{(5)}$'],
            'Training':[np.round(np.sqrt(np.mean((highfid_train-lowfid3_train)**2)),3).astype(str)],
            'Validation':[np.round(np.sqrt(np.mean((highfid_val-lowfid3_val)**2)),3).astype(str)],
            'Test':[np.round(np.sqrt(np.mean((highfid_test-lowfid3_test)**2)),3).astype(str)],            
            }

#sensor LF4= SDS011_PM2.5, now sensor LF(4)
dfLF4RMSE = {'Model':[r'${y}_\mathrm{LF}^{(4)}$'],
            'Training':[np.round(np.sqrt(np.mean((highfid_train-lowfid4_train)**2)),3).astype(str)],
            'Validation':[np.round(np.sqrt(np.mean((highfid_val-lowfid4_val)**2)),3).astype(str)],
            'Test':[np.round(np.sqrt(np.mean((highfid_test-lowfid4_test)**2)),3).astype(str)],            
            }

dfAll = pd.concat([dfRMSE,pd.DataFrame(dfLF4RMSE),pd.DataFrame(dfLF5RMSE)])

latex_code = dfAll.to_latex(escape = False, column_format='cccc', index = False)
with open('air_RMSEcompare.tex', 'w') as file:
    file.write(latex_code)

#%% adding max absolute error as metric
dfRMSEMaxErrME = pd.DataFrame({'Model': ['$\hat{y}_\\text{HF}^{('+str(i+1)+')}$' for i in range(n_models)],
                       'RMSE_Training':np.round(np.sqrt(np.array(all_goodness_train)),3).astype(str),
                       'RMSE_Validation':np.round(np.sqrt(np.array(all_goodness_val)),3).astype(str),
                       'RMSE_Test':np.round(np.sqrt(np.array(all_goodness_test)),3).astype(str),
                       'MaxErr_Training':np.round(MaxErr_train,3).astype(str),
                       'MaxErr_Validation':np.round(MaxErr_val,3).astype(str),
                       'MaxErr_Test':np.round(MaxErr_test,3).astype(str),
                       'ME_Training':np.round(ME_train,3).astype(str),
                       'ME_Validation':np.round(ME_val,3).astype(str),
                       'ME_Test':np.round(ME_test,3).astype(str),                
                       })

dfLF5RMSEMaxErrME = {'Model':[r'${y}_\mathrm{LF}^{(5)}$'],
            'RMSE_Training':[np.round(np.sqrt(np.mean((highfid_train-lowfid3_train)**2)),3).astype(str)],
            'RMSE_Validation':[np.round(np.sqrt(np.mean((highfid_val-lowfid3_val)**2)),3).astype(str)],
            'RMSE_Test':[np.round(np.sqrt(np.mean((highfid_test-lowfid3_test)**2)),3).astype(str)],   
            'MaxErr_Training':[np.round((np.max((highfid_train-lowfid3_train))),3).astype(str)],
            'MaxErr_Validation':[np.round((np.max((highfid_val-lowfid3_val))),3).astype(str)],
            'MaxErr_Test':[np.round((np.max((highfid_test-lowfid3_test))),3).astype(str)],   
            'ME_Training':[np.round((np.mean((highfid_train-lowfid3_train))),3).astype(str)],
            'ME_Validation':[np.round((np.mean((highfid_val-lowfid3_val))),3).astype(str)],
            'ME_Test':[np.round((np.mean((highfid_test-lowfid3_test))),3).astype(str)],   
            
            }

#sensor LF4= SDS011_PM2.5, now sensor LF(4)
dfLF4RMSEMaxErrME = {'Model':[r'${y}_\mathrm{LF}^{(4)}$'],
            'RMSE_Training':[np.round(np.sqrt(np.mean((highfid_train-lowfid4_train)**2)),3).astype(str)],
            'RMSE_Validation':[np.round(np.sqrt(np.mean((highfid_val-lowfid4_val)**2)),3).astype(str)],
            'RMSE_Test':[np.round(np.sqrt(np.mean((highfid_test-lowfid4_test)**2)),3).astype(str)],
            'MaxErr_Training':[np.round((np.max((highfid_train-lowfid4_train))),3).astype(str)],
            'MaxErr_Validation':[np.round((np.max((highfid_val-lowfid4_val))),3).astype(str)],
            'MaxErr_Test':[np.round((np.max((highfid_test-lowfid4_test))),3).astype(str)],
            'ME_Training':[np.round((np.mean((highfid_train-lowfid4_train))),3).astype(str)],
            'ME_Validation':[np.round((np.mean((highfid_val-lowfid4_val))),3).astype(str)],
            'ME_Test':[np.round((np.mean((highfid_test-lowfid4_test))),3).astype(str)],
            }
dfAllMaxErrME = pd.concat([dfRMSEMaxErrME,pd.DataFrame(dfLF5RMSEMaxErrME),pd.DataFrame(dfLF4RMSEMaxErrME)])

latex_code = dfAllMaxErrME.to_latex(escape = False, column_format='cccc', index = False)
with open('air_RMSEMaxErrMEcompare.tex', 'w') as file:
    file.write(latex_code)








