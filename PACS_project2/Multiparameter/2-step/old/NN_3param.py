######### LIBRARIES ############
# RICORDA DI ELIMINARE TUTTE LIBRERIE CHE NON UTILIZZI
import keras.backend as K
from keras.regularizers import l2
from hyperopt import STATUS_OK, tpe, Trials, hp, fmin
from hyperopt.pyll.stochastic import sample
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from keras.optimizers import Adam,Nadam,Adamax
from ann_functions3D import getModel, kCrossVal, transfBestparam
from time import perf_counter
import pandas
import pickle
import os

seed = 7
np.random.seed(seed)

########################     PREPARATION      ##########################
HF_data = np.loadtxt("../DATA_shear_cube/3p/Data_Train_bases_new/HF_num_data.txt").astype(int) #list of number of HF data
Nlf_models = np.loadtxt("../DATA_shear_cube/3p/Data_Train_bases_new/N_bases.txt").astype(int)
HF_data_str = [str(num) for num in HF_data]
r2_HF_df = pandas.DataFrame(index=HF_data_str)             #dataframe which stores HF R^2
r2_LF_df = pandas.DataFrame()                              #dataframe which stores LF R^2

mse_HF_df = pandas.DataFrame(index=HF_data_str)            #dataframe which stores HF MSE
mse_LF_df = pandas.DataFrame()                             #dataframe which stores LF MSE

U_HF_list  = []
U_LF_list  = []

#########################     TRAIN SET      ##########################
mu_train_LF = np.loadtxt("../DATA_shear_cube/3p/Data_Train_bases_new/mu_train_LF.txt")
Nlf = np.size(mu_train_LF)    # number of low-fidelity data to TRAIN the NN

U_lf_train_full = np.abs(np.loadtxt("../DATA_shear_cube/3p/Data_Train_bases_new/Ulf_train.txt"))

NepoLF = 3000   # number of epochs for first NN: NN_LF
NepoHF = 3000   # number of epochs for third NN: NN_HF

#########################     TEST SET      ##########################
mu_test = np.loadtxt("../DATA_shear_cube/3p/Data_Test_bases_new/mu_test.txt")
N_test = np.size(mu_test)

U_lf_test_full = np.abs(np.loadtxt("../DATA_shear_cube/3p/Data_Test_bases_new/Ulf_test.txt"))

# ADD NOISE

permutation = np.random.permutation(len(mu_train_LF))
mu_train_LF=mu_train_LF[permutation,:][0:10,:]

noise_stddev = np.mean(mu_train_LF,axis=0)*0.025
noise = np.random.normal(0, noise_stddev, np.shape(mu_train_LF))
mu_train_LF=mu_train_LF+noise 

Nlf = np.size(mu_train_LF)  #number of low-fidelity data to TRAIN the NN

U_lf_train_full=U_lf_train_full[permutation][0:10,:]

#Input
mu_max = np.max(mu_test,axis=0)
mu_min = np.min(mu_test,axis=0)

mu_test_norm = (mu_test - mu_min) / (mu_max - mu_min)
mu_train_LF_norm = (mu_train_LF - mu_min) / (mu_max - mu_min)

U_t_max_train=np.max(U_lf_train_full)
U_t_min_train=np.min(U_lf_train_full)

U_t_max_test=np.max(U_lf_test_full)
U_t_min_test=np.min(U_lf_test_full)

U_train_LF_full=(U_lf_train_full-U_t_min_train)/(U_t_max_train-U_t_min_train)
U_test_LF_full=(U_lf_test_full-U_t_min_test)/(U_t_max_test-U_t_min_test)

for m in range(len(Nlf_models)):
# loop over the basis    
        print(f"********************  #basis functions = {Nlf_models[m]}  ********************")
        
        test_mse_HF_list  = []
        test_mse_LF_list  = []

        r2_HF_list  = []
        r2_LF_list  = []
        
        #########################     TRAIN SET      ##########################
        U_train_LF=U_train_LF_full[:,m]
        
        #########################     TRAIN SET      ##########################
        U_test_LF=U_test_LF_full[:,m]
        
        ##########################     NORMALIZATION  ##########################
        #Output
        #lfmean = np.mean(U_test_LF)        # PERCHE LA MEDIA SOLO DI U_test_LF ???????????????????

        #U_train_LF = U_train_LF - lfmean
        #U_test_LF = U_test_LF - lfmean
        
        #U_train_LF= np.exp(U_train_LF*100)*1e-6
        #U_test_LF = np.exp(U_test_LF*100)*1e-6
        
        ##########################       FIRST NN: NN_LF     ##########################
        K.clear_session()
        bestLF_params = {'lr' : 0.0255, 'kernel_init' : 'glorot_uniform', 'opt' : 'Adam'}  # <------------------ 

        modelLF = getModel(bestLF_params,'LF')     # ann_functions
        histLF = modelLF.fit(mu_train_LF_norm, U_train_LF ,epochs=NepoLF,batch_size=Nlf, verbose = 0)
#        histLF = modelLF.fit(Young_LF_norm[:,0], U_train_LF ,epochs=NepoLF,batch_size=Nlf, verbose = 0)
# attenzione -> QUANDO LO FARAI IN PIU DIMENSIONI DI INPUT QUESTA PARTE E' DA SCRIVERE TENENDO CONTO DELLE DIMNSIONI EFFETTIVE DELLA MATRICE DI INPUT, DATO CHE NON POTRAI TAGLIARE ALL'INIZIO
        print('LF NN done')

        ULF = modelLF.predict(mu_test_norm)
        
        U_LF_list.append(ULF)
        print('\nLF Model:')

        test_mse = np.mean(np.square(U_test_LF - ULF[:,0]))
        test_mse_LF_list.append(test_mse)
        print(f"Test MSE: {test_mse}")

        r_2 = 1 - np.sum(np.square(U_test_LF - ULF[:,0])) / np.sum(np.square(U_test_LF - np.mean(U_test_LF)))
        r2_LF_list.append(r_2)
        print(f"R^2: {r_2}")

        
        for n_HF in HF_data:
        #loop over the possible numbers of high-fidelity data

                print(f"\n-------  #HF data = {n_HF}  -------")
                start = perf_counter()

                n_HF_txt = str(n_HF) + '.txt'

                #########################     TRAIN SET      #########################
                mu_train_HF = np.loadtxt("../DATA_shear_cube/3p/Data_Train_bases_new/mu_train_HF_" + n_HF_txt)   # serve quando si provano diverso numero di dati                  
                Nhf = np.size(mu_train_HF)  # number of high-fidelity data to TRAIN the NN
# APPUNTO: cambiare nome a U_hf, per adattarlo a scrittura Uhf?
                U_hf_train = np.abs(np.loadtxt("../DATA_shear_cube/3p/Data_Train_bases_new/Uhf_train_" + n_HF_txt))      # ""
                
                permutation = np.random.permutation(len(mu_train_HF))  # <---
                mu_train_HF=mu_train_HF[permutation][0:5]              # <--
                U_hf_train=U_hf_train[permutation][0:5]                # <--
                
                #########################     TEST SET      ##########################
                U_hf_test = np.abs(np.loadtxt("../DATA_shear_cube/3p/Data_Test_bases_new/Uhf_test.txt"))

                ##########################     NORMALIZATION  ##########################
                # TRANSFORMATION
#                U_hf_train = np.exp(U_hf_train*100)*1e-6
#                U_hf_test = np.exp(U_hf_test*100)*1e-6
                
                #Input
                mu_train_HF_norm = (mu_train_HF - mu_min) / (mu_max - mu_min)

                #Output
                #hfmean = np.mean(U_hf_test)

                #U_hf_train = U_hf_train - hfmean
                #U_hf_test = U_hf_test - hfmean
                
                U_t_min_test=np.min(U_hf_test,axis=0)
                U_t_max_test=np.max(U_hf_test,axis=0)
                
                U_t_min_train=np.min(U_hf_train,axis=0)
                U_t_max_train=np.max(U_hf_train,axis=0)
                
                U_hf_train=(U_hf_train-U_t_min_train)/(U_t_max_train-U_t_min_train)
                U_hf_test=(U_hf_test-U_t_min_test)/(U_t_max_test-U_t_min_test)

                ##########################    SECOND NN: NN_HF    ##########################
                mu_test_help = modelLF.predict(mu_test_norm)[:,0]
                mu_test_in = np.vstack((mu_test_norm.transpose(),mu_test_help)).transpose() # <- TEST INPUT for the second NN: NN_HF

                mu_train_help = modelLF.predict(mu_train_HF_norm)[:,0] #f_LF(mu_hf_train)
                mu_final = np.vstack((mu_train_HF_norm.transpose(),mu_train_help)).transpose() # <- TRAINING INPUT for the second NN: NN_HF

                name = '2step'
                K.clear_session()
                #best paramters obtained by HPO:
                # Young
                #best_param = {'kernel_init': 'uniform', 'l2weight': 0.00012051711558537875, 'lr': 0.0017624104175516413, 'nodes': 18.0, 'opt': 'Adamax'}
                #best_params = {'kernel_init': 'uniform', 'l2weight': 0.00010669470437554761, 'lr': 0.015298517193401867, 'nodes': 12.0, 'opt': 'Adamax'}
                #best_params={'kernel_init': 'uniform', 'l2weight': 0.00021280958667909701, 'lr': 0.01280013966561744, 'nodes': 6.0, 'opt': 'Adam'}
                # Poisson
                best_params={'kernel_init': 'uniform', 'l2weight': 0.00014792980965688785, 'lr': 0.00033536498844198136, 'nodes': 56.0, 'opt': 'Adam'}
                #100%|██████████| 15/15 [28:24<00:00, 113.64s/trial, best loss: 4.971638652351829e-06] {'kernel_init': 'uniform', 'l2weight': 0.000807831416040378, 'lr': 0.0052322673829551325, 'nodes': 48.0, 'opt': 'Adam'
                finalModel = getModel(best_params,name) #final model chosen according to the best paramters
                hist = finalModel.fit(mu_final, U_hf_train, validation_data=(mu_test_in, U_hf_test),epochs=NepoHF,batch_size=Nhf,verbose=0, validation_freq = 20)

                UHF = finalModel.predict(mu_test_in)
                U_HF_list.append(UHF)

                stop = perf_counter()
                elapsed = stop - start
                print('Elapsed time: ', elapsed)
                print('\nHF Model:')

                test_mse = np.mean(np.square(U_hf_test - UHF[:,0]))
                test_mse_HF_list.append(test_mse)
                print(f"Test MSE: {test_mse}")

                r2_HF = 1 - np.sum(np.square(U_hf_test - UHF[:,0])) / np.sum(np.square(U_hf_test - np.mean(U_hf_test)))
                r2_HF_list.append(r2_HF)
                print(f"R^2: {r2_HF}")        
        
        
        r2_LF_df[str(Nlf_models[m])] = r2_LF_list
        r2_HF_df[str(Nlf_models[m])] = r2_HF_list

        mse_LF_df[str(Nlf_models[m])] = test_mse_LF_list
        mse_HF_df[str(Nlf_models[m])] = test_mse_HF_list      

print(r2_HF_df.round(5))
print(mse_HF_df.round(5))  

plt.figure()
plt.scatter(mu_test_in[:,0], U_hf_test,color='red', label = 'HF model')
plt.scatter(mu_final[:,0], U_hf_train,color='blue', label = 'HF training points')
plt.scatter(mu_test_in[:,0], finalModel.predict(mu_test_in),color='green', label = 'Predicted HF model')
plt.legend(prop={'size': 8.3})
plt.show()

plt.figure()
plt.scatter(mu_test_in[:,1], U_hf_test,color='red', label = 'HF model')
plt.scatter(mu_final[:,1], U_hf_train,color='blue', label = 'HF training points')
plt.scatter(mu_test_in[:,1], finalModel.predict(mu_test_in), color='green', label = 'Predicted HF model')
plt.legend(prop={'size': 8.3})
plt.show()

plt.figure()
plt.scatter(mu_test_in[:,2], U_hf_test,color='red', label = 'HF model')
plt.scatter(mu_final[:,2], U_hf_train,color='blue', label = 'HF training points')
plt.scatter(mu_test_in[:,2], finalModel.predict(mu_test_in),color='green', label = 'Predicted HF model')
plt.legend(prop={'size': 8.3})
plt.show()

#########################     SAVE the OUTPUT      ##########################
os.makedirs('Output_MF_new')

r2_HF_df.to_csv('./Output_MF_new/r2_HF_lhs_transformation.txt', header=True, index=False, sep='\t', mode='a')
mse_HF_df.to_csv('./Output_MF_new/mse_HF_lhs_transformation.txt', header = True, index = False, sep = '\t', mode = 'a')
r2_LF_df.to_csv('./Output_MF_new/r2_LF_lhs_transformation.txt', header=True, index=False, sep='\t', mode='a')
mse_LF_df.to_csv('./Output_MF_new/mse_LF_lhs_transformation.txt', header = True, index = False, sep = '\t', mode = 'a')