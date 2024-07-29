# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:09:22 2024

@author: Lips

calculating principal components, correlation coefficient and mutual information of the discontinuous functions to check if they would also be able to discard useless data
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
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# %%

#######################     CONFIGURATIONS     ##########################
seed = 0

#Choose the benchmark case
case = 'discontinuousNoise'#'discontinuous' #'linear'
example = './1D_Benchmark_' + case

scaling = True
train = True
validation = False
save = False
original_order = True

if case == 'linear':
    Nhf = 5
elif 'discontinuous' in case:
    Nhf = 8
N_test = 100
N_val = 7
Nhf_total = 14 #used in new case: takes total number of points for train and val

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
    
    
if case == 'discontinuousNoise':
    #1D Benchmark: Discontinuous functions
    lowfid0 = lambda x: 30*x -10
    lowfid1 = lambda x: (0.5*(6.*x-2.)**2*np.sin(12.*x-4) + 10*(x-0.5) -5)*(x<0.5) + (3+0.5*(6.*x-2)**2*np.sin(12.*x-4)+10*(x-0.5)-5)*(x>0.5)
    highfid = lambda x: (2*lowfid1(x)- 20*x+20)*(x<0.5) + (4 + 2*lowfid1(x)- 20*x+20)*(x>0.5) 
    N_noise = 200
    def noise1(x, seed=1):
        # generate a high resolution noise vector what will be interpolated to systematically get the same noise characteristics
        np.random.seed(seed)
        x_test = np.linspace(0,1,N_noise)
        return np.interp(x, x_test, np.random.random(x_test.shape))
    def cumNoise1(x, seed = 1):
        np.random.seed(seed)
        x_test = np.linspace(0,1,N_noise)
        return np.interp(x, x_test, np.cumsum(np.random.random(x_test.shape)))

    lowfid2 = lambda x: lowfid1(x) + 4*(noise1(x,1)-0.5)
    lowfid3 = lambda x: lowfid1(x) + 1.05e-1*cumNoise1(x,2)
    lowfid4 = lambda x: lowfid1(x) - 8e-2*cumNoise1(x,3) + 2*(noise1(x,4)-0.5)
    lowfid5 = lambda x: 8*(noise1(x,5)-0.5)


#%% Create dataset 

#Input
xhf_total = np.linspace(0,1,Nhf_total+1)-0.03
xhf = xhf_total[2::2]
xhf_val = xhf_total[1::2]
# xhf = np.linspace(0,1,Nhf)
# dx = xhf[1] - xhf[0]
# np.random.seed(3)
# xhf_val = np.random.random(N_val) #
# xhf_val = np.linspace(0+dx/2,1-dx/2,N_val)-0.005
xhf_test = np.linspace(0,1,N_test)

#Plot
plt.figure()
plt.plot(xhf_test, lowfid0(xhf_test), 'k--', label = "$f_{LF^{(0)}}$")
plt.plot(xhf_test, lowfid1(xhf_test), 'r--', label = "$f_{LF^{(1)}}$")
plt.plot(xhf_test, lowfid2(xhf_test), 'b--', label = "$f_{LF^{(2)}}$")
plt.plot(xhf_test, lowfid3(xhf_test), 'g--', label = "$f_{LF^{(3)}}$")
if 'lowfid4' in globals():
    plt.plot(xhf_test, lowfid4(xhf_test), 'm--', label = "$f_{LF^{(4)}}$")
if 'lowfid5' in globals():
    plt.plot(xhf_test, lowfid5(xhf_test), 'c--', label = "$f_{LF^{(5)}}$")
plt.plot(xhf_test, highfid(xhf_test), 'r-', label = "$f_{HF}$")
plt.plot(xhf, highfid(xhf), 'ro')
plt.plot(xhf_val, highfid(xhf_val), 'kx')
plt.xlabel('x')
plt.legend(ncol = 2)
plt.show()


#%% getting mutual_info and correlation coefs for full feature matrix
X = np.array([highfid(xhf_total),
              lowfid0(xhf_total),
              lowfid1(xhf_total),
              lowfid2(xhf_total),
              lowfid3(xhf_total),
              lowfid4(xhf_total),
              lowfid5(xhf_total)
              ])

# Calculate correlation coefficient
corr_coef = np.corrcoef(X)

# Calculate univariate mutual information
mutual_info = [mutual_info_regression(X[i].reshape(-1, 1), X[j])[0] for i in range(X.shape[0]) for j in range(X.shape[0])]
mutual_info = np.array(mutual_info).reshape(X.shape[0], X.shape[0])
# Print correlation coefficient and mutual information
print("Correlation Coefficient:")
print(corr_coef)
print("\nUnivariate Mutual Information:")
print(mutual_info)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax[0].imshow(corr_coef, cmap='coolwarm', vmin=-1, vmax=1)
ax[0].set_title('Correlation Coefficient')
fig.colorbar(im1, ax=ax[0])

# Mutual information color map plot
im2 = ax[1].imshow(mutual_info, cmap='viridis')
ax[1].set_title('Univariate Mutual Information')
fig.colorbar(im2, ax=ax[1])

#%% getting mutual_info and PCA for feature matrix and highfid as target variable
X = np.array([lowfid0(xhf_total),
              lowfid1(xhf_total),
              lowfid2(xhf_total),
              lowfid3(xhf_total),
              lowfid4(xhf_total),
              lowfid5(xhf_total)
              ])
X = X.transpose()
y = highfid(xhf_total)

mutual_info = [mutual_info_regression(X[:, i].reshape(-1, 1), y)[0] for i in range(X.shape[1])]

# Perform PCA considering the target variable
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_pca = pca.fit_transform(X.transpose(), y.transpose())
model = LinearRegression()
model.fit(X_pca, y)

# Print mutual information and transformed features
print("Mutual Information:", mutual_info)
print("\nTransformed Features (after PCA):\n", X_pca)
