import numpy as np
from matplotlib import pyplot as plt
import pickle
import re
import seaborn as sns
import pymc3 as pm
from ann_functions import import_data

seed = 7
np.random.seed(seed)


file_path_HF = "../DATA/reaction_diffusion_HF.mat"
(reaction_HF, U_HF) = import_data(file_path_HF)
reaction_HF_test = reaction_HF

# Specifica il percorso del file di testo da cui importare i dati
#file_path = 'estimated_UHF.txt'

# Usa la funzione numpy.loadtxt() per importare i dati
# delimiter specifica il separatore utilizzato nel file (spazio o virgola)
#U_est = np.loadtxt(file_path, delimiter=' ')

# Dati di input e output
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Definizione del modello
with pm.Model() as model:
    # Priori sui parametri
    slope = pm.Normal('slope', mu=0, sd=10)
    intercept = pm.Normal('intercept', mu=0, sd=10)

    # Modello lineare
    mu = slope * X + intercept

    # Likelihood
    likelihood = pm.Normal('y', mu=mu, sd=1, observed=y)

    # Campionamento MCMC
    trace = pm.sample(2000, cores=1)  # Esegui 2000 iterazioni di MCMC

# Analisi dei risultati
pm.summary(trace)
pm.traceplot(trace)
pm.plot_posterior(trace)

# Altri statistiche e grafici possono essere estratti dal "trace" per l'analisi dei risultati
