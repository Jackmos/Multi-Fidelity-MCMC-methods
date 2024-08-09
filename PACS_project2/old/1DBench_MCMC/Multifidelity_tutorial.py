# BENTCHMARK CASES
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
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import tinyDA as tda
from scipy.stats import multivariate_normal, uniform
import arviz as az
import keras




from module_utils import * 
from data_collection import *
sys.path.append('../utils')
from Structure2 import *
from pathlib import Path



"""
Tutorial for the creation of multifidelity Neural Networks
"""

# # path to the current notebook
# current_file_path = Path().resolve()
# # path to the current folder
# load_context_functions(current_file_path.parent.name)



# # fixing the seeds for reproducibility purposes
# seed = 10
# tf.random.set_seed(seed)
# np.random.seed(seed)
# keras.utils.set_random_seed(seed)


import numpy as np
import inspect
from matplotlib import pyplot as plt
from typing import Dict, Any

class FidelityFunctions:
    def __init__(self, case: str, **kwargs: Any) -> None:
        """
        Initializes the FidelityFunctions object with a specified case and optional parameters.

        :param case: The case to use ('Basic', 'Discontinuous', or 'Oscillatory').
        :param kwargs: Additional parameters to override defaults for the specified case.
        """
        self.cases = {
            "Basic": {
                "high_fid": self.high_fidelity_basic,
                "low_fid": self.low_fidelity_basic,
                "Nhf": 5,
                "Nlf": 32,
                "NepoLF": 1000,
                "NepoHF": 500
            },
            "Discontinuous": {
                "high_fid": self.high_fidelity_discontinuous,
                "low_fid": self.low_fidelity_discontinuous,
                "Nhf": 6,
                "Nlf": 32,
                "NepoLF": 5000,
                "NepoHF": 1200
            },
            "Oscillatory": {
                "high_fid": self.high_fidelity_oscillatory,
                "low_fid": self.low_fidelity_oscillatory,
                "Nhf": 15,
                "Nlf": 64,
                "NepoLF": 3000,
                "NepoHF": 3000
            }
        }
        if case not in self.cases:
            raise ValueError(f"Example {case} not recognized")
        
        self.data = self.cases[case]
        self.data.update(kwargs)
        self.evaluate_case_data()

    @staticmethod
    def high_fidelity_basic(x: np.ndarray) -> np.ndarray:
        """
        High fidelity function for the 'Basic' case.
        
        :param x: Input array.
        :return: Calculated high fidelity values.
        """
        return (6.*x/5-2.)**2 * np.sin(12.*x/5-4.)

    @staticmethod
    def low_fidelity_basic(x: np.ndarray) -> np.ndarray:
        """
        Low fidelity function for the 'Basic' case.
        
        :param x: Input array.
        :return: Calculated low fidelity values.
        """
        return 0.5 * FidelityFunctions.high_fidelity_basic(x) + 10 * (x/5-0.5) + 5.

    @staticmethod
    def high_fidelity_discontinuous(x: np.ndarray) -> np.ndarray:
        """
        High fidelity function for the 'Discontinuous' case.
        
        :param x: Input array.
        :return: Calculated high fidelity values.
        """    
        lowfid = FidelityFunctions.low_fidelity_discontinuous(x)
        return (2*lowfid - 20*x/5 + 20) * (x/5 < 0.5) + (4 + 2*lowfid - 20*x/5 + 20) * (x/5 > 0.5)

    @staticmethod
    def low_fidelity_discontinuous(x: np.ndarray) -> np.ndarray:
        """
        Low fidelity function for the 'Discontinuous' case.
        
        :param x: Input array.
        :return: Calculated low fidelity values.
        """
        return (0.5 * (6.*x/5.-2.)**2 * np.sin(12.*x/5.-4) + 10.*(x/5-0.5) - 5.) * (x < 2.5) + \
               (3 + 0.5 * (6.*x/5.-2)**2 * np.sin(12.*x/5.-4) + 10 * (x/5.-0.5) - 5.) * (x > 2.5)

    @staticmethod
    def high_fidelity_oscillatory(x: np.ndarray) -> np.ndarray:
        """
        High fidelity function for the 'Oscillatory' case.
        
        :param x: Input array.
        :return: Calculated high fidelity values.
        """
        lowfid = FidelityFunctions.low_fidelity_oscillatory(x)
        return (x/5 - np.sqrt(2)) * lowfid**2

    @staticmethod
    def low_fidelity_oscillatory(x: np.ndarray) -> np.ndarray:
        """
        Low fidelity function for the 'Oscillatory' case.
        
        :param x: Input array.
        :return: Calculated low fidelity values.
        """
        return np.sin(8 * np.pi * x / 5)

    @staticmethod
    def function_to_string(func: Any) -> str:
        """
        Converts a function to its string representation.

        :param func: The function to convert.
        :return: String representation of the function.
        """
        return inspect.getsource(func).strip()
    
    def plot_functions(self) -> None:
        """
        Plots the high fidelity and low fidelity functions along with their data points.
        """
        #data = self.get_data()
        xhf = self.data["xhf"]
        xlf = self.data["xlf"]
        x_test = self.data["x_test"]
        high_fid = self.data["high_fid"]
        low_fid = self.data["low_fid"]

        plt.figure()
        plt.plot(x_test, high_fid(x_test), 'r', label='high-fidelity sol')
        plt.plot(x_test, low_fid(x_test), 'g', label='low-fidelity sol')
        plt.plot(xhf, high_fid(xhf), 'ro', label='high-fidelity data', markersize=4)
        plt.plot(xlf, low_fid(xlf), 'go', label='low-fidelity data', markersize=4)
        plt.xlabel('x')
        plt.legend()
        plt.title('Benchmark 1D')
        plt.show()

    def evaluate_case_data(self) -> None:
        """
        Evaluates and stores the necessary data for the selected case.
        """
        Nhf = self.data["Nhf"]
        Nlf = self.data["Nlf"]
        self.data["xhf"] = np.linspace(0, 5, Nhf)
        self.data["xlf"] = np.linspace(0, 5, Nlf)
        self.data["x_test"] = np.linspace(0, 5, 1000)
        self.data["Yhf"] = self.data["high_fid"](self.data["xhf"])
        self.data["Ylf"] = self.data["low_fid"](self.data["xlf"])

    def get_data(self) -> Dict[str, Any]:
        """
        Returns the evaluated data for the selected case.

        :return: Dictionary containing evaluated data and function representations.
        """

        return self.data




def main():
    # path to the current notebook
    current_file_path = Path().resolve()
    # path to the current folder
    load_context_functions(current_file_path.parent.name)

    # introduction of a seed for reproducibility purposes
    seed = 10
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


    # Case to consider:
    example = "Discontinuous"
    fidelity_func = FidelityFunctions(example) #fidelity_func = FidelityFunctions(example, Nlf=40, NepoLF=6000)

    data = fidelity_func.get_data()

    # Print the functions in a readable format
    print("High Fidelity Function:\n", fidelity_func.function_to_string(data["high_fid"]), "\n")
    print("Low Fidelity Function:\n", fidelity_func.function_to_string(data["low_fid"]), "\n")
    #print(data)
    # Plot the functions
    fidelity_func.plot_functions()


    ### LOW FIDELITY NEURAL NETWORK ###
    bestLF_params = {'lr' : 0.0255, 'kernel_init' : 'glorot_uniform', 'opt' : 'Adam'}
    definition_LF={
        "network_type": "LF",
        "params": bestLF_params,
        "data_train": data["xlf"],
        "output_train": data["Ylf"],
        "N": data["NepoLF"],
        "n": data["Nlf"],
        "train": True,
        "do_HPO": False,
        "verbose": False
    }
    modelLF=NetworkFactory.build_network(**definition_LF)
    yLF=modelLF.prediction(data["x_test"])
    (mse_LF,R_LF)=modelLF.performance(data["x_test"],data["high_fid"](data["x_test"]))


    ### HIGH FIDELITY NEURAL NETWORK ###
    bestHF_params = {'lr' : 0.0255, 'kernel_init' : 'glorot_uniform', 'opt' : 'Adam'}
    definition_HF={
        "network_type": "LF",
        "params": bestHF_params,
        "data_train": data["xhf"],
        "output_train": data["Yhf"],
        "N": data["NepoHF"],
        "n": data["Nhf"],
        "train": True,
        "do_HPO": False,
        "verbose": False
    }
    modelHF=NetworkFactory.build_network(**definition_HF)
    yHF=modelHF.prediction(data["x_test"])
    (mse_HF,R_HF)=modelHF.performance(data["x_test"],data["high_fid"](data["x_test"]))


    ### 2 steps NEURAL NETWORK ###
    # collection of parameters
    bestLF_params = {
        'lr': 0.0255, 
        'kernel_init': 'glorot_uniform', 
        'opt': 'Adam'
    }
    
    best_params = {
        'kernel_init': 'uniform', 
        'l2weight': 0.00010018625799978436, 
        'lr': 0.07093837044166487, 
        'nodes': 74.0, 
        'opt': 'Adamax'
    }
    params = [bestLF_params, best_params]

    N = [data["NepoLF"], data["NepoHF"]]
    n = [data["Nlf"], data["Nhf"]]

    # names of the networks 
    names = ['LF', 'HF']

    # collection of the parameters
    definition_2steps = {
        "network_type": "2step",
        "names": names,
        "params": params,
        "data_train": [data["xlf"], data["xhf"]],
        "output_train": [data["Ylf"], data["Yhf"]],
        "N": N,
        "n": n,
        "train": True,
        "do_HPO": False,
        "verbose": False
    }


    # creation of 2 steps model 
    final_model = NetworkFactory.build_network(**definition_2steps)
    
    if final_model == -1:
        raise ValueError("Failed to build the network. Check the network type and parameters.")
    

    y_test = final_model.prediction(data["x_test"])

    # 2 steps performance 
    (mse_MF,R_MF)=final_model.performance(data["x_test"].reshape(-1,1),data["high_fid"](data["x_test"]))


    ### PLOT of predicted models ###
    #LF single-fidelity
    plt.figure()
    plt.plot(data["xlf"],data["Ylf"],'go', label = 'LF training data', markersize = 4)
    plt.plot(data["x_test"],data["low_fid"](data["x_test"]),'g', label = 'exact LF') 
    plt.plot(data["x_test"],yLF,'k--', label = 'pred LF')
    plt.legend()
    plt.title('Low fidelity model')

    #HF single-fidelity
    plt.figure()
    plt.plot(data["xhf"],data["Yhf"],'ro', label = 'HF training data')
    plt.plot(data["x_test"],data["high_fid"](data["x_test"]),'-r', label = 'exact HF')
    plt.plot(data["x_test"],yHF,'--k', label = 'pred HF')
    plt.legend()
    plt.title('High fidelity model - single-fidelity')

    #2-step MF
    plt.figure()
    plt.plot(data["xhf"],data["Yhf"],'ro', label = 'HF training data')
    plt.plot(data["x_test"],data["high_fid"](data["x_test"]),'-r', label = 'exact HF')
    plt.plot(data["x_test"],y_test,'--k', label = 'pred HF')
    plt.legend()
    plt.title('High fidelity model - 2-step')



if __name__ == "__main__":
    main()
