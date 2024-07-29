# import numpy as np
# from numba import jit

# # Basic case
# def create_basic_functions():
#     @jit
#     def modified_highfid(x, delta):
#         period = 5.54
#         phase_within_period = np.mod(x + delta / 6, period)
#         return (6. * phase_within_period / 5 - 2.) ** 2 * np.sin(12. * phase_within_period / 5 - 4.)
    
#     @jit
#     def modified_lowfid(x, delta):
#         period2 = 5.96
#         phase_within_period = np.mod(x + 0.2 + delta / 6, period2)
#         return 0.5 * (6. * phase_within_period / 5 - 2.) ** 2 * np.sin(12. * phase_within_period / 5 - 4.) + 10 * (phase_within_period / 5 - 0.5) + 5.

#     return modified_highfid, modified_lowfid

# # Discontinuous case
# def create_discontinuous_functions():
#     @jit
#     def modified_highfid(x, delta):
#         return (2*modified_lowfid(x,delta)- 20*x/5.+20)*(x/5.<0.5) + (4+2*modified_lowfid(x,delta)- 20*x/5.+20+delta)*(x/5.>0.5)  
        
#     @jit
#     def modified_lowfid(x, delta):
#         return (0.5*(6.*x/5.-2.)**2*np.sin(12.*x/5.-4)+10.*(x/5-0.5)-5.)*(x<2.5) + (3+0.5*(6.*x/5.-2)**2*np.sin(12.*x/5.-4)+10*(x/5.-0.5)-5.+delta)*(x>2.5)

#     return modified_highfid, modified_lowfid

# # Oscillatory case
# def create_oscillatory_functions():
#     @jit
#     def modified_highfid(x, delta):
#         return (x/5-np.sqrt(2))*modified_lowfid(x, delta)**2

#     @jit
#     def modified_lowfid(x, delta):
#         return np.sin(delta*x) 

#     return modified_highfid, modified_lowfid



# def get_parameters(example):
#     if example == "Basic":
#         modified_highfid, modified_lowfid = create_basic_functions()

#         # highfid = lambda x,delta: (6.*(x+delta/6)/5-2.)**2 * np.sin(12.*(x+delta/6)/5-4.)
#         # lowfid = lambda x, delta: 0.5*highfid(x, delta) + 10*((x-delta/6)/5-0.5) + 5.
        

#         Nhf = 50
#         Nlf = 100
#         NepoLF = 2000
#         NepoHF = 2000
#         deltas=np.array([0,10,20])

#     elif example == "Discontinuous":
#         modified_highfid, modified_lowfid = create_discontinuous_functions()


                
#         Nhf = 16
#         Nlf = 40
#         NepoLF = 2000
#         NepoHF = 5200
#         deltas=np.linspace(0.,15.,5)

#     elif example == "Oscillatory":
#         modified_highfid, modified_lowfid = create_oscillatory_functions()
        
#         lowfid = lambda x: np.sin(8*np.pi*x/5)
#         highfid = lambda x: (x/5-np.sqrt(2))*lowfid(x)**2
        
#         Nhf = 15
#         Nlf = 64
#         NepoLF = 1000
#         NepoHF = 3000
#         deltas=np.linspace(2/5,8/5,4)*np.pi
#     else:
#         raise ValueError(f"Unsupported example type: {example}")
    
#     xhf = np.linspace(0,5,Nhf)
#     xlf = np.linspace(0,5,Nlf)
#     #x_test = np.linspace(0,5,1000)

#     datahf= np.array(np.meshgrid(xhf,deltas)).T.reshape(-1, 2)
#     ord_index = np.lexsort((datahf[:, 0], datahf[:, 1]))
#     datahf = datahf[ord_index]
#     Yhf = modified_highfid(datahf[:,0],datahf[:,1])
#     datalf= np.array(np.meshgrid(xlf,deltas)).T.reshape(-1, 2)
#     ord_index = np.lexsort((datalf[:, 0], datalf[:, 1]))
#     datalf = datalf[ord_index]
#     Ylf = modified_lowfid(datalf[:,0],datalf[:,1])


#     return {
#         "modified_highfid": modified_highfid,
#         "modified_lowfid": modified_lowfid,
#         # "highfid": highfid,
#         # "lowfid": lowfid,
#         "Nhf": Nhf,
#         "Nlf": Nlf,
#         "xhf": xhf,
#         "xlf": xlf,
#         "Yhf": Yhf,
#         "Ylf": Ylf,
#         "NepoLF": NepoLF,
#         "NepoHF": NepoHF,
#         "deltas": deltas
#     }

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Callable, List
from numba import jit

class FidelityFunctionModified:
    def __init__(self, case: str) -> None:
        """
        Initializes the FidelityFunctionModified object with a specified case.

        :param case: The case to use ('Basic', 'Discontinuous', or 'Oscillatory').
        """
        self.cases = {
            "Basic": self.create_basic_functions,
            "Discontinuous": self.create_discontinuous_functions,
            "Oscillatory": self.create_oscillatory_functions
        }

        if case not in self.cases:
            raise ValueError(f"Example {case} not recognized")

        self.modified_highfid, self.modified_lowfid = self.cases[case]()
        self.data = self.get_parameters(case)

    @staticmethod
    def create_basic_functions() -> Tuple[Callable, Callable]:
        """
        Creates the high-fidelity and low-fidelity functions for the Basic case.
        
        :return: Tuple of modified high fidelity and low fidelity functions.
        """
        @jit(nopython=True)
        def modified_highfid(x: np.ndarray, delta: float) -> np.ndarray:
            period = 5.54
            phase_within_period = np.mod(x + delta / 6, period)
            return (6. * phase_within_period / 5 - 2.) ** 2 * np.sin(12. * phase_within_period / 5 - 4.)

        @jit(nopython=True)
        def modified_lowfid(x: np.ndarray, delta: float) -> np.ndarray:
            period2 = 5.96
            phase_within_period = np.mod(x + 0.2 + delta / 6, period2)
            return 0.5 * (6. * phase_within_period / 5 - 2.) ** 2 * np.sin(12. * phase_within_period / 5 - 4.) + 10 * (phase_within_period / 5 - 0.5) + 5.

        return modified_highfid, modified_lowfid

    @staticmethod
    def create_discontinuous_functions() -> Tuple[Callable, Callable]:
        """
        Creates the high-fidelity and low-fidelity functions for the Discontinuous case.
        
        :return: Tuple of modified high fidelity and low fidelity functions.
        """
        @jit(nopython=True)
        def modified_highfid(x: np.ndarray, delta: float) -> np.ndarray:
            return (2 * modified_lowfid(x, delta) - 20 * x / 5 + 20) * (x / 5 < 0.5) + \
                   (4 + 2 * modified_lowfid(x, delta) - 20 * x / 5 + 20 + delta) * (x / 5 > 0.5)
        @jit(nopython=True)
        def modified_lowfid(x: np.ndarray, delta: float) -> np.ndarray:
            return (0.5 * (6. * x / 5 - 2.) ** 2 * np.sin(12. * x / 5 - 4) + 
                    10 * (x / 5 - 0.5) - 5.) * (x < 2.5) + \
                   (3 + 0.5 * (6. * x / 5 - 2.) ** 2 * np.sin(12. * x / 5 - 4) + 
                    10 * (x / 5 - 0.5) - 5. + delta) * (x > 2.5)

        return modified_highfid, modified_lowfid

    @staticmethod
    def create_oscillatory_functions() -> Tuple[Callable, Callable]:
        """
        Creates the high-fidelity and low-fidelity functions for the Oscillatory case.
        
        :return: Tuple of modified high fidelity and low fidelity functions.
        """
        @jit(nopython=True)
        def modified_highfid(x: np.ndarray, delta: float) -> np.ndarray:
            return (x / 5 - np.sqrt(2)) * modified_lowfid(x, delta) ** 2
        @jit(nopython=True)
        def modified_lowfid(x: np.ndarray, delta: float) -> np.ndarray:
            return np.sin(delta * x)

        return modified_highfid, modified_lowfid

    def get_parameters(self, example: str) -> Dict[str, Any]:
        """
        Returns the parameters for the specified example case.

        :param example: The case to get parameters for.
        :return: Dictionary containing parameters and evaluated function values.
        """
        if example == "Basic":
            Nhf, Nlf, NepoLF, NepoHF = 50, 100, 2000, 2000
            deltas = np.array([0, 10, 20])
        elif example == "Discontinuous":
            Nhf, Nlf, NepoLF, NepoHF = 16, 40, 2000, 5200
            deltas = np.linspace(0., 15., 5)
        elif example == "Oscillatory":
            Nhf, Nlf, NepoLF, NepoHF = 15, 64, 1000, 3000
            deltas = np.linspace(2 / 5, 8 / 5, 4) * np.pi
        else:
            raise ValueError(f"Unsupported example type: {example}")

        xhf = np.linspace(0, 5, Nhf)
        xlf = np.linspace(0, 5, Nlf)

        # Prepare high fidelity data
        datahf = self._create_meshgrid(xhf, deltas)
        Yhf = self.modified_highfid(datahf[:, 0], datahf[:, 1])

        # Prepare low fidelity data
        datalf = self._create_meshgrid(xlf, deltas)
        Ylf = self.modified_lowfid(datalf[:, 0], datalf[:, 1])

        return {
            "modified_highfid": self.modified_highfid,
            "modified_lowfid": self.modified_lowfid,
            "Nhf": Nhf,
            "Nlf": Nlf,
            "xhf": datahf,            # "xhf": xhf,
            "xlf": datalf,            # "xlf": xlf,            
            # "datahf": datahf,
            # "datalf": datalf,
            "x_test":np.linspace(0, 5, 10000),
            "datatest":self._create_meshgrid(np.linspace(0, 5, 10000), deltas),        
            "Yhf": Yhf,
            "Ylf": Ylf,
            "NepoLF": NepoLF,
            "NepoHF": NepoHF,
            "deltas": deltas
        }

    @staticmethod
    def _create_meshgrid(x: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """
        Creates a meshgrid and reshapes it for vectorized computation.
        
        :param x: Array of x values.
        :param deltas: Array of delta values.
        :return: Reshaped meshgrid array.
        """
        meshgrid = np.array(np.meshgrid(x, deltas)).T.reshape(-1, 2)
        ord_index = np.lexsort((meshgrid[:, 0], meshgrid[:, 1]))
        return meshgrid[ord_index]

    def get_data(self) -> Dict[str, Any]:
        """
        Returns the evaluated data for the selected case.

        :return: Dictionary containing evaluated data.
        """
        return self.data

    def plot_functions(self) -> None:
        """
        Plots the high fidelity and low fidelity functions along with their data points.
        """
        x_values = np.linspace(0, 5, 1000)
        deltas = self.data["deltas"]

        plt.figure(figsize=(12, 8))
        for delta in deltas:
            y_highfid = self.data["modified_highfid"](x_values, delta)
            y_lowfid = self.data["modified_lowfid"](x_values, delta)
            plt.plot(x_values, y_highfid, label=fr'High Fidelity, $\delta$={delta}')
            plt.plot(x_values, y_lowfid, label=fr'Low Fidelity, $\delta$={delta}', linestyle='--')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.title('Fidelity Functions for Various Deltas')
        plt.show()

    def plot_detailed_functions(self) -> None:
        """
        Plots the high fidelity and low fidelity functions along with their detailed data points.
        """
        deltas = self.data["deltas"]

        datahf = self._create_meshgrid(self.data["xhf"], deltas)
        Yhf = self.data["modified_highfid"](datahf[:, 0], datahf[:, 1])

        datalf = self._create_meshgrid(self.data["xlf"], deltas)
        Ylf = self.data["modified_lowfid"](datalf[:, 0], datalf[:, 1])

        x_test = np.linspace(0, 5, 10000)
        datatest = self._create_meshgrid(x_test, deltas)

        plt.figure(figsize=(12, 8))
        plt.plot(datatest[:, 0], self.data["modified_highfid"](datatest[:, 0], datatest[:, 1]), 'r', label='High-Fidelity Sol')
        plt.plot(datatest[:, 0], self.data["modified_lowfid"](datatest[:, 0], datatest[:, 1]), 'g', label='Low-Fidelity Sol')
        plt.xlabel('x')
        plt.legend()
        plt.grid(True)
        plt.title('Benchmark 1D - Detailed')
        plt.show()


def process_data(datahf: np.ndarray, parameters: np.ndarray, t_eval: np.ndarray, Yhf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the elements of a dataset nearest to the ones given.
    
    Parameters:
    - datahf: 2D numpy array where datahf[:,1] contains parameter values.
    - parameters: 1D numpy array of parameter values to find in datahf.
    - t_eval: 2D numpy array of evaluation times.
    - Yhf: 1D numpy array of corresponding y values.
    
    Returns:
    - nearest_x: 1D numpy array of x values closest to each t_eval.
    - y_obs: 1D numpy array of corresponding y values from Yhf.
    """
    indices = np.where(datahf[:, 1] == parameters[0])[0]
    if len(indices) == 0:
        raise ValueError(f"No observations related to parameter: {parameters[0]}")
    
    datahf_values = datahf[indices, 0].reshape(-1, 1)
    t_eval_values = t_eval.reshape(1, -1)
    differences = np.abs(datahf_values - t_eval_values)
    closest_indices = np.argmin(differences, axis=0)
    nearest_x = datahf[indices[closest_indices], 0]
    y_obs = Yhf[indices[closest_indices]]
    
    return nearest_x, y_obs




def run_simulation(
    datahf: np.ndarray, mean_prior: np.ndarray, cov_prior: np.ndarray, Yhf: np.ndarray, 
    sigma_noise: List[float], n_data: List[int], parameters: np.ndarray, sigma: np.ndarray, 
    rwmh_scaling: np.ndarray, rwmh_cov: np.ndarray, rwmh_adaptive: bool, 
    iterations: int, burnin: int, n_chains: int, final_model, algo: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a simulation to estimate parameters and calculate errors.
    
    Parameters:
    - datahf: 2D numpy array containing data.
    - mean_prior: 1D numpy array for the mean of the prior.
    - cov_prior: 2D numpy array for the covariance of the prior.
    - Yhf: 1D numpy array of observed values.
    - sigma_noise: List of noise levels.
    - n_data: List of number of data points.
    - parameters: 1D numpy array of parameters.
    - sigma: 1D numpy array of standard deviations for the likelihood.
    - rwmh_scaling: 1D numpy array of scaling factors for the RWMH algorithm.
    - rwmh_cov: 2D numpy array for the RWMH covariance.
    - rwmh_adaptive: Boolean indicating if RWMH is adaptive.
    - iterations: Integer for the number of iterations.
    - burnin: Integer for the burn-in period.
    - n_chains: Integer for the number of chains.
    - final_model: The model object with the param_inverse method.
    - algo: String indicating the algorithm to use.
    
    Returns:
    - best_estimate: The best parameter estimate.
    - best_error: The error corresponding to the best estimate.
    """
    
    # Initialize error and estimate arrays
    error = np.zeros((len(sigma_noise), len(n_data), len(sigma), len(rwmh_scaling)))
    estimates = np.zeros((len(sigma_noise), len(n_data), len(sigma), len(rwmh_scaling)))
    
    # Iterate over noise levels
    for i, noise in enumerate(sigma_noise):
        # Iterate over number of data points
        for k, n in enumerate(n_data):
            t_eval = np.linspace(0., 5., n).reshape(-1, 1)  # Generate evaluation times
            nearest_x, y_obs = process_data(datahf, parameters, t_eval, Yhf)  # Get nearest x values and observations
            
            # Iterate over sigma values
            for t, s in enumerate(sigma):
                cov_likelihood = s ** 2 * np.eye(t_eval.shape[0])  # Compute the covariance for the likelihood
                
                # Iterate over scaling factors
                for j, r in enumerate(rwmh_scaling):
                    # Perform parameter estimation and calculate error
                    estimates[i, k, t, j], error[i, k, t, j] = final_model.param_inverse(
                        mean_prior, t_eval, cov_prior=cov_prior, rmwh_scaling=r, 
                        cov_noise=noise, cov_likelihood=cov_likelihood, y_obs=y_obs, 
                        x_real=parameters, number_chains=n_chains, N=iterations, 
                        burn_in=burnin, diagnostic=True, rwmh_cov=rwmh_cov, 
                        rwmh_adaptive=rwmh_adaptive, algo=algo
                    )
    
    # Identify the index of the minimum error
    smallest_index = np.unravel_index(np.argmin(error), error.shape)
    best_estimate = estimates[smallest_index]
    best_error = error[smallest_index]
    
    # Print the best parameters
    print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, "
          f"number of data={n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, "
          f"rwmh_scaling={rwmh_scaling[smallest_index[3]]}")

    return best_estimate, best_error





def run_simulation_cuqi(data,
    mean_prior,
    x_real,
    N,
    burn_in,
    cov_prior,
    sd_noise,  
    adapt,
    proposal_sd,
    number_chains,
    algo,
    x_data,
    n_data,
    final_model
) :

    
    # Initialize error and estimate arrays
    #error = np.zeros((len(sigma_noise), len(n_data), len(sigma), len(rwmh_scaling)))
    estimates = np.zeros((len(sd_noise), len(n_data), len(proposal_sd)))
    
    # Iterate over noise levels
    for i, noise in enumerate(sd_noise):
        # Iterate over number of data points
        for k, n in enumerate(n_data):
            x_data = np.linspace(0., 5., n).reshape(-1, 1)  # Generate evaluation times
            nearest_x, y_obs = process_data(data["xhf"], x_real, x_data, data["Yhf"]) 
            
            # Iterate over sigma values
            for t, s in enumerate(proposal_sd):
                cov_likelihood = s ** 2 * np.eye(x_data.shape[0])  # Compute the covariance for the likelihood
                            
                # Perform parameter estimation and calculate error
                estimates[i, k, t] = final_model.inverse_cuqi(
                                mean_prior=mean_prior,
                                x_real=x_real,
                                y_obs=y_obs,
                                N=N,
                                burn_in=burn_in,
                                cov_prior=cov_prior,
                                sd_noise=noise,  
                                adapt=adapt,
                                scale=s,
                                proposal_sd=s,
                                number_chains=number_chains,
                                algo=algo,
                                x_data=x_data
                            )
    # Identify the index of the minimum error
    # smallest_index = np.unravel_index(np.argmin(error), error.shape)
    # best_estimate = estimates[smallest_index]
    # best_error = error[smallest_index]
    
    # # Print the best parameters
    # print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, "
    #       f"number of data={n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, "
    #       f"rwmh_scaling={rwmh_scaling[smallest_index[3]]}")

    return estimates # best_estimate, #best_error






