import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Callable, List



""" 
Contains the definition of the Benchmark cases and the loop to investigate parameters to solve the inverse Problem
"""


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
            lowfid_val = modified_lowfid(x, delta)
            return (2 * lowfid_val - 20 * x / 5 + 20) * (x / 5 < 0.5) + \
                (4 + 2 * lowfid_val - 20 * x / 5 + 20 + delta) * (x / 5 > 0.5)

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
            Nhf, NhPer, Nlf, NepoLF, NepoPer, NepoHF = 50, 40, 60, 2500,1500,2500 #50,100,2000, 2000
            deltas = np.linspace(0., 28., 4)
            deltas_val=np.linspace(np.min(deltas),np.max(deltas),3)
            deltas_test=np.linspace(np.min(deltas),np.max(deltas),5)

        elif example == "Discontinuous":
            Nhf, NhPer, Nlf, NepoLF,NepoPer, NepoHF = 16, 16, 40, 2000, 1000,5200
            deltas = np.linspace(0., 15., 5)
            deltas_val=np.linspace(np.min(deltas),np.max(deltas),4)
            deltas_test=np.linspace(np.min(deltas),np.max(deltas),6)

        elif example == "Oscillatory":
            Nhf, NhPer, Nlf, NepoLF,NepoPer, NepoHF = 20, 20, 64, 1000, 1000, 2000       # Nhf=15  ,64,1000,  3000
            deltas = np.linspace(1 / 5, 6 / 5, 5) * np.pi               #8
            deltas_val=np.linspace(np.min(deltas),np.max(deltas),4)
            deltas_test=np.linspace(np.min(deltas),np.max(deltas),6)

        else:
            raise ValueError(f"Unsupported example type: {example}")
        
        xhf = np.linspace(0, 5, Nhf)
        xhfPer = np.linspace(0, 5, NhPer)
        xlf = np.linspace(0, 5, Nlf)

        # Prepare high fidelity data
        datahf = self._create_meshgrid(xhf, deltas)
        Yhf = self.modified_highfid(datahf[:, 0], datahf[:, 1])

        datahfPer = self._create_meshgrid(xhfPer, deltas)
        YhfPer = self.modified_highfid(datahfPer[:, 0], datahf[:, 1])

        # Prepare low fidelity data
        datalf = self._create_meshgrid(xlf, deltas)
        Ylf = self.modified_lowfid(datalf[:, 0], datalf[:, 1])

        return {
            "modified_highfid": self.modified_highfid,
            "modified_lowfid": self.modified_lowfid,
            "Nhf": Nhf,
            "NhPer": NhPer,
            "Nlf": Nlf,
            "xhf": datahf,
            "xhfPer":datahfPer,          
            "xlf": datalf,                       
            "x_vallf":self._create_meshgrid(np.linspace(0, 5, Nlf), deltas_val),
            "x_valhfper":self._create_meshgrid(np.linspace(0, 5, NhPer), deltas_val),
            "x_valhf":self._create_meshgrid(np.linspace(0, 5, Nhf), deltas_val),
            "x_test":np.linspace(0, 5, 10000),
            "Yhf": Yhf,
            "YhfPer":YhfPer,
            "Ylf": Ylf,
            "dataval_lf":self.modified_lowfid(self._create_meshgrid(np.linspace(0, 5, Nlf), deltas_val)[:,0],self._create_meshgrid(np.linspace(0, 5, Nlf), deltas_val)[:,1]),  
            "dataval_hfper":self.modified_highfid(self._create_meshgrid(np.linspace(0, 5, NhPer), deltas_val)[:,0],self._create_meshgrid(np.linspace(0, 5, NhPer), deltas_val)[:,1]),              
            "dataval_hf":self.modified_highfid(self._create_meshgrid(np.linspace(0, 5, Nhf), deltas_val)[:,0],self._create_meshgrid(np.linspace(0, 5, Nhf), deltas_val)[:,1]),        
            "datatest":self._create_meshgrid(np.linspace(0, 5, 10000), deltas_test),        
            "NepoLF": NepoLF,
            "NepoPer": NepoPer,
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
        
        # Plot each segment of high-fidelity solutions separately
        for delta in deltas:
            datatest_segment = self._create_meshgrid(x_test, [delta])
            plt.plot(datatest_segment[:, 0], 
                    self.data["modified_highfid"](datatest_segment[:, 0], datatest_segment[:, 1]), 
                    'r')

        # Plot each segment of low-fidelity solutions separately
        for delta in deltas:
            datatest_segment = self._create_meshgrid(x_test, [delta])
            plt.plot(datatest_segment[:, 0], 
                    self.data["modified_lowfid"](datatest_segment[:, 0], datatest_segment[:, 1]), 
                    'g')

        plt.xlabel('x')
        plt.legend(['High-Fidelity Sol', 'Low-Fidelity Sol'])
        plt.grid(True)
        plt.title('Benchmark 1D - Detailed')
        plt.show()

