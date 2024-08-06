
import numpy as np
import h5py
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable, Tuple, List


class SystemSolver(ABC):
    """
    Abstract base class for solving systems of differential equations.
    """
    def __init__(self):
        """
        Initialize the solver with default values.
        """
        self.time_points = None  # Time points for the simulation
        self.y_values_list = None  # List to store the solutions for each parameter set
        self.params = None  # List of parameter sets
        self.system_solver = None  # Placeholder for the specific solver instance

    @abstractmethod
    def equations(self, t: float, y: np.ndarray, *params) -> np.ndarray:
        """
        Define the system of differential equations.
        
        Args:
        - t: Current time point
        - y: Current state of the system (array of values)
        - params: Parameters for the differential equations
        """
        pass

    def _runge_kutta_2nd_order(self, y: np.ndarray, t: float, h: float, params: Tuple) -> np.ndarray:
        """
        Perform a single step using the second-order Runge-Kutta method.

        Args:
        - y: Current state of the system (array of values)
        - t: Current time point
        - h: Time step size
        - params: Parameters for the differential equations

        Returns:
        - y_next: State of the system at the next time point
        """
        k1 = h * self.equations(t, y, *params)
        k2 = h * self.equations(t + h, y + k1, *params)
        y_next = y + 0.5 * (k1 + k2)
        return y_next

    def _euler_method_step(self, y: np.ndarray, t: float, h: float, params: Tuple) -> np.ndarray:
        """
        Perform a single step using the Euler method.

        Args:
        - y: Current state of the system (array of values)
        - t: Current time point
        - h: Time step size
        - params: Parameters for the differential equations

        Returns:
        - y_next: State of the system at the next time point
        """
        y_next = y + h * self.equations(t, y, *params)
        return y_next

    def generate_dataset(self, T: float, params: np.ndarray, h: float, fidelity: str) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple]]:
        """
        Generate a dataset of the system's evolution over time for different parameter sets.

        Args:
        - T: Total simulation time
        - params: Array of parameter sets for the differential equations
        - h: Time step size
        - fidelity: Solver method to use ('HF' for high fidelity (Runge-Kutta), otherwise Euler)

        Returns:
        - time_points: Array of time points
        - y_values_list: List of solution arrays for each parameter set
        - params: List of parameter sets
        """
        if T <= 0 or h <= 0:
            raise ValueError("T and h must be positive")

        num_steps = int(T / h)
        self.time_points = np.linspace(0, T, num_steps + 1)
        self.y_values_list = []
        self.params = params

        for param_set in params:
            if self.dim_sys is not None:
                y_initial = np.zeros(self.dim_sys)
            else:
                raise ValueError("Dimension of the system unknown")
            y_initial[:] = 0.5
            y_values = np.zeros((num_steps + 1, len(y_initial)))
            y_values[0, :] = y_initial

            for i in range(num_steps):
                if fidelity == "HF":
                    y_values[i + 1, :] = self._runge_kutta_2nd_order(y_values[i, :], self.time_points[i], h, param_set)
                else:
                    y_values[i + 1, :] = self._euler_method_step(y_values[i, :], self.time_points[i], h, param_set)

            self.y_values_list.append(y_values)

        return self.time_points, self.y_values_list, self.params

    def save_dataset(self, filename: str = 'dataset.h5'):
        """
        Save the generated dataset to an HDF5 file.

        Args:
        - filename: Name of the file to save the dataset
        """
        try:
            with h5py.File(filename, 'w') as h5file:
                h5file.create_dataset('time_points', data=np.array(self.time_points))
                param_group = h5file.create_group('params')
                for i, param_set in enumerate(self.params):
                    param_group.create_dataset(f'set_{i}', data=np.array(param_set))

                y_group = h5file.create_group('y_values_list')
                for i, y_values in enumerate(self.y_values_list):
                    y_group.create_dataset(f'set_{i}', data=y_values)
        except Exception as e:
            print(f"Error saving dataset: {e}")

    def load_dataset(self, filename: str = 'dataset.h5') -> Tuple[np.ndarray, List[np.ndarray], List[Tuple]]:
        """
        Load a dataset from an HDF5 file.

        Args:
        - filename: Name of the file to load the dataset from

        Returns:
        - time_points: Array of time points
        - y_values_list: List of solution arrays for each parameter set
        - params: List of parameter sets
        """
        try:
            with h5py.File(filename, 'r') as h5file:
                self.time_points = np.array(h5file['time_points'])

                self.params = []
                param_group = h5file['params']
                for key in param_group.keys():
                    self.params.append(tuple(param_group[key][:]))

                self.y_values_list = []
                y_group = h5file['y_values_list']
                for key in y_group.keys():
                    self.y_values_list.append(np.array(y_group[key]))
        except Exception as e:
            print(f"Error loading dataset: {e}")

        return self.time_points, self.y_values_list, self.params

    def plot_results(self):
        """
        Plot the results of the system's evolution for different parameter sets.
        """
        plt.figure(figsize=(12, 8))

        for i, y_values in enumerate(self.y_values_list):
            plt.plot(self.time_points, y_values[:, 0], label=f'y1(t) for params={self.params[i]}', marker='o')
            if y_values.shape[1] > 1:
                plt.plot(self.time_points, y_values[:, 1], label=f'y2(t) for params={self.params[i]}', marker='x')
            if y_values.shape[1] > 2:
                plt.plot(self.time_points, y_values[:, 2], label=f'y3(t) for params={self.params[i]}', marker='s')

            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.legend()
            plt.title('System of Differential Equations for Different Sets of Parameters')
            plt.show()

    def get_dim(self):
        """
        Obtain the dimension of the system of equations
        """
        return self.dim_sys 
    
    
class System1(SystemSolver):
    """
    Solver for a specific system of differential equations (System1).
    """
    def __init__(self):
        super().__init__()
        self.dim_sys = 3  # Dimension of the system (number of equations)

    def equations(self, t: float, y: np.ndarray, mu: float) -> np.ndarray:
        """
        Define the system of differential equations for System1.

        Args:
        - t: Current time point
        - y: Current state of the system (array of values)
        - mu: Parameter for the differential equations

        Returns:
        - Array of derivatives (dy/dt)
        """
        dy1dt = y[0] * (mu - 0.1 * y[0] - 0.5 * y[1] - 0.5 * y[2])
        dy2dt = y[1] * (-mu + 0.5 * y[0] - 0.3 * y[2])
        dy3dt = y[2] * (-mu + 0.2 * y[0] + 0.5 * y[1])
        return np.array([dy1dt, dy2dt, dy3dt])

    def print_system(self):
        """
        Print the system of differential equations for System1.
        """
        print("The system of differential equations is:")
        print("dy1/dt = y1 * (mu - 0.1 * y1 - 0.5 * y2 - 0.5 * y3)")
        print("dy2/dt = y2 * (-mu + 0.5 * y1 - 0.3 * y3)")
        print("dy3/dt = y3 * (-mu + 0.2 * y1 + 0.5 * y2)")
        

    
class System2(SystemSolver):
    """
    Solver for a specific system of differential equations (System2).
    """
    def __init__(self):
        """
        Initialize the solver for System2 with default values.
        """
        super().__init__()
        self.dim_sys = 2  # Dimension of the system (number of equations)

    def equations(self, t: float, y: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """
        Define the system of differential equations for System2.

        Args:
        - t: Current time point
        - y: Current state of the system (array of values)
        - a, b, c, d: Parameters for the differential equations

        Returns:
        - Array of derivatives (dy/dt)
        """
        dy1dt = a * y[0] - b * y[0] * y[1]
        dy2dt = c * y[0] * y[1] - d * y[1]
        return np.array([dy1dt, dy2dt])

    def print_system(self):
        """
        Print the system of differential equations for System2.
        """
        print("The system of differential equations is:")
        print("dy1/dt = a * y1 - b * y1 * y2")
        print("dy2/dt = c * y1 * y2 - d * y2")
