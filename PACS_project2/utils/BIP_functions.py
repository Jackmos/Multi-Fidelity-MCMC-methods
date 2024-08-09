import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from typing import Any, Tuple, List
import ray
import logging
import tensorflow as tf
from tinyDA import get_MAP, GaussianRandomWalk, AdaptiveMetropolis, CrankNicolson, DREAMZ, MLDA, sample, to_inference_data
import tinyDA as tda
from cuqi.distribution import JointDistribution
from cuqi.sampler import MH, NUTS, pCN

def MCMC(
    my_posterior: List[Any], 
    N: int, 
    burnin: int, 
    n: int = 1, 
    diagnostic: bool = True,
    rwmh_cov: np.ndarray = None, 
    rmwh_scaling: float = 0.1, 
    period: int = 100, 
    t0: int = 0, 
    rwmh_adaptive: bool = False, 
    algo: str = "MH", 
    dim: int = 0
) -> np.ndarray:
    """
    Perform MCMC sampling.

    Args:
        my_posterior (List[Any]): List of posterior distributions.
        N (int): Number of samples to draw.
        burnin (int): Number of burn-in samples to discard.
        n (int, optional): Number of chains. Defaults to 1.
        diagnostic (bool, optional): Whether to plot diagnostic plots. Defaults to True.
        rwmh_cov (np.ndarray, optional): Covariance matrix for RW-MH. Defaults to None.
        rmwh_scaling (float, optional): Scaling factor for RW-MH. Defaults to 0.1.
        period (int, optional): Adaptation period. Defaults to 100.
        t0 (int, optional): Initial time step. Defaults to 0.
        rwmh_adaptive (bool, optional): Whether to use adaptive RW-MH. Defaults to False.
        algo (str, optional): MCMC algorithm to use. Defaults to "MH".
        dim (int, optional): Dimensionality of the problem. Defaults to 0.

    Returns:
        np.ndarray: Array of estimated parameter means.
    """
    MAP = get_MAP(my_posterior) if dim != 1 else None

    if algo == "MH":
        my_proposal = GaussianRandomWalk(C=rwmh_cov, scaling=rmwh_scaling, adaptive=rwmh_adaptive)
    elif algo == "AM":
        my_proposal = AdaptiveMetropolis(C0=rwmh_cov, adaptive=rwmh_adaptive, period=period, t0=t0)
    elif algo == "CN":
        my_proposal = CrankNicolson(scaling=rmwh_scaling, adaptive=rwmh_adaptive, period=period)
    elif algo == "DREAMZ":
        my_proposal = DREAMZ(M0=10 * dim, adaptive=rwmh_adaptive, period=period)
    elif algo == "MLDA":
        my_proposal = MLDA(
            posteriors=my_posterior, subsampling_rates=[5, 5],
            adaptive_error_model='state-independent', initial_parameters=MAP,
            store_coarse_chain=True, proposal=AdaptiveMetropolis(C0=rwmh_cov)
        )
    else:
        raise ValueError("Unknown algorithm %s" % algo)
    
    my_chains = sample(my_posterior, my_proposal, iterations=N, n_chains=n, force_sequential=True, initial_parameters=MAP)
    idata = to_inference_data(my_chains, burnin=burnin)
    estimates = np.array(az.summary(idata)['mean'])
    print(f"Estimated values are {estimates}")

    if diagnostic:
        print(az.summary(idata))
        az.plot_trace(idata)
        print("Autocorrelation...")
        az.plot_autocorr(idata)

    return estimates


def MCMC_cuqi(
    y: Any, 
    x: Any, 
    observation: np.ndarray, 
    N: int, 
    burn_in: int, 
    n: int = 1, 
    diagnostic: bool = True, 
    algo: str = "MH", 
    adapt: bool = False, 
    scale: float = 0.3,
    parallel: bool = False
) -> np.ndarray:
    """
    Perform MCMC sampling using CUQI library.

    Args:
        y (Any): Dependent variable.
        x (Any): Independent variable.
        observation (np.ndarray): Observed data.
        N (int): Number of samples to draw.
        burn_in (int): Number of burn-in samples to discard.
        n (int, optional): Number of chains. Defaults to 1.
        diagnostic (bool, optional): Whether to plot diagnostic plots. Defaults to True.
        algo (str, optional): Sampling algorithm to use. Defaults to "MH".
        adapt (bool, optional): Whether to use adaptive sampling. Defaults to False.
        scale (float, optional): Scaling factor for MH algorithm. Defaults to 0.3.
        parallel (bool, optional): Whether to run chains in parallel. Defaults to False.

    Returns:
        np.ndarray: Array of estimated parameter means.
    """
    dim = observation.shape[0]
    x_init = np.random.rand(dim)
    estimates = np.empty((1, 0))
    chains = np.empty((0, 1, N - burn_in))
    post = np.empty((1, 0))
    posterior = JointDistribution(x, y)(y=observation)

    if parallel:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        tf.get_logger().setLevel('ERROR')
        ray.init(ignore_reinit_error=True, logging_level=logging.WARNING, log_to_driver=False)
        futures = [chain_creation_parallel.remote(N, burn_in, diagnostic, algo, adapt, scale, posterior, x_init) for _ in range(n)]
        results = ray.get(futures)
        ray.shutdown()
    else:
        results = [chain_creation(N, burn_in, diagnostic, algo, adapt, scale, posterior, x_init) for _ in range(n)]

    for result in results:
        estimates = np.column_stack((estimates, result[0]))
        chains = np.concatenate((chains, result[1]), axis=0)
        post = np.concatenate((post, result[2]), axis=1)

    for l in range(chains.shape[1]):
        plt.figure(figsize=(10, 4))
        for i in range(chains.shape[0]):
            plt.plot(chains[i, l, :])
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title(f'Trace Plot for variable {l}')
        plt.legend([f'Chain {i+1}' for i in range(chains.shape[0])])
        plt.show()

    if diagnostic:
        num_bins = 20
        plt.figure()
        for num in range(post.shape[0]):
            bin_edges = np.linspace(np.min(post[num, :]), np.max(post[num, :]), num_bins + 1)
            hist, _ = np.histogram(post[num, :], bins=bin_edges)
            hist = hist / post.shape[1]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', label=f'Var {num + 1}')
            plt.xlabel('Value')
            plt.ylabel('Probability')
            plt.title('Distribution')
            plt.legend()
            plt.show()

        autocov = az.autocov(chains[:, 0, :])
        ess = az.ess(chains[:, 0, :])
        print(f"ESS values = {ess}")
        plt.figure()
        plt.plot(autocov[0, :])
        plt.title('Autocovariance first chain')
        plt.xlabel('Lag')
        plt.ylabel('Autocovariance')
        plt.legend()
        plt.show()

    return estimates


@ray.remote
def chain_creation_parallel(
    N: int, 
    burn_in: int, 
    diagnostic: bool, 
    algo: str, 
    adapt: bool, 
    scale: float,
    posterior: Any, 
    x_init: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper function to parallelize chain creation using Ray.

    Args:
        N (int): Number of samples.
        burn_in (int): Number of burn-in samples.
        diagnostic (bool): Whether to plot diagnostics.
        algo (str): MCMC algorithm to use.
        adapt (bool): Whether to use adaptive sampling.
        scale (float): Scaling factor for MH algorithm.
        posterior (Any): Posterior distribution to sample from.
        x_init (np.ndarray): Initial parameter values.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Estimates, chains, and posterior samples.
    """
    return chain_creation(N, burn_in, diagnostic, algo, adapt, scale, posterior, x_init)


def preprocess_input(x: np.ndarray, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the input to ensure shapes are compatible for broadcasting.
    
    Parameters:
    x (np.ndarray): The input data.
    mean (np.ndarray): The mean data or other parameter.

    Returns:
    tuple: Processed x and mean.
    """
    if mean.ndim == 1 and mean.size != x.size:
        mean_reshaped = mean.reshape(1, -1)
    else:
        mean_reshaped = mean
    return x, mean_reshaped


def chain_creation(
    N: int, 
    burn_in: int, 
    diagnostic: bool, 
    algo: str, 
    adapt: bool, 
    scale: float,
    posterior: Any, 
    x_init: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a chain using specified MCMC algorithm.

    Args:
        N (int): Number of samples.
        burn_in (int): Number of burn-in samples.
        diagnostic (bool): Whether to plot diagnostics.
        algo (str): MCMC algorithm to use.
        adapt (bool): Whether to use adaptive sampling.
        scale (float): Scaling factor for MH algorithm.
        posterior (Any): Posterior distribution to sample from.
        x_init (np.ndarray): Initial parameter values.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Estimates, chains, and posterior samples.
    """
    if algo == "NUTS":
        sampler = NUTS(posterior, x0=x_init)
    elif algo == "MH":
        sampler = MH(posterior, scale=scale) if not adapt else MH(posterior)
    elif algo == "pCN":
        sampler = pCN(posterior, x0=x_init)
    else:
        raise ValueError(f"Unknown algorithm {algo}")

    samples = sampler.sample_adapt(N - burn_in, burn_in) if adapt else sampler.sample(N - burn_in, burn_in)

    estimates = samples.mean()[:, np.newaxis]
    chains = np.expand_dims(samples.samples, axis=0)
    post = samples.samples

    print(f"Mean values: {estimates.mean(axis=1)}")

    return (estimates, chains, post)


def plot_hist(
    estimates: np.ndarray, 
    real_x: np.ndarray, 
    output1: np.ndarray, 
    output2: np.ndarray,
    par:float
) -> None:
    """
    Plot histogram comparing estimated and real values.

    Args:
        estimates (np.ndarray): Estimated values from the model.
        real_x (np.ndarray): True values to compare against.
        output1 (np.ndarray): First set of output values for comparison.
        output2 (np.ndarray): Second set of output values for comparison.

    Returns:
        None
    """
    diff_output = np.abs(output1 - output2)
    diff_value = np.abs(real_x - estimates)
    print(f"The difference between estimated values {diff_value}\n")

    values = np.vstack((real_x, estimates))
    categories = np.arange(1, values.shape[1] + 1)
    bar_width = 0.35
    bar_positions = [categories - bar_width / 2 + i * bar_width for i in range(values.shape[0])]

    plt.figure()
    for i in range(values.shape[0]):
        plt.bar(bar_positions[i], values[i, :], width=bar_width)

    plt.ylabel('Value')
    plt.title('Comparison of Real and Estimated Values')
    plt.ylim([0, np.max(par) * 1.1])
    plt.xticks(categories)
    plt.legend(["Real value", "Estimate"])
    plt.show()