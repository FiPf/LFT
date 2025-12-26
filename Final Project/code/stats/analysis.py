import numpy as np
from physics.observables import topological_charge_QR
from tqdm.notebook import tqdm

def bin_sizes_finder(sample_size: int) -> list[int]: 
    """find all integer bin sizes that divide a given sample size, helper for apply_binning. https://en.wikipedia.org/wiki/Highly_composite_number 

    Args:
        sample_size (int): Number of measurements in the dataset. 

    Returns:
        list[int] : sorted list of the divisors
    """
    max_search = int(np.sqrt(sample_size)) + 2
    bin_sizes = []
    for i in range(1, max_search):
        if sample_size % i == 0:
            bin_sizes.append(i)
            bin_sizes.append(sample_size // i)
    bin_sizes.sort()
    return bin_sizes

def apply_binning(measurements: np.ndarray, min_num_measurement_for_bin: int = 200) -> tuple[(np.ndarray, np.ndarray)]:
    """apply binning analysis to a 1D array of measurements to estimate statistical errors. 

    Args: 
        measurements (np.ndarray): 1d array of measurements from a correlated datatset
        min_num_measurements_for_bin (int, optional): minimum number of bins allowed, binning stops if fewer bins would result. Defaults to 200. Minimum is 2. 

    Returns:
        tuple[(np.ndarray, np.ndarray)]: bin_sizes, binned errors
    """
    bin_sizes = bin_sizes_finder(len(measurements))

    binned_errors = []
    for i, size in enumerate(tqdm(bin_sizes, desc="Applying binning")):
        Markov_chain = len(measurements) // size
        if Markov_chain < min_num_measurement_for_bin: 
            bin_sizes = bin_sizes[:i]
            break
        binned_obs = measurements.reshape(Markov_chain, size)
        binned_obs = np.mean(binned_obs, axis = 1)
        binned_errors.append(np.std(binned_obs, ddof = 1)/np.sqrt(Markov_chain)) #one degree of freedom

    return bin_sizes, binned_errors  

def autocorrelation(x: np.ndarray) -> np.array:
    """compute autocorrelation for an array x

    Args:
        x (np.ndarray): array for which to compute the autocorrelation

    Returns:
        np.array: autocorrelation
    """
    x = np.asarray(x)
    n = len(x)
    x_mean = np.mean(x)
    x_var = np.var(x)
    
    autocorr = np.correlate(x - x_mean, x - x_mean, mode='full')[n-1:] / (x_var * n)
    return autocorr

def integrated_autocorr_time(acf: np.ndarray) -> float:
    """compute integrated autocorrelation time

    Args:
        acf (np.ndarray): autocorrelation

    Returns:
        float: integrated autocorrelation time
    """
    acf_positive = acf[acf > 0]
    return 0.5 + np.sum(acf_positive[1:])

def Q_R_time_series(phi_samples_pbc: np.array, phi_samples_apbc: np.array) -> np.array:
    """Compute the time series of the topological charge Q_R

    Args:
        phi_samples_pbc (np.array): field configurations generated with periodic boundary conditions
        phi_samples_apbc (np.array): field configurations generated with aperiodic boundary conditions

    Returns:
        np.array: 1d topological charge time series, evaluated for each Monte Carlo sample
    """
    series = []
    for phi_pbc, phi_apbc in zip(phi_samples_pbc, phi_samples_apbc):
        series.append(topological_charge_QR([phi_pbc], [phi_apbc], spatial_axis=0))
    return np.array(series)
