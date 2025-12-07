import numpy as np
from tqdm.notebook import tqdm
import metropolis

def mean_field(): 
    pass

def susceptibility(): 
    pass

def sample_observable(initial_phi: np.ndarray, num_samples: int, mass2: float, lamb: float, width: float, rng: np.random.Generator): 
    def mean_field_obs(phi: np.ndarray): 
        return np.abs(np.mean(phi))
    
    phi = initial_phi.copy()
    measurements = []
    acceptance = []

    for _ in range(num_samples): 
        phi, accepted = metropolis.metropolis_step(phi, mass2=mass2, lamb=lamb, width=width, rng=rng)
        measurements.append(mean_field_obs(phi))
        acceptance.append(accepted)

    return measurements, acceptance

def bin_sizes_finder(sample_size: int): 
    max_search = int(np.sqrt(sample_size)) + 2
    bin_sizes = []
    for i in range(1, max_search):
        if sample_size % i == 0:
            bin_sizes.append(i)
            bin_sizes.append(sample_size // i)
    bin_sizes.sort()
    return bin_sizes

def apply_binning(measurements: np.ndarray, min_num_measurement_for_bin: int = 200):
    bin_sizes = bin_sizes_finder(len(measurements))

    binned_errors = []
    for i, size in enumerate(tqdm(bin_sizes, desc="Applying binning")):
        Markov_chain = len(measurements) // size
        if Markov_chain < min_num_measurement_for_bin: 
            bin_sizes = bin_sizes[:i]
            break
        binned_obs = measurements.reshape(Markov_chain, size)
        binned_obs = np.mean(binned_obs, axis = 1)
        binned_errors.append(np.std(binned_obs)/np.sqrt(Markov_chain))

    return bin_sizes, binned_errors  