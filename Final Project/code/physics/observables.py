import numpy as np
from tqdm.notebook import tqdm
import algorithms.metropolis as metropolis
from numpy.random import Generator
from typing import Callable, Optional
from algorithms.metropolis import MetropolisSim
from algorithms.hmc import HMCSim

def susceptibility(measurements, L):
    V = L**2
    return V * (np.mean(measurements**2) - np.mean(measurements)**2)

def magnetization(measurements):
    return np.abs(np.mean(measurements))

def sample_observable(initial_phi: np.ndarray, num_samples: int, mass2: float, lamb: float, width: float, rng: Generator, observable: Optional[Callable] = None, algorithm: str = None, progress: bool = True, **kwargs): 
    if observable is None:
        observable = lambda phi: np.abs(np.mean(phi))

    if algorithm is None: 
        raise ValueError("algorithm must be 'Metropolis' or 'HMC'")
    if algorithm == "Metropolis": 
        sim = MetropolisSim(initial_phi, mass2, lamb, width, rng)
    elif algorithm == "HMC": 
        sim = HMCSim(initial_phi, mass2, lamb, width, rng, **kwargs) 

    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'")

    measurements = []

    iterator = range(num_samples)
    if progress:
        iterator = tqdm(iterator, desc=f"Sampling ({algorithm})")

    for _ in iterator:
        sim.update()
        measurements.append(observable(sim.phi))

    accepted = getattr(sim, "accepted_history", None)

    return np.asarray(measurements, dtype=np.float32), accepted

def phi_diff(phi: np.array, spatial_axis: int = 0) -> float:
    phi_0 = np.mean(np.take(phi, indices = 0, axis = spatial_axis))
    phi_L = np.mean(np.take(phi, indices=-1, axis=spatial_axis))
    return 0.5*(phi_L - phi_0)

def topological_charge_QR(phi_PBC: np.ndarray, phi_APBC: np.ndarray, spatial_axis: int = 0) -> float: 
    # https://arxiv.org/pdf/hep-lat/0506003

    phi_mean = np.mean([np.mean(phi) for phi in phi_PBC])
    if np.abs(phi_mean) < 1e-12:
        return 0.0 #by definition in symmetric phase

    phi_diff_mean = np.mean([
        phi_diff(phi, spatial_axis=spatial_axis)
        for phi in phi_APBC
    ])

    return phi_diff_mean / phi_mean