import numpy as np
from tqdm.notebook import tqdm
import algorithms.metropolis
import physics.actions as actions
from typing import Callable

def phi4_action(phi: np.ndarray, mass2: float, lamb: float) -> np.ndarray:
    action = np.sum((2 + 0.5 * mass2) * (phi ** 2) + (lamb / 24) * (phi ** 4))
    for mu in range(2):
        action -= np.sum(phi*np.roll(phi, shift=-1, axis=mu))
    return action

def gradient_phi4_action(phi: np.ndarray, mass2: float, lamb: float): 
    grad = (2 + 0.5 * mass2) * 2 * phi + (lamb / 24) * 4 * phi**3

    for mu in range(2):
        grad -= np.roll(phi, shift=-1, axis=mu)
        grad -= np.roll(phi, shift=+1, axis=mu)

    return grad

def kinetic_energy(pi: np.array): 
    return 0.5 * np.sum(pi**2) 


def hamiltonian(phi: np.array, pi: np.array, mass2: float, lamb: float, action: Callable = None):
    if action is None:
        potential = phi4_action(phi, mass2, lamb)
    else:
        potential = action(phi, mass2, lamb)

    return kinetic_energy(pi) + potential