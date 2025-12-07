import numpy as np
from tqdm.notebook import tqdm
import metropolis

def phi4_action(phi: np.ndarray, mass2: float, lamb: float) -> np.ndarray:
    action = np.sum((2 + 0.5 * mass2) * (phi ** 2) + (lamb / 24) * (phi ** 4))
    for mu in range(2):
        action -= np.sum(phi * np.roll(phi, shift=-1, axis=mu))

    return action

def grad_phi4_action(): 
    pass

def kinetic_energy(): 
    pass

def hamiltonian(): 
    pass