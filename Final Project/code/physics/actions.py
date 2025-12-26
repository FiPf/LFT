import numpy as np
from tqdm.notebook import tqdm
import physics.actions as actions
from typing import Callable
from enum import Enum

class BoundaryCondition(Enum):
    """enum containing the boundary condition options

    Args:
        Enum (enum): enum 
    """
    PBC = "PBC"
    APBC = "APBC"

def phi4_action(phi: np.ndarray, mass2: float, lamb: float, bc: BoundaryCondition = BoundaryCondition.PBC) -> float:
    """action of the phi^4 theory

    Args:
        phi (np.ndarray): field configuration
        mass2 (float): mass parameter of the theory
        lamb (float): coupling strength, must be positive
        bc (BoundaryCondition, optional): Which boundary conditions to choose (either period (PBC) or aperiodic (APBC)). Defaults to BoundaryCondition.PBC.

    Returns:
        float: action of the phi^4 theory
    """
    action = np.sum((2 + 0.5 * mass2) * (phi ** 2) + (lamb / 24) * (phi ** 4))
    for mu in range(2):
        phi_fwd = np.roll(phi, shift=-1, axis=mu)

        if bc == BoundaryCondition.APBC and mu == 0:
            slicer = [slice(None)] * phi.ndim
            slicer[mu] = -1
            phi_fwd[tuple(slicer)] *= -1

        action -= np.sum(phi * phi_fwd)

    return action

def gradient_phi4_action(phi: np.ndarray, mass2: float, lamb: float, bc: BoundaryCondition = BoundaryCondition.PBC) -> float: 
    """gradient of the action of the phi^4 theory

    Args:
        phi (np.ndarray): field configuration
        mass2 (float): mass parameter of the theory
        lamb (float): coupling strength, must be positive
        bc (BoundaryCondition, optional): Which boundary conditions to choose (either period (PBC) or aperiodic (APBC)). Defaults to BoundaryCondition.PBC.

    Returns:
        float: gradient of the phi^4 theory action
    """
    grad = (2 + 0.5 * mass2) * 2 * phi + (lamb / 24) * 4 * phi**3

    for mu in range(2):
        phi_fwd = np.roll(phi, shift=-1, axis=mu)
        phi_bwd = np.roll(phi, shift=+1, axis=mu)

        if bc == BoundaryCondition.APBC and mu == 0:
            slicer_fwd = [slice(None)] * phi.ndim
            slicer_fwd[mu] = -1
            phi_fwd[tuple(slicer_fwd)] *= -1

            slicer_bwd = [slice(None)] * phi.ndim
            slicer_bwd[mu] = 0
            phi_bwd[tuple(slicer_bwd)] *= -1

        grad -= phi_fwd
        grad -= phi_bwd

    return grad

def kinetic_energy(pi: np.array) -> float: 
    """kinetic energy of a configuration

    Args:
        pi (np.array): momentum configuration

    Returns:
        float: kinetic energy of the configuration
    """
    return 0.5 * np.sum(pi**2) 

def hamiltonian(phi: np.array, pi: np.array, mass2: float, lamb: float, bc: BoundaryCondition = BoundaryCondition.PBC, action: Callable = None) -> float:
    """hamiltonian from classical mechanics

    Args:
        phi (np.ndarray): field configuration
        mass2 (float): mass parameter of the theory
        lamb (float): coupling strength, must be positive
        bc (BoundaryCondition, optional): Which boundary conditions to choose (either period (PBC) or aperiodic (APBC)). Defaults to BoundaryCondition.PBC.
        action (Callable, optional): Action function of the theory. Defaults to None.

    Returns:
        float: hamiltonian
    """
    if action is None:
        potential = phi4_action(phi, mass2, lamb, bc)
    else:
        potential = action(phi, mass2, lamb, bc)

    return kinetic_energy(pi) + potential