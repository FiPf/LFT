import numpy as np
import physics.actions as actions
from algorithms.base import Base_sim
from numpy.random import Generator, default_rng
from typing import Optional, Callable
from physics.actions import BoundaryCondition

class MetropolisSim(Base_sim): 
    def __init__(self, initial_phi: np.array, mass2: float, lamb: float, width: float, rng: Generator, bc: BoundaryCondition = BoundaryCondition.PBC):
        super().__init__()
        self.phi = initial_phi.copy()
        self.mass2 = float(mass2)
        self.lamb = float(lamb)
        self.width = float(width)
        self.rng = rng
        self.accepted_history = []
        self.bc = bc

    def propose_phi(self, phi: np.ndarray,
                width: float,
                rng: np.random.Generator) -> np.ndarray:
        delta = 2. * rng.random(size=phi.shape) - 1.  # 2D array with random numbers between -1 and 1.
        return phi + width * delta

    def update(self) -> None:
        proposed_phi = self.propose_phi(self.phi, self.width, self.rng)

        current_action = actions.phi4_action(self.phi, mass2=self.mass2, lamb=self.lamb, bc=self.bc)
        proposed_action = actions.phi4_action(proposed_phi, mass2=self.mass2, lamb=self.lamb, bc=self.bc)
        p_acceptance = np.min([1.0, np.exp(current_action - proposed_action)])

        r = self.rng.random()

        if r <= p_acceptance:  # Accept.
            self.phi = proposed_phi
            self.accepted_history.append(1)
        else: 
            self.accepted_history.append(0)

def run_metropolis(phi0: np.array, mass2: float, lamb: float, width: float, n_steps: int, rng: Optional[Generator] = None, logger: Optional[Callable] = None, progress: bool = False, bc: BoundaryCondition = BoundaryCondition.PBC) -> MetropolisSim:
    rng = rng or default_rng()
    sim = MetropolisSim(phi0, mass2=mass2, lamb=lamb, width=width, rng=rng)
    sim.run_sim(n_steps, logger=logger, progress_bar=True)
    return sim.accepted_history