import numpy as np
import physics
from base import Base_sim
from numpy.random import Generator, default_rng
from typing import Optional, Callable

def propose_phi(phi: np.ndarray,
                width: float,
                rng: np.random.Generator) -> np.ndarray:
    delta = 2. * rng.random(size=phi.shape) - 1.  # 2D array with random numbers between -1 and 1.
    return phi + width * delta

class MetropolisSim(Base_sim): 
    def __init__(self, initial_phi: np.array, mass2: float, lamb: float, width: float, rng: Generator):
        super().__init__()
        self.phi = initial_phi.copy()
        self.mass2 = float(mass2)
        self.lamb = float(lamb)
        self.width = float(width)
        self.rng = rng
        self.accepted = 0

    def update(self) -> None:
        proposed_phi = propose_phi(self.phi, self.width, self.rng)

        current_action = physics.phi4_action(self.phi, mass2=self.mass2, lamb=self.lamb)
        proposed_action = physics.phi4_action(proposed_phi, mass2=self.mass2, lamb=self.lamb)
        p_acceptance = np.min([1.0, np.exp(current_action - proposed_action)])

        r = self.rng.random()

        if r <= p_acceptance:  # Accept.
            self.phi = proposed_action
            self.accepted += 1

def run_metropolis(phi0: np.array, mass2: float, lamb: float, width: float, n_steps: int, rng: Optional[Generator] = None, logger: Optional[Callable] = None, progress: bool = False) -> MetropolisSim:
    rng = rng or default_rng()
    sim = MetropolisSim(phi0, mass2=mass2, lamb=lamb, width=width, rng=rng)
    sim.run(n_steps, logger=logger, progress=progress)
    return sim