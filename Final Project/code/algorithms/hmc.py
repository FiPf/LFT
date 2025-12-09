import numpy as np
import physics
from base import Base_sim
from numpy.random import Generator, default_rng
from typing import Optional, Callable
import physics.actions as actions
import integrators

class HMCSim(Base_sim):
    def __init__(self, initial_phi: np.array, mass2: float, lamb: float, width: float, rng: Generator, integrator: Callable):
        super().__init__()
        self.phi = initial_phi.copy()
        self.mass2 = float(mass2)
        self.lamb = float(lamb)
        self.width = float(width)
        self.rng = rng
        self.accepted_history = []
        self.integrator = integrator

    def update(self) -> None:
        pi0 = self.rng.normal(loc=0.0, scale=1.0, size=self.lamb)
        proposed_phi = self.phi.copy()
        proposed_pi = pi0.copy()

        for i in range(self.steps): 
            proposed_phi, proposed_pi = self.integrator(proposed_phi, proposed_pi, self.eps)
        
        H_old = actions.hamiltonian(self.phi, pi0, mass2=self.mass2, lamb=self.lamb)
        H_new = actions.hamiltonian(proposed_phi, pi0, mass2=self.mass2, lamb=self.lamb)
        p_accept = np.min(1, np.exp(H_old - H_new))
        if self.rng.random() <= p_accept: 
            self.phi = proposed_phi
            self.accepted_history.append(1)
        else: 
            self.accepted_history.append(0)

def run_hmc(phi0: np.array, mass2: float, lamb: float, width: float, n_steps: int, rng: Optional[Generator] = None, logger: Optional[Callable] = None, progress: bool = False, integrator: Callable = integrators.leapfrog) -> HMCSim:
    rng = rng or default_rng()
    sim = HMCSim(phi0, mass2=mass2, lamb=lamb, width=width, rng=rng, integrator=integrator)
    sim.run_sim(n_steps, logger=logger, progress_bar=True)
    return sim.accepted_history