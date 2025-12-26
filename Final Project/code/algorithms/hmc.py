import numpy as np
import physics
from algorithms.base import Base_sim
from numpy.random import Generator, default_rng
from typing import Optional, Callable
import physics.actions as actions
import algorithms.integrators as integrators
from physics.actions import BoundaryCondition

class HMCSim(Base_sim):
    def __init__(self, initial_phi: np.array, mass2: float, lamb: float, width: float, rng: Generator, integrator: Callable, bc: BoundaryCondition = BoundaryCondition.PBC):
        super().__init__()
        self.phi = initial_phi.copy()
        self.mass2 = float(mass2)
        self.lamb = float(lamb)
        self.eps = float(width)
        self.rng = np.random.Generator(np.random.PCG64(42))
        self.accepted_history = []
        self.integrator = integrator
        self.steps = 100
        self.bc = bc
        self.grad_action = actions.gradient_phi4_action
        self.grad_kwargs = dict(mass2=self.mass2, lamb=self.lamb, bc = self.bc)

    def update(self) -> None:
        pi0 = self.rng.normal(size=self.phi.shape)
        proposed_phi = self.phi.copy()
        proposed_pi = pi0.copy()

        for _ in range(self.steps): 
            proposed_phi, proposed_pi = self.integrator(proposed_phi, proposed_pi, self.eps, grad_action=self.grad_action, grad_kwargs=self.grad_kwargs)
        
        H_old = actions.hamiltonian(self.phi, pi0, mass2=self.mass2, lamb=self.lamb, bc = self.bc)
        H_new = actions.hamiltonian(proposed_phi, proposed_pi, mass2=self.mass2, lamb=self.lamb, bc = self.bc)
        p_accept = min(1, np.exp(H_old - H_new))
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