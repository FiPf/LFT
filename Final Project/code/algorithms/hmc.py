import numpy as np
import physics
from algorithms.base import Base_sim
from numpy.random import Generator, default_rng
from typing import Optional, Callable
import physics.actions as actions
import algorithms.integrators as integrators
from physics.actions import BoundaryCondition

class HMCSim(Base_sim):
    """class to run the simulation with the HMC (Hybrid Monte Carlo) algorithm

    Args:
        Base_sim (abstract base class): HMCSim inherits from an abstract base class, which forces a common structure to other algorithms. 
    """
    def __init__(self, initial_phi: np.array, mass2: float, lamb: float, width: float, rng: Generator, integrator: Callable, bc: BoundaryCondition = BoundaryCondition.PBC):
        """initialize a HMC simulation

        Args:
            initial_phi (np.array): initial configuration
            mass2 (float): mass parameter, can be any real number (also negative)
            lamb (float): coupling strength of phi^4 theory, must be positive
            width (float): parameter to tune the acceptance rate, used to sample the configuration space
            rng (Generator): random number generator
            integrator (Callable): which integrator function to use
            bc (BoundaryCondition, optional): What kind of boundary conditions to use (period = PBC, aperiodic = APBC). Defaults to BoundaryCondition.PBC.
        """
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
        self.hamiltonian_diffs = []

    def update(self) -> list:
        """performs a single HMC algorithm step
        """
        pi0 = self.rng.normal(size=self.phi.shape)
        proposed_phi = self.phi.copy()
        proposed_pi = pi0.copy()
        diff_list = []

        for _ in range(self.steps): 
            proposed_phi, proposed_pi = self.integrator(proposed_phi, proposed_pi, self.eps, grad_action=self.grad_action, grad_kwargs=self.grad_kwargs)
        
        H_old = actions.hamiltonian(self.phi, pi0, mass2=self.mass2, lamb=self.lamb, bc = self.bc)
        H_new = actions.hamiltonian(proposed_phi, proposed_pi, mass2=self.mass2, lamb=self.lamb, bc = self.bc)
        p_accept = min(1, np.exp(H_old - H_new))
        diff_list.append(H_old - H_new)
        if self.rng.random() <= p_accept: 
            self.phi = proposed_phi
            self.accepted_history.append(1)
        else: 
            self.accepted_history.append(0)

        delta_H = H_old - H_new
        self.hamiltonian_diffs.append(delta_H)

def run_hmc(phi0: np.array, mass2: float, lamb: float, width: float, n_steps: int, rng: Optional[Generator] = None, logger: Optional[Callable] = None, progress: bool = False, integrator: Callable = integrators.leapfrog, return_diff: bool = False) -> HMCSim:
    """function to run the HMC simulation, a HMCSim object is initialized and the simulation is run with the desired parameters. 

    Args:
        phi0 (np.array): initial configuration
        mass2 (float): mass parameter, can be any real number (also negative)
        lamb (float): coupling strength of phi^4 theory, must be positive
        width (float): parameter to tune the acceptance rate, used to sample the configuration space
        n_steps (int): number of steps to run the simulation
        rng (Optional[Generator], optional): random number generator. Defaults to None.
        logger (Optional[Callable], optional): Logger function to use. Defaults to None.
        progress (bool, optional): Whether to display a progress bar or not. Defaults to False.
        integrator (Callable, optional): Which integrator to use. Defaults to integrators.leapfrog.
        return_diff (bool, optional): Whether to return the hamiltonian difference or not, used to check for energy conservation.

    Returns:
        HMCSim: A HMCSim object, the accepted history is returned
    """
    rng = rng or default_rng()
    sim = HMCSim(phi0, mass2=mass2, lamb=lamb, width=width, rng=rng, integrator=integrator)
    sim.run_sim(n_steps, logger=logger, progress_bar=True)
    if return_diff:
        return np.array(sim.hamiltonian_diffs)
    else:
        return sim.accepted_history