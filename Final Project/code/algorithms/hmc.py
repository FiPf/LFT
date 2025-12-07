import numpy as np
import physics
from base import Base_sim
from numpy.random import Generator, default_rng
from typing import Optional, Callable

class HMCSim(Base_sim):
    def __init__(self, initial_phi: np.array, mass2: float, lamb: float, width: float, rng: Generator):
        super().__init__()
        self.phi = initial_phi.copy()
        self.mass2 = float(mass2)
        self.lamb = float(lamb)
        self.width = float(width)
        self.rng = rng
        self.accepted = 0

    def update(self):
        return super().update()
    
def run_hmc(): 
    pass