import numpy as np
from ising_model_physics import IsingModel
from abc import ABC

class Metropolis(IsingModel):

    def __init__(self, shape, B: float, J: float, beta: float):
        super().__init__(shape, B, J, beta)

    def update(self): 
        super().update()
        index = np.random.randint(self.get_lattice().shape)
        delta_energy = self.delta_energy(index)

        if delta_energy <= 0: 
            self.flip_spin(index)

        elif np.random.rand() < np.exp(-self.beta * delta_energy): 
            self.flip_spin(index)

    def delta_energy(self, index): 
        external = -2*self.lattice[*index]*self.MU*self.B

        internal = 0

        for n in self.neighbours(index): 
            internal += self.lattice[*n]

        internal *= 2*self.J*self.lattice[*index]

        return internal + external 