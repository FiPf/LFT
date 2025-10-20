import numpy as np
from base_simulation import Base_sim
from abc import ABC
from base_simulation import Base_sim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class IsingModel(Base_sim, ABC): 
    MU = 1.0 

    def __init__(self, shape, B: float, J: float, beta: float):
        super().__init__()
        assert J >= 0 #ferromagnetic coupling constant must be larger than 1
        self.J = J #ferromagnetic coupling constant
        self.B = B #external magnetic field
        self.beta = beta #inverse temperature
        self.lattice = np.random.choice([-1.0, 1.0], size=shape)

    def hamiltonian(self): 
        return -self.J * self.neighbor_spin_corr() - self.MU * self.B* self.magnetization()

    def magnetization(self): 
        return np.sum(self.lattice)
    
    def neighbor_spin_corr(self): 
        total = 0

        for axis in range(self.lattice.ndim): #loop over all dimensions
            if self.lattice.shape[axis] > 1: # Skip axes with size 1
                shifted = np.roll(self.lattice, shift=1, axis=axis)
                total += np.sum(self.lattice * shifted)

        return total
    
    def get_lattice(self): 
        return self.lattice.copy()
    
    def flip_spin(self, index): 
        assert self.lattice.ndim == len(index)
        self.lattice[*index] *= -1
    
    def neighbours(self, index):
        for i in range(self.lattice.ndim): #loop over all dimensions
            unit_vec = np.eye(1, self.lattice.ndim, k = i, dtype = int) [0] # create a unit vector along the current axis in n dimensions

            forward = (index + unit_vec) % self.lattice.shape #neighbor one step forward along this axis, apply modula for period boundaries
            yield forward #produces one value and pauses the function, when the function is called in a loop, it can resume from that point in the next step / later

            backward = (index - unit_vec) % self.lattice.shape
            yield backward 

    def get_steps(self):
        return super().get_steps()
    
    def get_size(self): 
        return self.lattice.size
    
    def plot_lattice(self, ax=None):
        """
        Plot the Ising model lattice as a heatmap.

        Args:
            ax (matplotlib.axes.Axes, optional): The axis object to plot on. If None, the current axis is used.

        Raises:
            Exception: If the lattice is not 1D or 2D.
        """
        if ax is None:
            ax = plt.gca()

        match self.lattice.ndim:
            case 1:
                ax.imshow(self.lattice[None, :], cmap='hot')
            case 2:
                ax.imshow(self.lattice, cmap='hot')
            case 3:
                ax.set_box_aspect((1,1,1))
                ax.voxels(self.lattice == 1)
            case _:
                raise Exception('plot lattice is only defined for dimensions 1, 2 and 3')
        ax.set_title("Ising Model Lattice")

    def animate(self, total_steps, update_interval=200, fps=15, logger = None):
        """
        Animate the evolution of the Ising model lattice over time.

        Args:
            total_steps (int): Total number of simulation steps.
            update_interval (int, optional): Number of steps between each frame in the animation. Defaults to 200.
            fps (int, optional): Frames per second for the animation. Defaults to 15.

        Returns:
            IPython.display.HTML: An HTML object of the animation, suitable for display in Jupyter notebooks.
        """
        ax = None
        fig = None
        match self.lattice.ndim:
            case 1 | 2:
                fig, ax = plt.subplots()
            case 3:
                fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            case _:
                raise Exception('plot lattice is only defined for dimensions 1, 2 and 3')
                

        def update_plot(frame):
            ax.clear()
            self.plot_lattice(ax)
            ax.set_title(rf"Step: {frame * update_interval}, $\mathcal{{H}} = {self.hamiltonian():.1f}$")
            self.run_sim(update_interval, logger = logger)  # Update the lattice for each frame

        ani = FuncAnimation(fig, update_plot, frames=total_steps // update_interval,
                            interval=1000 // fps)
        animation = HTML(ani.to_jshtml())
        plt.close(fig)
        return animation