from abc import ABC, abstractmethod
from typing import Callable
from tqdm import trange

class Base_sim(ABC): 
    """Abstract base class for a general simulation. Subclasses must implement the update function. 

    Args:
        ABC (abstract base class): https://docs.python.org/3/library/abc.html
    """
    def __init__(self):
        """initialize the simulation, set the internal step counter to zero. 
        """
        self.steps = 0

    @abstractmethod
    def update(self): 
        """Perform a single update step for the simulation, increment the step counter. 

        Raises:
            NotImplementedError: Raises an error if update method is not implemented. 
        """
        self.steps += 1
        raise NotImplementedError

    def get_steps(self) -> int: 
        """get the number of steps

        Returns:
            int: number of steps at the moment
        """
        return self.steps
    
    def run_sim(self, steps: int, logger: Callable, progress_bar: bool = False): 
        """run the simulation for a certain number of steps

        Args:
            steps (int): number of steps to run the simulation
            logger (Callable): logger function, optional
            progress_bar (bool, optional): Whether to display a progress bar or not. Defaults to False.
        """
        for _ in trange(steps, desc = 'running simulation') if progress_bar else range(steps):
                self.update()
                if logger:
                    logger(self)
                    
                