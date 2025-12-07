from abc import ABC, abstractmethod
from typing import Callable
from tqdm import trange

class Base_sim(ABC): 
    @abstractmethod

    def __init__(self):
        self.steps = 0

    def update(self): 
        self.steps += 1

    def get_steps(self): 
        return self.steps
    
    def run_sim(self, steps, logger: Callable, progress_bar: bool = False): 
        for _ in trange(steps, desc = 'running simulation') if progress_bar else range(steps):
                self.update()
                if logger:
                    logger(self)
                    
                