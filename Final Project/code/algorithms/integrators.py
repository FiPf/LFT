import numpy as np 
from typing import Callable
import physi.actions as actions

def omelyan2(phi: np.array, pi: np.array, eps: float, grad_action: Callable = None, lamb=0.1931833): 
    if grad_action is None: 
        grad_action = actions.gradient_phi4_action
    pi1 = pi - lamb * eps * (phi)
    phi1 = phi + 0.5 * eps * pi1
    pi2 = pi1 - (1 - 2*lamb) * eps * grad_action(phi1)
    phi2 = phi1 + 0.5 * eps * pi2
    pi3 = pi2 - lamb * eps * grad_action(phi2)
    return phi2, pi3
    
def leapfrog(phi: np.array, pi: np.array, eps: float, grad_action: Callable = None): 
    if grad_action is None: 
        grad_action = actions.gradient_phi4_action
    pi_half = pi -1/2*eps*grad_action(phi)
    phi_new = phi + eps * pi_half
    pi_new  = pi_half -1/2*eps*grad_action(phi_new)
    return phi_new, pi_new