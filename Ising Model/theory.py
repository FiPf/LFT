import numpy as np
from ising_model_physics import IsingModel
from base_simulation import Base_sim

# 1 dimension analytic results (formulas from script)
def partition_function_1d(ising_model: IsingModel)-> float:
    beta = ising_model.beta
    J = ising_model.J
    L = ising_model.lattice.size
    Z = (2*np.cosh(beta*J))**2 + (2*np.sinh(beta*J))**L
    return Z

def spin_correlation_1d(ising_model: IsingModel, i: int, j: int) -> float:
    beta = ising_model.beta
    J = ising_model.J
    L = ising_model.lattice.size
    Z = partition_function_1d(ising_model)
    term1 = (2*np.cosh(beta*J)) 
    term2 = (2*np.sinh(beta*J))
    n = abs(ising_model.lattice[i] -ising_model.lattice[j])
    
    corr = term1**(L - n)*term2**(n) + term1**(L-n)*term2**n
    corr /= Z
    
    return corr

def susceptibilty_1d(ising_model: IsingModel):
    beta = ising_model.beta
    J = ising_model.J
    L = ising_model.lattice.size
    
    term1 = (1 - np.tanh(beta*J)**L)
    term2 = (1 + np.tanh(beta*J)**L)
    
    return (term1/term2)*np.exp(2*beta*J)

