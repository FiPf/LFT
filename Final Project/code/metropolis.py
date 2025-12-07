import numpy as np
import physics

def propose_phi(phi: np.ndarray,
                width: float,
                rng: np.random.Generator) -> np.ndarray:
    delta = 2. * rng.random(size=phi.shape) - 1.  # 2D array with random numbers between -1 and 1.
    return phi + width * delta


def metropolis_step(phi: np.ndarray,
                    mass2: float,
                    lamb: float,
                    width: float,
                    rng: np.random.Generator):
    # 1. Proposal
    proposed_phi = propose_phi(phi, width, rng)

    # 2. Acceptance probability.
    current_action = physics.phi4_action(phi, mass2=mass2, lamb=lamb)
    proposed_action = physics.phi4_action(proposed_phi, mass2=mass2, lamb=lamb)
    p_acceptance = np.min([1.0, np.exp(current_action - proposed_action)])

    # 3. Accept / reject
    r = rng.random()

    if r <= p_acceptance:  # Accept.
        return proposed_phi, 1

    # Reject.
    return phi, 0


def sample_field(initial_phi: np.ndarray,
                 num_samples: int,
                 mass2: float,
                 lamb: float,
                 width: float,
                 rng: np.random.Generator):
    chain = [initial_phi]
    acceptance = []

    for _ in range(num_samples):
        phi, accepted = metropolis_step(phi=chain[-1], mass2=mass2, lamb=lamb, width=width, rng=rng)
        chain.append(phi)
        acceptance.append(accepted)

    return chain, acceptance