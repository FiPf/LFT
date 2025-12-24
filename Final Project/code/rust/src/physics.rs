//periodic or aperiodic boundary conditions
#[derive(Clone, Copy)]
pub enum BoundaryCondition {
    PBC,
    APBC, 
}

//this replaces np.roll
#[inline(always)]
fn roll(i: isize, l: usize) -> usize {
    ((i + l as isize) % l as isize) as usize
}

pub fn phi4_action(phi: &[f64], L: usize, mass2: f64, lambda: f64, bc: BoundaryCondition) -> f64 {
    let mut action = 0.0; 

    for &x in phi {
        action += (2.0 + 0.5 *mass2)*x*x + (lambda/24.0)*x.powi(4); 
    }

    for i in 0..L{
        for j in 0..L{
            let idx0 = i*L + j; 
            let phi_ij = phi[idx0]; 

            let mut i_fwd = idx(i as isize + 1, L);
            let mut phi_fwd = phi[i_fwd * L + j];

            // APBC sign change at boundary
            if bc == BoundaryCondition::APBC && i == L - 1 {
                phi_fwd *= -1.0;
            }

            action -= phi_ij * phi_fwd;

            // mu = 1 (space direction)
            let j_fwd = idx(j as isize + 1, L);
            action -= phi_ij * phi[i * L + j_fwd];
        }
    }action

}

pub fn gradient_phi4_action(phi: &[f64], grad: &mut [f64], L: usize, mass2: f64, lambda: f64, bc: BoundaryCondition) {

    for (g, &x) in grad.iter_mut().zip(phi.iter()) {
        *g = (2.0 + 0.5 * mass2) * 2.0 * x + (lamb / 6.0) * x.powi(3);
    }

     for i in 0..L {
        for j in 0..L {
            let idx0 = i * L + j;

            // mu = 0
            let mut i_fwd = idx(i as isize + 1, L);
            let mut i_bwd = idx(i as isize - 1, L);

            let mut phi_fwd = phi[i_fwd * L + j];
            let mut phi_bwd = phi[i_bwd * L + j];

            if bc == BoundaryCondition::APBC {
                if i == L - 1 {
                    phi_fwd *= -1.0;
                }
                if i == 0 {
                    phi_bwd *= -1.0;
                }
            }

            grad[idx0] -= phi_fwd + phi_bwd;

            // mu = 1
            let j_fwd = idx(j as isize + 1, L);
            let j_bwd = idx(j as isize - 1, L);

            grad[idx0] -= phi[i * L + j_fwd];
            grad[idx0] -= phi[i * L + j_bwd]; 
        }
    }
}

#[inline(always)]
pub fn kinetic_energy(pi: &[f64]) -> f64 {
    pi.iter().map(|x| x * x).sum::<f64>() * 0.5
}

pub fn hamiltonian(phi: &[f64], pi: &[f64], L: usize, mass2: f64, lambda: f64, bc: BoundaryCondition) -> f64 {
    kinetic_energy(pi) + phi4_action(phi, L, mass2, lambda, bc)
}
