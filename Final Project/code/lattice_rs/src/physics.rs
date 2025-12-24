use pyo3::prelude::*; 
use pyo3::{pyclass, pymethods, pyfunction}; 


//periodic or aperiodic boundary conditions
#[derive(Clone, Copy)]
#[derive(PartialEq)]
#[pyclass]
pub enum BoundaryCondition {
    PBC,
    APBC, 
}

//this replaces np.roll
#[inline(always)]
fn roll(i: isize, l: usize) -> usize {
    ((i + l as isize) % l as isize) as usize
}

#[pyfunction]
pub fn phi4_action(
    mut phi: Vec<f64>, // take ownership instead of &mut
    L: usize,
    mass2: f64,
    lambda: f64,
    bc: BoundaryCondition,
) -> f64 {
    let mut action = 0.0;

    for &x in &phi { // iterate immutably over the owned Vec
        action += (2.0 + 0.5 * mass2) * x * x + (lambda / 24.0) * x.powi(4);
    }

    for i in 0..L {
        for j in 0..L {
            let idx0 = i * L + j;
            let phi_ij = phi[idx0];

            let i_fwd = roll(i as isize + 1, L);
            let mut phi_fwd = phi[i_fwd * L + j];

            // APBC sign change at boundary
            if bc == BoundaryCondition::APBC && i == L - 1 {
                phi_fwd *= -1.0;
            }

            action -= phi_ij * phi_fwd;

            // mu = 1 (space direction)
            let j_fwd = roll(j as isize + 1, L);
            action -= phi_ij * phi[i * L + j_fwd];
        }
    }

    action
}

#[pyfunction]
pub fn gradient_phi4_action(
    phi: Vec<f64>,          // take ownership
    mut grad: Vec<f64>,     // take ownership for mutation
    L: usize,
    mass2: f64,
    lambda: f64,
    bc: BoundaryCondition,
) {
    for i in 0..L {
        for j in 0..L {
            let idx0 = i * L + j;
            let phi_ij = phi[idx0];

            let i_fwd = roll(i as isize + 1, L);
            let j_fwd = roll(j as isize + 1, L);

            let mut phi_fwd_i = phi[i_fwd * L + j];
            let mut phi_fwd_j = phi[i * L + j_fwd];

            if bc == BoundaryCondition::APBC && i == L - 1 {
                phi_fwd_i *= -1.0;
            }

            grad[idx0] = (2.0 + 0.5 * mass2) * phi_ij + (lambda / 6.0) * phi_ij.powi(3)
                         - phi_fwd_i - phi_fwd_j;
        }
    }
}

#[inline(always)]
pub fn kinetic_energy(pi: &[f64]) -> f64 {
    pi.iter().map(|x| x * x).sum::<f64>() * 0.5
}

#[pyfunction]
pub fn hamiltonian(
    phi: Vec<f64>,
    pi: Vec<f64>,
    L: usize,
    mass2: f64,
    lambda: f64,
    bc: BoundaryCondition,
) -> f64 {
    kinetic_energy(&pi) + phi4_action(phi, L, mass2, lambda, bc)
}
