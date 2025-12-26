use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use crate::integrators::GradParams;
use crate::physics::gradient_phi4_action;
mod physics;
mod metropolis;
mod base;
mod hmc;
mod integrators;

/// Python-friendly wrapper for the gradient function
#[pyfunction]
fn gradient_phi4_action_py(
    phi: Vec<f64>, 
    L: usize, 
    mass2: f64, 
    lambda: f64, 
    bc: physics::BoundaryCondition  // Accept the enum directly
) -> Vec<f64> {
    let mut grad = vec![0.0; phi.len()];
    physics::gradient_phi4_action(&phi, &mut grad, L, mass2, lambda, bc);
    grad
}
// Define wrapper locally
fn grad_action_wrapper(phi: &[f64], grad: &mut [f64], params: &GradParams) {
    gradient_phi4_action(phi, grad, params.L, params.mass2, params.lambda, params.bc);
}
/// Wrapper for hamiltonian function
#[pyfunction]
fn hamiltonian_py(
    phi: Vec<f64>,
    pi: Vec<f64>,
    L: usize,
    mass2: f64,
    lambda: f64,
    bc: physics::BoundaryCondition,
) -> f64 {
    physics::hamiltonian(phi, pi, L, mass2, lambda, bc)
}

#[pyfunction]
fn test_leapfrog(
    phi: Vec<f64>,
    pi: Vec<f64>,
    L: usize,
    mass2: f64,
    lambda: f64,
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    use crate::physics::BoundaryCondition;
    use crate::integrators::{leapfrog, GradParams};
    
    let params = GradParams {
        L,
        mass2,
        lambda,
        bc: BoundaryCondition::PBC,
    };
    
    let mut phi_mut = phi.clone();
    let mut pi_mut = pi.clone();
    
    leapfrog(&mut phi_mut, &mut pi_mut, eps, grad_action_wrapper, &params);
    
    (phi_mut, pi_mut)
}

#[pyfunction]
fn test_omelyan2(
    phi: Vec<f64>,
    pi: Vec<f64>,
    L: usize,
    mass2: f64,
    lambda: f64,
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    use crate::physics::BoundaryCondition;
    use crate::integrators::{omelyan2, GradParams};
    
    let params = GradParams {
        L,
        mass2,
        lambda,
        bc: BoundaryCondition::PBC,
    };
    
    let mut phi_mut = phi.clone();
    let mut pi_mut = pi.clone();
    
    omelyan2(&mut phi_mut, &mut pi_mut, eps, grad_action_wrapper, &params);
    
    (phi_mut, pi_mut)
}

/// Python module
#[pymodule]
fn lattice_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Export BoundaryCondition enum
    m.add_class::<physics::BoundaryCondition>()?;
    
    // Physics functions
    m.add_function(wrap_pyfunction!(physics::phi4_action, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_phi4_action_py, m)?)?;
    m.add_function(wrap_pyfunction!(hamiltonian_py, m)?)?;

    // Metropolis HMC
    m.add_class::<metropolis::MetropolisSim>()?;
    m.add_function(wrap_pyfunction!(metropolis::run_metropolis, m)?)?;

    // HMC
    m.add_class::<hmc::HMCSim>()?;

    m.add_function(wrap_pyfunction!(test_leapfrog, m)?)?;
    m.add_function(wrap_pyfunction!(test_omelyan2, m)?)?;

    Ok(())
}