use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use rand_distr::{Normal, Distribution};
use crate::physics::{hamiltonian, gradient_phi4_action, BoundaryCondition};
use crate::integrators::leapfrog;

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let l = 4;
    let mass2 = -0.5;
    let lambda = 5.0;
    let eps = 0.1;

    // Initial phi
    let normal = Normal::new(0.0, 1.0).unwrap();
    let phi: Vec<f64> = (0..l).map(|_| normal.sample(&mut rng)).collect();

    // Initial momentum
    let pi: Vec<f64> = (0..l).map(|_| normal.sample(&mut rng)).collect();

    // Gradient
    let mut grad = vec![0.0; l];
    gradient_phi4_action(&phi, &mut grad, l, mass2, lambda, BoundaryCondition::PBC);
    println!("Rust grad: {:?}", grad);

    // One leapfrog step
    let mut phi_new = phi.clone();
    let mut pi_new = pi.clone();
    leapfrog(&mut phi_new, &mut pi_new, eps, |phi, grad, params| {
        gradient_phi4_action(phi, grad, params.L, params.mass2, params.lambda, BoundaryCondition::PBC)
    }, &GradParams { L: l, mass2, lambda });

    println!("Rust phi_new: {:?}", phi_new);
    println!("Rust pi_new: {:?}", pi_new);

    // Hamiltonians
    let h_old = hamiltonian(phi.clone(), pi.clone(), l, mass2, lambda, BoundaryCondition::PBC);
    let h_new = hamiltonian(phi_new.clone(), pi_new.clone(), l, mass2, lambda, BoundaryCondition::PBC);
    println!("Rust H_old: {:?}", h_old);
    println!("Rust H_new: {:?}", h_new);

    // Acceptance probability
    let p_accept = (h_old - h_new).exp().min(1.0);
    println!("Rust p_accept: {:?}", p_accept);
}
