use crate::physics::BoundaryCondition;

pub struct GradParams {
    pub L: usize,
    pub mass2: f64,
    pub lambda: f64,
    pub bc: BoundaryCondition,
}

pub type GradAction = fn(phi: &[f64], grad: &mut [f64], params: &GradParams);

// Omelyan2 integrator (in-place, matches Python)
/// Omelyan2 integrator (in-place, same style as leapfrog)
// In integrators.rs - Modified to work in-place
pub fn omelyan2(
    phi: &mut [f64],
    pi: &mut [f64],
    eps: f64,
    grad_action: GradAction,
    params: &GradParams,
) {
    let n = phi.len();
    let mut grad = vec![0.0; n];
    let lambda = 0.1931833;

    // Step 1
    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= lambda * eps * grad[i];
    }
    for i in 0..n {
        phi[i] += 0.5 * eps * pi[i];
    }

    // Step 2
    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= (1.0 - 2.0 * lambda) * eps * grad[i];
    }
    for i in 0..n {
        phi[i] += 0.5 * eps * pi[i];
    }

    // Step 3
    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= lambda * eps * grad[i];
    }
}



// Leapfrog integrator (already OK)
pub fn leapfrog(
    phi: &mut [f64],
    pi: &mut [f64],
    eps: f64,
    grad_action: GradAction,
    params: &GradParams,
) {
    let n = phi.len();
    let mut grad = vec![0.0; n];

    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= 0.5 * eps * grad[i];
    }

    for i in 0..n {
        phi[i] += eps * pi[i];
    }

    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= 0.5 * eps * grad[i];
    }
}
