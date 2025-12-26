use crate::physics::BoundaryCondition;

pub struct GradParams {
    pub L: usize,
    pub mass2: f64,
    pub lambda: f64,
    pub bc: BoundaryCondition,
}

pub type GradAction = fn(phi: &[f64], grad: &mut [f64], params: &GradParams); //create our own type

pub fn omelyan2(
    phi: &mut [f64],
    pi: &mut [f64],
    eps: f64,
    grad_action: GradAction,
    params: &GradParams,
) {
    let n = phi.len();
    let mut grad = vec![0.0; n];
    let lambda = 0.1931833; //hardcoded (different from Python, but works so far)

    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= lambda * eps * grad[i];
    }
    for i in 0..n {
        phi[i] += 0.5 * eps * pi[i];
    }

    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= (1.0 - 2.0 * lambda) * eps * grad[i];
    }
    for i in 0..n {
        phi[i] += 0.5 * eps * pi[i];
    }

    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= lambda * eps * grad[i];
    }
}


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
