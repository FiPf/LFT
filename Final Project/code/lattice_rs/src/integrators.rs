use crate::physics::BoundaryCondition; 

pub trait Integrator {
    fn step(
        &self,
        phi: &mut [f64],
        pi: &mut [f64],
        eps: f64,
        mass2: f64,
        lambda: f64,
        bc: BoundaryCondition,
    );
}

pub type GradAction = fn(phi: &[f64], out: &mut [f64], params: &GradParams);
//new type for the gradient of the action
//similar to Python Callable
//params replaces **grad_kwargs

//this struct replaces grad_kwargs
#[derive(Clone, Copy)]
pub struct GradParams {
    pub mass2: f64,
    pub lambda: f64,
}

//omelyan2
pub fn omelyan2(
    phi: &mut [f64],
    pi: &mut [f64],
    eps: f64,
    grad_action: GradAction,
    params: &GradParams,
    lambda: f64,
) {
    let n = phi.len();
    let mut grad = vec![0.0; n];

    // Step 1
    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= lambda * eps * grad[i];
    }

    // Step 2
    for i in 0..n {
        phi[i] += 0.5 * eps * pi[i];
    }

    // Step 3
    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= (1.0 - 2.0 * lambda) * eps * grad[i];
    }

    // Step 4
    for i in 0..n {
        phi[i] += 0.5 * eps * pi[i];
    }

    // Step 5
    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= lambda * eps * grad[i];
    }
}



//leapfrog
pub fn leapfrog(
    phi: &mut [f64],
    pi: &mut [f64],
    eps: f64,
    grad_action: GradAction,
    params: &GradParams,
) {
    let n = phi.len();
    let mut grad = vec![0.0; n];

    // Half-step momentum update
    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= 0.5 * eps * grad[i]; // subtract gradient
    }

    // Full-step position update
    for i in 0..n {
        phi[i] += eps * pi[i];
    }

    // Half-step momentum update
    grad_action(phi, &mut grad, params);
    for i in 0..n {
        pi[i] -= 0.5 * eps * grad[i]; // subtract gradient
    }
}


