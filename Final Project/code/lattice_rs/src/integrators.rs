use crate::physics::BoundaryCondition; 

pub trait Integrator {
    fn step(
        &self,
        phi: &mut [f64],
        pi: &mut [f64],
        eps: f64,
        mass2: f64,
        lamb: f64,
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
    pub lamb: f64,
}

//omelyan2
pub fn omelyan2(
    phi: &mut [f64], //instead of np.arrays, we use &mut [f64]
    pi: &mut [f64],
    eps: f64,
    grad_action: GradAction,
    params: &GradParams,
    lambda: f64, //lambda instead of lamb since rust already has this keyword
) {
    let n = phi.len();

    let mut grad = vec![0.0; n]; // Temporary buffer for gradients

    for i in 0..n {
        pi[i] -= lambda * eps * phi[i];
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