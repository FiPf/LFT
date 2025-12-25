use pyo3::prelude::*;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use rand::Rng;

use crate::integrators::{leapfrog, omelyan2, GradParams, GradAction};
use crate::physics::{hamiltonian, BoundaryCondition};

#[pyclass]
pub struct HMCSim {
    #[pyo3(get, set)]
    pub phi: Vec<f64>,
    #[pyo3(get, set)]
    pub L: usize,
    #[pyo3(get, set)]
    pub mass2: f64,
    #[pyo3(get, set)]
    pub lambda_: f64,
    #[pyo3(get, set)]
    pub eps: f64,
    #[pyo3(get, set)]
    pub steps: usize,

    rng: ChaCha20Rng,
    #[pyo3(get, set)]
    pub accepted_history: Vec<u8>,
    integrator_choice: IntegratorChoice,
}

#[derive(Clone, Copy)]
enum IntegratorChoice {
    Leapfrog,
    Omelyan,
}

#[pymethods]
impl HMCSim {
    #[new]
    #[pyo3(signature = (phi0, L, mass2, lambda_, eps, steps, integrator="leapfrog"))]
    fn new(
        phi0: Vec<f64>,
        L: usize,
        mass2: f64,
        lambda_: f64,
        eps: f64,
        steps: usize,
        integrator: &str,
    ) -> PyResult<Self> {
        let mut trng = rand::thread_rng();
        let rng = ChaCha20Rng::from_rng(&mut trng);

        let integrator_choice = match integrator {
            "leapfrog" => IntegratorChoice::Leapfrog,
            "omelyan2" => IntegratorChoice::Omelyan,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                "Integrator must be 'leapfrog' or 'omelyan2'",
            )),
        };

        Ok(Self {
            phi: phi0,
            L,
            mass2,
            lambda_,
            eps,
            steps,
            rng,
            accepted_history: Vec::new(),
            integrator_choice,
        })
    }

    fn update(&mut self) {
        // Fix: Use random() instead of gen() to avoid deprecation warning
        let mut pi: Vec<f64> = self.phi.iter().map(|_| self.rng.random::<f64>()).collect();
        let mut phi_new = self.phi.clone();
        let mut pi_new = pi.clone();

        let params = GradParams {
            mass2: self.mass2,
            lambda: self.lambda_,
        };

        // Define grad_action functions with correct signatures
        let grad_action = |phi: &[f64], pi: &mut [f64], params: &GradParams| {
            // You'll need to implement the gradient calculation here
            // This is a placeholder - you need to replace it with your actual gradient
            for (i, p) in pi.iter_mut().enumerate() {
                // Example: simple gradient calculation
                // This should be your actual gradient of the action
                let grad = params.mass2 * phi[i] + params.lambda * phi[i].powi(3);
                *p -= grad;
            }
        };

        for _ in 0..self.steps {
            match self.integrator_choice {
                IntegratorChoice::Leapfrog => {
                    // Fix: Pass grad_action function reference, not the integrator itself
                    leapfrog(
                        &mut phi_new, 
                        &mut pi_new, 
                        self.eps, 
                        grad_action, 
                        &params
                    );
                }
                IntegratorChoice::Omelyan => {
                    // Fix: Pass grad_action function reference
                    omelyan2(
                        &mut phi_new, 
                        &mut pi_new, 
                        self.eps, 
                        grad_action, 
                        &params, 
                        self.lambda_
                    );
                }
            }
        }

        let h_old = hamiltonian(
            self.phi.clone(),
            pi,
            self.L,
            self.mass2,
            self.lambda_,
            BoundaryCondition::PBC,
        );

        let h_new = hamiltonian(
            phi_new.clone(),
            pi_new,
            self.L,
            self.mass2,
            self.lambda_,
            BoundaryCondition::PBC,
        );

        let p_accept = (h_old - h_new).exp().min(1.0);

        if self.rng.random::<f64>() < p_accept {
            self.phi = phi_new;
            self.accepted_history.push(1);
        } else {
            self.accepted_history.push(0);
        }
    }

   
}