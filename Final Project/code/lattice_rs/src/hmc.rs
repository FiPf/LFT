// src/hmc.rs
use pyo3::prelude::*;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use rand::Rng;
use rand_distr::{Normal, Distribution};

use crate::integrators::{leapfrog, omelyan2, GradParams};
use crate::physics::{hamiltonian, gradient_phi4_action, BoundaryCondition};

/// Wrapper to match GradAction type
fn grad_action_wrapper(phi: &[f64], grad: &mut [f64], params: &GradParams) {
    gradient_phi4_action(phi, grad, params.L, params.mass2, params.lambda, params.bc);
}

#[derive(Clone, Copy)]
enum IntegratorChoice {
    Leapfrog,
    Omelyan,
}

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

    rng: Pcg64,
    #[pyo3(get, set)]
    pub accepted_history: Vec<u8>,
    integrator_choice: IntegratorChoice,
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
        let rng = Pcg64::seed_from_u64(42);

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

    /// Perform a single HMC update
    /// Optional `pi_override` allows using predefined momenta (for testing)
    #[pyo3(signature = (pi_override = None))]
    fn update(&mut self, pi_override: Option<Vec<f64>>) {
        let n = self.phi.len();
        let mut pi: Vec<f64> = if let Some(p) = pi_override {
            p
        } else {
            let normal = Normal::new(0.0, 1.0).unwrap();
            self.phi.iter().map(|_| normal.sample(&mut self.rng)).collect()
        };

        let mut phi_new = self.phi.clone();
        let mut pi_new = pi.clone();

        let params = GradParams {
            L: self.L,
            mass2: self.mass2,
            lambda: self.lambda_,
            bc: BoundaryCondition::PBC,
        };

        // Integrator step(s) in-place
        for _ in 0..self.steps {
            match self.integrator_choice {
                IntegratorChoice::Leapfrog => {
                    leapfrog(&mut phi_new, &mut pi_new, self.eps, grad_action_wrapper, &params);
                }
                IntegratorChoice::Omelyan => {
                    omelyan2(&mut phi_new, &mut pi_new, self.eps, grad_action_wrapper, &params);
                }
            }
        }

        let h_old = hamiltonian(self.phi.clone(), pi.clone(), self.L, self.mass2, self.lambda_, params.bc);
        let h_new = hamiltonian(phi_new.clone(), pi_new.clone(), self.L, self.mass2, self.lambda_, params.bc);

        let p_accept = (h_old - h_new).exp().min(1.0);
        if self.rng.r#gen::<f64>() <= p_accept {
            self.phi = phi_new;
            self.accepted_history.push(1);
        } else {
            self.accepted_history.push(0);
        }
    }
}
