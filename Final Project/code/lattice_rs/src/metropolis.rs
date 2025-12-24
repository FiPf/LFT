use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use crate::physics;
use crate::physics::BoundaryCondition;
use crate::base::{BaseSim, SimulationBase};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, pyfunction};


#[pyclass]
pub struct MetropolisSim {
    base: SimulationBase,
    pub phi: Vec<f64>,
    pub L: usize,
    pub mass2: f64,
    #[pyo3(get, set)]
    pub lambda_: f64,
    pub eps: f64,
    rng: ChaCha20Rng,
    pub accepted_history: Vec<u8>,
    pub bc: BoundaryCondition,
}

#[pymethods]
impl MetropolisSim {
    #[new]
    #[pyo3(signature = (phi0, L, mass2, lambda_, eps, bc = BoundaryCondition::PBC))]
    fn new(phi0: Vec<f64>, L: usize, mass2: f64, lambda_: f64, eps: f64, bc: BoundaryCondition) -> Self {
        let mut thread_rng = rand::thread_rng();
        let rng = ChaCha20Rng::from_rng(&mut thread_rng);

        MetropolisSim {
            base: SimulationBase::new(),
            phi: phi0,
            L,
            mass2,
            lambda_,
            eps,
            rng,
            accepted_history: Vec::new(),
            bc,
        }
    }

    fn propose_phi(&mut self) -> Vec<f64> {
        self.phi
            .iter()
            .map(|&x| x + self.eps * self.rng.gen_range(-1.0..1.0))
            .collect()
    }

    fn update(&mut self) {
        let proposed_phi = self.propose_phi();

        let current_action =
            physics::phi4_action(self.phi.clone(), self.L, self.mass2, self.lambda_, self.bc);
        let proposed_action =
            physics::phi4_action(proposed_phi.clone(), self.L, self.mass2, self.lambda_, self.bc);

        let p_accept = (current_action - proposed_action).exp().min(1.0);
        let r: f64 = self.rng.gen_range(0.0..1.0);

        if r <= p_accept {
            self.phi = proposed_phi;
            self.accepted_history.push(1);
        } else {
            self.accepted_history.push(0);
        }
    }

    fn get_steps(&self) -> usize {
        self.accepted_history.len()
    }
}

#[pyfunction]
pub fn run_metropolis(
    phi0: Vec<f64>,
    L: usize,
    mass2: f64,
    lambda_: f64,
    eps: f64,
    n_steps: usize,
    bc: BoundaryCondition,
) -> Vec<u8> {
    let mut sim = MetropolisSim::new(phi0, L, mass2, lambda_, eps, bc);

    for _ in 0..n_steps {
        sim.update();
    }

    sim.accepted_history
}