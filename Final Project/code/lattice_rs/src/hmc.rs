use rand::Rng;
use crate::integrator::Integrator; 
use crate::base::BaseSim;
use crate::physics::hamiltonian;

pub struct HMCSim<R, I>
where 
    R: Rng, 
    I: Integrator, 
{
    pub phi: Vec<f64>, //no longer a np.array

    pub mass2: f64, 
    pub lambda: f64, 
    pub eps: f64, 

    pub rng: R, 

    pub accepted_history: Vec<u8>, 
    pub integrator: I, //create your own type
    pub steps: usize, 
    pub bc: BoundaryCondition, 
}

impl<R, I> BaseSim for HCMSim<R, I>
where
    R: Rng, 
    I: Integrator, 
{
    fn update(&mut self) {
        let mut pi: Vec<f64> = self.phi.iter().map(|_| self.rng.sample(rand::Distributions::StandardNormal)).collect();  // sample pi (initial momentum) from standard normal distribution (var = 1, mean = 0)

        let mut proposed_phi = self.phi.clone(); 
        let mut proposed_pi = pi.clone(); 

        for _ in 0..self.steps { //call the integrator (leapfrog or omelayan2 or maybe later some other)
            self.integrator.step(
                &mut proposed_phi,
                &mut proposed_pi,
                self.eps,
                self.mass2,
                self.lambda,
                self.bc,
            );
        }

        let h_old = hamiltonian(&self.phi, &pi, self.mass2, self.lambda, self.bc);
        let h_new = hamiltonian(&proposed_phi, &proposed_pi, self.mass2, self.lambda, self.bc);

        let p_accept = (h_old - h_new).exp().min(1.0);

        if self.rng.gen_range::<f64>() <= p_accept {
            self.phi = proposed_phi; 
            self.accepted_history.push(1);
        }else{
            self.accepted_history.push(0); 
        }
    }
    fn get_steps(&self) -> usize{
        self.steps
    }
}

//free run simulation function 
pub fn run_hmc<R, I>(
    phi0: Vec<f64>,
    mass2: f64,
    lambda: f64,
    width: f64,
    n_steps: usize,
    rng: R,
    integrator: I,
    logger: Option<impl FnMut(&HMCSim<R, I>)>,
) -> Vec<u8>
where
    R: Rng,
    I: Integrator,
{
    let mut sim = HMCSim {
        phi: phi0,
        mass2,
        lambda,
        eps: width,
        rng,
        accepted_history: Vec::new(),
        integrator,
        steps: 100,
        bc: BoundaryCondition::PBC,
    };

    // Run simulation
    sim.run_sim(n_steps, logger);

    // return acceptance history
    sim.accepted_history
}
