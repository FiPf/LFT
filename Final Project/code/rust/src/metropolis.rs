use rand::Rng;

pub struct MetropolisSim<R>
where 
    R: Rng, 
{
    pub phi: Vec<f64>, //no longer a np.array

    pub mass2: f64, 
    pub lamb: f64, 
    pub eps: f64, 

    pub rng: R, 

    pub accepted_history: Vec<u8>, 
    pub integrator: I, 
    pub steps: usize, 
    pub bc: BoundaryCondition, 
}

impl<R> BaseSim for MetropolisSim<R>
where
    R: Rng, 
{
    pub type ActionFn = fn(phi: &[f64], mass2: f64, lamb: f64, bc: BoundaryCondition) -> f64;
    
    pub fn propose_phi(&mut self) -> Vec<f64> {
        self.phi.iter().map(|&x|{
            let delta: f64 = 2.0*self.rng.gen::<f64>() - 1.0; 
            x + self.width*delta
        }).collect()
    }

    fn update(&mut self) {
        let proposed_phi = self.propose_phi();

        let current_action =
            (self.action)(&self.phi, self.mass2, self.lamb, self.bc);
        let proposed_action =
            (self.action)(&proposed_phi, self.mass2, self.lamb, self.bc);

        let p_accept = (current_action - proposed_action)
            .exp()
            .min(1.0);

        let r: f64 = self.rng.gen();// random number in [0,1)

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

pub fn run_metropolis<R>(
    phi0: Vec<f64>,
    mass2: f64,
    lamb: f64,
    width: f64,
    n_steps: usize,
    rng: R,
    action: ActionFn,
    logger: Option<impl FnMut(&MetropolisSim<R>)>,
    bc: BoundaryCondition,
) -> Vec<u8>
where
    R: Rng,
{
    let mut sim = MetropolisSim {
        phi: phi0,
        mass2,
        lamb,
        width,
        rng,
        accepted_history: Vec::new(),
        bc,
        action,
    };

    sim.run_sim(n_steps, logger);

    sim.accepted_history
}
