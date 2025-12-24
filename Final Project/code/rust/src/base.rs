pub trait BaseSim {
    fn update(&mut self); 
    fn get_steps(&self) -> usize; 
    fn run_sim<F>(&mut self, steps: usize, mut logger: Option<F>, progress_bar: bool)
    where
        F: FnMut(&Self), // `logger` closure takes a reference to Self and can mutate its own state.
        Self: Sized,     // The trait can only be implemented for sized types (lattice!)
    {
        for _ in 0..steps {
            self.update(); 

            if let Some(ref mut log_fn) = logger {
                log_fn(self); 
            }//no progress bar at the moment
        }
}

impl BaseSim {
    pub fn new() -> Self {
        Self {steps: 0}
    }

    pub fn increment_steps(&mut self){
        self.steps += 1; 
    }

    pub fn get_steps(&self) -> usize{
        self.steps
    }
}

