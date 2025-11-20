use crate::lattice::{SU2, SU3}; 

pub struct Lattice{
    pub nx: usize; 
    pub ny: usize; 
    pub nz: usize;
    pub nt: usize; 
    pub vol: usize; //vol = nx*ny*nz*nt

    pub gauge:Group, 
    //pub links_su2: Option<Vec<SU2>>;  //https://www.youtube.com/watch?v=z8k_EViGC10
    //pub links_su3: Option<Vec<SU3>>; //https://doc.rust-lang.org/std/option/
    pub neigh_forward: Vec<[usize;4]>, //stores 4 matrices per site (one for each direction mu = 0...3)
    pub neigh_backward: Vec<[usize;4]>,
}

pub enum Gauge{
    SU2(Vec<[SU2; 4]>), //one SU2 matrix for each direction mu = 0 ... 3
    SU3(Vec<[SU3; 4]>), 
}

impl Lattice {
    pub fn new_su2(nx: usize, ny: usize, nz: usize, nt: usize, cold: bool) -> Self {
        Self::new_internal(nx, ny, nz, nt, cold, true)
    }

    pub fn new_su3(nx: usize, ny: usize, nz: usize, nt: usize, cold: bool) -> Self {
        Self::new_internal(nx, ny, nz, nt, cold, false)
    }

    fn new_internal(nx: usize, ny: usize, nz: usize, nt: usize, cold: bool, su2: bool) -> Self {
        let vol = nx*ny*nz*nt; 

        //periodic boundary conditions (forward/backward neighbors)
        let neigh_forward = Self::build_neighbors(nx, ny, nz, nt, true);
        let neigh_backward = Self::build_neighbors(nx, ny, nz, nt, false); 

        //build link variables
        let gauge = if su2{
            let mut links = vec![[SU2::identity(); 4]; vol];//cold start

            if !cold{//hot start
                for site in &mut links{
                    for link in site{
                        *link = SU2::random(); 
                    }
                }
            }Gauge::SU2(links)
        }else{
            let mut links = vec![[SU3:identity(); 4]; vol]; 

            if !cold{
                for site in &mut links{
                    for link in site{
                        *link = SU3::random();
                    }
                }
            }Gauge::SU3(links)
        };
        Self{nx, ny, nz, nt, vol, gauge, neigh_forward, neigh_backward}
    }

    fn build_neighbors(nx: usize, ny: usize, nz: usize, nt: usize, forward: bool) -> Vec<[usize; 4]> {
        //neighbor table
        let mut n = vec![[0;4]; nx*ny*nz*nt]; //for each site, store 4 neighbor indices

        //we want periodic boundaries
        fn wrap(i: usize, max: usize, forward: bool) -> usize {
            if forward {
                (i + 1) % max
            } else {
                (i + max - 1) % max
            }
        }

        for x in 0..nx{
            for y in 0..ny{
                for z in 0..nz{
                    for t in 0..nt{
                        
                        //https://stackoverflow.com/questions/20992156/need-of-an-algorithm-for-arrays-index-in-a-flat-representation
                        let idx = (((t*nt + z)*ny + y)*nx + x)//convert 4d coordinate to flat index
                    
                        n[idx] = [
                            (wrap(x, nx, forward), y, z, t),
                            (x, wrap(y, ny, forward), z, t),
                            (x, y, wrap(z, nz, forward), t),
                            (x, y, z, wrap(t, nt, forward)),
                        ].map(|(xx, yy, zz, tt)| (((tt * nz + zz) * ny + yy) * nx + xx));
                        //map applies this to every tuple

                    }
                }
            }
        }

        n
    }

    pub fn wilson_action(&self, beta: f64) -> f64 {
        match &self.gauge {
            Gauge::SU2(links) => self.action_su2(links, beta),
            Gauge::SU3(links) => self.action_su3(links, beta),
        }
    }

    fn action_su2(&self, links: &Vec<[SU2;4]>, beta: f64) -> f64 {
        let mut s = 0.0; 
        for x in 0..self.vec{
            for mu in 0..4 {
                for nu in (mu+1)..4{ 
                    // define the indices of forward neighbors
                    let f_mu = self.neigh_f[x][mu]
                    let f_nz = self.neigh_f[x][nu]
                    //plaquette: U_mu(x) U_nu(x+mu) U_mu^\dagger(x+nu) U_nu^\dagger(x)
                    let plaq = links[x][mu] * links[f_mu][nu] * links[f_nu][mu].dagger() * links[x][nu].dagger(); 

                    s += 1.0 - 0.5*plaq.trace().re; //SU(2) version
                }
            }
        }beta*s
    }

    fn action_su3(&self, links: &Vec<[SU3;4]>, beta: f64) -> f64 {
        let mut s = 0.0;

        for x in 0..self.vol {
            for mu in 0..4 {
                for nu in (mu+1)..4 {
                    let f_mu = self.neigh_f[x][mu];
                    let f_nu = self.neigh_f[x][nu];

                    let plaq = links[x][mu]
                        * links[f_mu][nu]
                        * links[f_nu][mu].dagger()
                        * links[x][nu].dagger();

                    s += 1.0 - (1.0/3.0) * plaq.trace().re;  // SU(3) version 
                }
            }
        }beta*s
    }
}