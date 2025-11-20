use crate::lattice::base::Lattice;
use crate::lattice::su2::SU2;

impl Lattice {
    pub fn average_plaquette_su2(&self) -> f64 {
        let links = self.links_su2.as_ref().unwrap();

        let mut sum = 0.0;
        let mut count = 0;

        for x in 0..self.vol {
            for mu in 0..4 {
                for nu in (mu+1)..4 {
                    let xp = self.neigh_forward[x][mu];
                    let yp = self.neigh_forward[x][nu];

                    let u_mu = links[x][mu];
                    let u_nu = links[x][nu];
                    let u_mu_nu = links[yp][mu];
                    let u_nu_mu = links[xp][nu];

                    let p = u_mu * u_nu_mu * u_mu_nu.dagger() * u_nu.dagger();
                    sum += p.trace().re;
                    count += 1;
                }
            }
        }

        sum / count as f64 / 2.0 // divide by N for SU(2) version
    }
}
