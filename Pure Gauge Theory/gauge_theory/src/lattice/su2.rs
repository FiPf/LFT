use super::GaugeGroup; 

use rand::Rng;
use nalgebra::{Complex, Matrix2};

pub struct SU2(pub Matrix2<Complex<f64>>); //https://docs.rs/cgmath/latest/cgmath/struct.Matrix2.html

impl GaugeGroup for SU2{

    pub fn identity() -> Self{
        Self(Matrix2::identity())
    }

    pub fn dagger(&self) -> Self{
        Self(self.0.transpose().map(|x| x.conj()))
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self(self.0*other.0)
    }

    pub fn trace(&self) -> Complex<f64> {
        self.0[(0,0)] + self.0[(1.1)]
    }

    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        let a0: f64 = rng.gen_range(-1.0..=1.0);
        let a1: f64 = rng.gen_range(-1.0..=1.0);
        let a2: f64 = rng.gen_range(-1.0..=1.0);
        let a3: f64 = rng.gen_range(-1.0..=1.0);
        let norm = (a0*a0 + a1*a1 + a2*a2 + a3*a3).sqrt();
        let (a0, a1, a2, a3) = (a0/norm, a1/norm, a2/norm, a3/norm);
        let mat = Matrix2::new(
            Complex::new(a0, a3),
            Complex::new(a2, a1),
            Complex::new(-a2, a1),
            Complex::new(a0, -a3),
        );
        Self(mat)
    }

}