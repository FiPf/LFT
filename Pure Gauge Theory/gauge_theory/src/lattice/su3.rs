use super::GaugeGroup;
use rand::Rng;
use nalgebra::{Complex, Matrix3};
use nalgebra::{QR, Unit, U3};

pub struct SU3(pub Matrix3<Complex<f64>>);

impl GaugeGroup for SU3 {
    fn identity() -> Self {
        Self(Matrix3::identity())
    }

    fn dagger(&self) -> Self {
        Self(self.0.transpose().map(|x| x.conj()))
    }

    fn mul(&self, other: &Self) -> Self {
        Self(self.0 * other.0)
    }

    fn trace(&self) -> Complex<f64> {
        self.0[(0, 0)] + self.0[(1, 1)] + self.0[(2, 2)]
    }

    fn random() -> Self {
        let mut rng = rand::thread_rng(); 

        let mut z = Matrix3::from_fn(|_, _| {
            let re: f64 = rng.gen_range(-1.0..=1.0);
            let im: f64 = rng.gen_range(-1.0..=1.0);
            Complex::new(re, im)
        });

        let qr = z.qr(); //Gram Schmidt orthogonalization
        let mut q = qr.q();

        let det = q.determinant();
        let phase = det / det.norm();
        q = q*phase.conj(); 

        Self(q)
    }

    fn has_det_one(&self, tol: f64) -> bool {
        let det = self.0.determinant();
        (det - Complex::new(1.0, 0.0)).norm() < tol
    }

    fn is_unitary(&self, tol: f64) -> bool {
        let uu = self.0.adjoint() * self.0;
        let id = Matrix3::<Complex<f64>>::identity();
        let diff = uu - id;
        diff.iter().map(|x| x.norm()).fold(0.0, f64::max) < tol
    }

    fn check_group(&self) -> bool {
        let tol = 1e-10; 
        Self::is_unitary(&self, tol) && Self::has_det_one(&self, tol)
    }
}
