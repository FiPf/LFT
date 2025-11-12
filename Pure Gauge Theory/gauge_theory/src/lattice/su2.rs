use super::GaugeGroup;

use rand::Rng;
use nalgebra::{Complex, Matrix2};

pub struct SU2(pub Matrix2<Complex<f64>>);

impl GaugeGroup for SU2 {
    fn identity() -> Self {
        Self(Matrix2::identity())
    }

    fn dagger(&self) -> Self {
        Self(self.0.transpose().map(|x| x.conj()))
    }

    fn mul(&self, other: &Self) -> Self {
        Self(self.0 * other.0)
    }

    fn trace(&self) -> Complex<f64> {
        self.0[(0, 0)] + self.0[(1, 1)]
    }

    fn random() -> Self {
        let mut rng = rand::thread_rng();
        let a0: f64 = rng.gen_range(-1.0..=1.0);
        let a1: f64 = rng.gen_range(-1.0..=1.0);
        let a2: f64 = rng.gen_range(-1.0..=1.0);
        let a3: f64 = rng.gen_range(-1.0..=1.0);
        let norm = (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3).sqrt();
        let (a0, a1, a2, a3) = (a0 / norm, a1 / norm, a2 / norm, a3 / norm);

        let mat = Matrix2::new(
            Complex::new(a0, a3),
            Complex::new(a2, a1),
            Complex::new(-a2, a1),
            Complex::new(a0, -a3),
        );
        Self(mat)
    }

    fn has_det_one(&self, tol: f64) -> bool {
        let det = self.0.determinant();
        (det - Complex::new(1.0, 0.0)).norm() < tol
    }

    fn is_unitary(&self, tol: f64) -> bool {
        let uu = self.0.adjoint() * self.0;
        let id = Matrix2::<Complex<f64>>::identity();
        let diff = uu - id;
        diff.iter().map(|x| x.norm()).fold(0.0, f64::max) < tol
    }

    fn check_group(&self) -> bool {
        let tol = 1e-10; 
        Self::is_unitary(&self, tol) && Self::has_det_one(&self, tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::ComplexField;

    #[test]
    fn test_su2_random_matrix_properties() {
        let u = SU2::random();

        // Check unitarity
        assert!(
            u.is_unitary(1e-10),
            "Matrix is not unitary: U†U != I"
        );

        // Check determinant ≈ 1
        assert!(
            u.has_det_one(1e-10),
            "Matrix determinant is not 1, det(U) = {}",
            u.0.determinant()
        );

        // Check combined group property
        assert!(
            u.check_group(),
            "Matrix does not satisfy SU(2) group conditions"
        );
    }

    #[test]
    fn test_su2_identity_properties() {
        let id = SU2::identity();
        assert!(
            id.is_unitary(1e-10),
            "Identity is not unitary"
        );
        assert!(
            id.has_det_one(1e-10),
            "Identity does not have determinant 1"
        );
    }
}

