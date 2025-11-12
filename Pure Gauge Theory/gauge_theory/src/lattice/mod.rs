pub mod su2;
pub mod su3;

use nalgebra::{Complex, Const, Matrix, MatrixN};
use nalgebra::base::dimension::DimMin;
use num_traits::Float;

pub trait GaugeGroup{
    fn random() -> Self; 
    fn identity() -> Self; 
    fn dagger(&self) -> Self;
    fn mul(&self, other: &Self) -> Self; 
    fn trace(&self) -> Complex<f64>;
    fn check_group(&self) -> bool; 
    fn has_det_one(&self, tol: f64) -> bool; 
    fn is_unitary(&self, tol: f64) -> bool; 
}