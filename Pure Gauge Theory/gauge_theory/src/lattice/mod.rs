pub mod su2
pub mod su3

pub trait GaugeGroup{
    fn random() -> Self; 
    fn identity() -> Self; 
    fn dag(&self) -> Self;
    fn mul(&self, other: &Self) -> Self; 
    fn trace(&self) -> Complex<f64>;
}