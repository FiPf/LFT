use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod physics;
mod metropolis;
mod base;
mod hmc; 
mod integrators;

#[pymodule]
fn lattice_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(physics::phi4_action, m)?)?;
    m.add_function(wrap_pyfunction!(physics::gradient_phi4_action, m)?)?;
    m.add_class::<metropolis::MetropolisSim>()?;
    m.add_function(wrap_pyfunction!(metropolis::run_metropolis, m)?)?;
    m.add_class::<hmc::HMCSim>()?; 
    Ok(())
}
