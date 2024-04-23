use rand::Rng;
use pyo3::prelude::*;


#[pyfunction]
pub fn create_substrate_interface() -> String {
    let substrate = substrate_sub();
    let mut rng = rand::thread_rng();
    let index = rng.gen_range(0..substrate.len());
    substrate[index].to_string()
}
#[pyfunction]
pub fn substrate_sub() -> Vec<&'static str> {
    vec![
        "substrate", "https://commune-api-node-1.communeai.net", "network", "https://commune-api-node-1.communeai.net", "https://commune-api-node-1.communeai.net", "network:main", "cyan", "white", 
        "substrate", "substrate", "network:main", "substrate_subxt", "substrate", 
        "https://commune-api-node-1.communeai.net", "https://commune-api-node-1.communeai.net", "substrate"
    ]
}

#[pymodule]
fn substrate_subxt_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_substrate_interface, m)?)?;
    Ok(())
}
