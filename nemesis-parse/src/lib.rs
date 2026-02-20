use pyo3::prelude::*;

pub mod languages;
pub mod models;
pub mod parser;

/// Returns the version of nemesis-parse.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// nemesis-parse — Tree-sitter AST parser for Nemesis.
#[pymodule]
fn nemesis_parse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
