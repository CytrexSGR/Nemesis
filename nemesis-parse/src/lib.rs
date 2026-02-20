#![allow(clippy::useless_conversion)]

mod models;
mod languages;
mod parser;
mod extractor;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::models::Language;
use crate::parser::NemesisParser;
use crate::extractor::extract;

/// Returns the version of nemesis-parse.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// List supported languages.
#[pyfunction]
fn supported_languages() -> Vec<String> {
    vec![
        "python".into(),
        "typescript".into(),
        "tsx".into(),
        "rust".into(),
    ]
}

/// Detect the language of a file from its extension.
/// Returns a language string or raises ValueError.
#[pyfunction]
fn detect_language(file_path: &str) -> PyResult<String> {
    let lang = NemesisParser::detect_language(file_path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(lang.as_str().to_string())
}

/// Parse a source string and extract all code nodes and edges.
/// Returns a JSON string with {file, language, nodes, edges}.
#[pyfunction]
fn parse_string(source: &str, language: &str, file_path: &str) -> PyResult<String> {
    let lang = match language {
        "python" => Language::Python,
        "typescript" => Language::TypeScript,
        "tsx" => Language::Tsx,
        "rust" => Language::Rust,
        other => return Err(PyValueError::new_err(format!("Unsupported language: {other}"))),
    };

    let mut parser = NemesisParser::new();
    let result = parser
        .parse_string(source, &lang, file_path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let extraction = extract(&result);
    serde_json::to_string(&extraction)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Parse a file from disk and extract all code nodes and edges.
/// Language is auto-detected from the file extension.
/// Returns a JSON string with {file, language, nodes, edges}.
#[pyfunction]
fn parse_file(file_path: &str) -> PyResult<String> {
    let mut parser = NemesisParser::new();
    let result = parser
        .parse_file(file_path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let extraction = extract(&result);
    serde_json::to_string(&extraction)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Extract only the nodes from a source string.
/// Returns a JSON array of node objects.
#[pyfunction]
fn extract_nodes(source: &str, language: &str, file_path: &str) -> PyResult<String> {
    let lang = match language {
        "python" => Language::Python,
        "typescript" => Language::TypeScript,
        "tsx" => Language::Tsx,
        "rust" => Language::Rust,
        other => return Err(PyValueError::new_err(format!("Unsupported language: {other}"))),
    };

    let mut parser = NemesisParser::new();
    let result = parser
        .parse_string(source, &lang, file_path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let extraction = extract(&result);
    serde_json::to_string(&extraction.nodes)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Extract only the edges from a source string.
/// Returns a JSON array of edge objects.
#[pyfunction]
fn extract_edges(source: &str, language: &str, file_path: &str) -> PyResult<String> {
    let lang = match language {
        "python" => Language::Python,
        "typescript" => Language::TypeScript,
        "tsx" => Language::Tsx,
        "rust" => Language::Rust,
        other => return Err(PyValueError::new_err(format!("Unsupported language: {other}"))),
    };

    let mut parser = NemesisParser::new();
    let result = parser
        .parse_string(source, &lang, file_path)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let extraction = extract(&result);
    serde_json::to_string(&extraction.edges)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// nemesis-parse -- Tree-sitter AST parser for Nemesis.
#[pymodule]
fn _nemesis_parse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(supported_languages, m)?)?;
    m.add_function(wrap_pyfunction!(detect_language, m)?)?;
    m.add_function(wrap_pyfunction!(parse_string, m)?)?;
    m.add_function(wrap_pyfunction!(parse_file, m)?)?;
    m.add_function(wrap_pyfunction!(extract_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(extract_edges, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
