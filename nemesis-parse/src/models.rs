use serde::{Deserialize, Serialize};

/// Supported programming languages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Python,
    TypeScript,
    Tsx,
    Rust,
}

impl Language {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "py" => Some(Language::Python),
            "ts" => Some(Language::TypeScript),
            "tsx" => Some(Language::Tsx),
            "rs" => Some(Language::Rust),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Python => "python",
            Language::TypeScript => "typescript",
            Language::Tsx => "tsx",
            Language::Rust => "rust",
        }
    }
}

/// The kind of a code node extracted from an AST.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    File,
    Module,
    Class,
    Function,
    Method,
    Interface,
    Variable,
    Import,
}

/// A structured code node extracted from a parsed AST.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeNode {
    pub id: String,
    pub kind: NodeKind,
    pub name: String,
    pub file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub language: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docstring: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_hint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(default)]
    pub is_async: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_class: Option<String>,
}

/// The kind of relationship between two code nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EdgeKind {
    Contains,
    HasMethod,
    Inherits,
    Implements,
    Calls,
    Imports,
    Returns,
    Accepts,
}

/// A directed edge between two code nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEdge {
    pub source_id: String,
    pub target_id: String,
    pub kind: EdgeKind,
    pub file: String,
}

/// The complete extraction result for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub file: String,
    pub language: String,
    pub nodes: Vec<CodeNode>,
    pub edges: Vec<CodeEdge>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("py"), Some(Language::Python));
        assert_eq!(Language::from_extension("ts"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("tsx"), Some(Language::Tsx));
        assert_eq!(Language::from_extension("rs"), Some(Language::Rust));
        assert_eq!(Language::from_extension("java"), None);
    }

    #[test]
    fn test_language_as_str() {
        assert_eq!(Language::Python.as_str(), "python");
        assert_eq!(Language::TypeScript.as_str(), "typescript");
        assert_eq!(Language::Rust.as_str(), "rust");
    }

    #[test]
    fn test_code_node_serialization() {
        let node = CodeNode {
            id: "func-1".into(),
            kind: NodeKind::Function,
            name: "main".into(),
            file: "src/main.rs".into(),
            line_start: 1,
            line_end: 10,
            language: "rust".into(),
            docstring: Some("Entry point".into()),
            signature: Some("fn main()".into()),
            type_hint: None,
            scope: None,
            source: None,
            alias: None,
            visibility: Some("pub".into()),
            is_async: false,
            parent_class: None,
        };
        let json = serde_json::to_string(&node).unwrap();
        assert!(json.contains("\"name\":\"main\""));
        assert!(!json.contains("type_hint")); // None fields skipped
    }

    #[test]
    fn test_code_edge_serialization() {
        let edge = CodeEdge {
            source_id: "file-1".into(),
            target_id: "func-1".into(),
            kind: EdgeKind::Contains,
            file: "main.py".into(),
        };
        let json = serde_json::to_string(&edge).unwrap();
        assert!(json.contains("\"CONTAINS\""));
    }

    #[test]
    fn test_extraction_result_roundtrip() {
        let result = ExtractionResult {
            file: "test.py".into(),
            language: "python".into(),
            nodes: vec![],
            edges: vec![],
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: ExtractionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.file, "test.py");
    }
}
