# Rust Parser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the nemesis-parse Rust crate — a high-performance multi-language AST parser using Tree-sitter, exposed to Python via PyO3/maturin.

**Architecture:** The Rust crate `nemesis-parse/` contains three core modules: `parser.rs` (Tree-sitter initialization and source parsing), `extractor.rs` (AST walking to extract structured code nodes and edges), and `languages/` (grammar bindings for Python, TypeScript, Rust). All data structures use serde for JSON serialization. PyO3 exposes `parse_file()`, `parse_string()`, `extract_nodes()`, and `extract_edges()` to Python. A Python bridge module (`nemesis/parser/bridge.py`) wraps the native extension with error handling, fallback support, and Pythonic dataclasses.

**Tech Stack:** Rust, PyO3 0.22, maturin, Tree-sitter 0.24, tree-sitter-python 0.23, tree-sitter-typescript 0.23, tree-sitter-rust 0.23, serde, serde_json

**Design Document:** [2026-02-20-nemesis-design.md](2026-02-20-nemesis-design.md)

**Depends on:** [01-project-setup](01-project-setup.md)

---

## Graph Schema (Reference)

```
CODE NODES (from Tree-sitter):
  :File          {path, language, hash, last_indexed, size}
  :Module        {name, path, docstring}
  :Class         {name, file, line_start, line_end, docstring}
  :Function      {name, file, line_start, line_end, signature, docstring, is_async}
  :Method        {name, class, file, line_start, line_end, signature, visibility}
  :Interface     {name, file, language}
  :Variable      {name, file, type_hint, scope}
  :Import        {name, source, alias}

CODE EDGES:
  (:File)-[:CONTAINS]->(:Class|:Function|:Variable)
  (:Class)-[:HAS_METHOD]->(:Method)
  (:Class)-[:INHERITS]->(:Class)
  (:Class)-[:IMPLEMENTS]->(:Interface)
  (:Function)-[:CALLS]->(:Function|:Method)
  (:Function)-[:IMPORTS]->(:Module|:File)
  (:Function)-[:RETURNS]->(:Class)
  (:Function)-[:ACCEPTS]->(:Class)
  (:File)-[:IMPORTS]->(:File)
```

---

## Task 1: Rust Data Models (models.rs)

Serde-serializable structs for all code node and edge types that the extractor will produce.

**Files:**
- Create: `nemesis-parse/src/models.rs`
- Modify: `nemesis-parse/src/lib.rs` (add `mod models;`)

### Step 1 — Write the Rust code

```rust
// nemesis-parse/src/models.rs
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
```

### Step 2 — Update lib.rs to include the module

Add `mod models;` to `nemesis-parse/src/lib.rs` (keep existing `version()` function and `nemesis_parse` pymodule).

### Step 3 — Validate compilation

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && cargo test`
Expected: 5 tests pass in `models::tests`

### Step 4 — Commit

```bash
git add nemesis-parse/src/models.rs nemesis-parse/src/lib.rs
git commit -m "feat(parse): add serde data models for code nodes, edges, and extraction results"
```

---

## Task 2: Language Grammar Bindings (languages/)

Module that initializes Tree-sitter grammars for Python, TypeScript, TSX, and Rust.

**Files:**
- Create: `nemesis-parse/src/languages/mod.rs`
- Create: `nemesis-parse/src/languages/python.rs`
- Create: `nemesis-parse/src/languages/typescript.rs`
- Create: `nemesis-parse/src/languages/rust_lang.rs`
- Modify: `nemesis-parse/src/lib.rs` (add `mod languages;`)

### Step 1 — Write the Rust code

```rust
// nemesis-parse/src/languages/mod.rs
pub mod python;
pub mod typescript;
pub mod rust_lang;

use tree_sitter::Language as TsLanguage;
use crate::models::Language;

/// Return the tree-sitter Language grammar for a given Language enum.
pub fn get_grammar(lang: &Language) -> TsLanguage {
    match lang {
        Language::Python => python::language(),
        Language::TypeScript => typescript::language_typescript(),
        Language::Tsx => typescript::language_tsx(),
        Language::Rust => rust_lang::language(),
    }
}
```

```rust
// nemesis-parse/src/languages/python.rs
use tree_sitter::Language as TsLanguage;

pub fn language() -> TsLanguage {
    tree_sitter_python::LANGUAGE.into()
}

/// Tree-sitter query patterns for Python code extraction.
pub const CLASS_QUERY: &str = r#"
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases
  body: (block) @class.body) @class.def
"#;

pub const FUNCTION_QUERY: &str = r#"
(function_definition
  name: (identifier) @func.name
  parameters: (parameters) @func.params
  return_type: (type)? @func.return
  body: (block) @func.body) @func.def
"#;

pub const IMPORT_QUERY: &str = r#"
[
  (import_statement
    name: (dotted_name) @import.name) @import.stmt
  (import_from_statement
    module_name: (dotted_name) @import.module
    name: (dotted_name) @import.name) @import.stmt
]
"#;

pub const ASSIGNMENT_QUERY: &str = r#"
(assignment
  left: (identifier) @var.name
  type: (type)? @var.type
  right: (_) @var.value) @var.def
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_grammar_loads() {
        let lang = language();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&lang).expect("Python grammar must load");
    }
}
```

```rust
// nemesis-parse/src/languages/typescript.rs
use tree_sitter::Language as TsLanguage;

pub fn language_typescript() -> TsLanguage {
    tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
}

pub fn language_tsx() -> TsLanguage {
    tree_sitter_typescript::LANGUAGE_TSX.into()
}

pub const CLASS_QUERY: &str = r#"
(class_declaration
  name: (type_identifier) @class.name
  body: (class_body) @class.body) @class.def
"#;

pub const FUNCTION_QUERY: &str = r#"
[
  (function_declaration
    name: (identifier) @func.name
    parameters: (formal_parameters) @func.params
    return_type: (type_annotation)? @func.return
    body: (statement_block) @func.body) @func.def
  (arrow_function
    parameters: (formal_parameters) @func.params
    body: (_) @func.body) @func.def
]
"#;

pub const INTERFACE_QUERY: &str = r#"
(interface_declaration
  name: (type_identifier) @iface.name
  body: (interface_body) @iface.body) @iface.def
"#;

pub const IMPORT_QUERY: &str = r#"
(import_statement
  source: (string) @import.source) @import.stmt
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typescript_grammar_loads() {
        let lang = language_typescript();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&lang).expect("TypeScript grammar must load");
    }

    #[test]
    fn test_tsx_grammar_loads() {
        let lang = language_tsx();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&lang).expect("TSX grammar must load");
    }
}
```

```rust
// nemesis-parse/src/languages/rust_lang.rs
use tree_sitter::Language as TsLanguage;

pub fn language() -> TsLanguage {
    tree_sitter_rust::LANGUAGE.into()
}

pub const FUNCTION_QUERY: &str = r#"
(function_item
  name: (identifier) @func.name
  parameters: (parameters) @func.params
  return_type: (_)? @func.return
  body: (block) @func.body) @func.def
"#;

pub const STRUCT_QUERY: &str = r#"
(struct_item
  name: (type_identifier) @struct.name
  body: (_)? @struct.body) @struct.def
"#;

pub const IMPL_QUERY: &str = r#"
(impl_item
  type: (type_identifier) @impl.type
  body: (declaration_list) @impl.body) @impl.def
"#;

pub const USE_QUERY: &str = r#"
(use_declaration
  argument: (_) @use.path) @use.stmt
"#;

pub const TRAIT_QUERY: &str = r#"
(trait_item
  name: (type_identifier) @trait.name
  body: (declaration_list) @trait.body) @trait.def
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_grammar_loads() {
        let lang = language();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&lang).expect("Rust grammar must load");
    }
}
```

### Step 2 — Validate compilation

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && cargo test`
Expected: All grammar load tests pass (4 new tests)

### Step 3 — Commit

```bash
git add nemesis-parse/src/languages/ nemesis-parse/src/lib.rs
git commit -m "feat(parse): add Tree-sitter grammar bindings for Python, TypeScript, TSX, and Rust"
```

---

## Task 3: Multi-Language Parser (parser.rs)

Core parser that takes source code + language and produces a Tree-sitter syntax tree.

**Files:**
- Create: `nemesis-parse/src/parser.rs`
- Modify: `nemesis-parse/src/lib.rs` (add `mod parser;`)

### Step 1 — Write the Rust code

```rust
// nemesis-parse/src/parser.rs
use std::path::Path;
use tree_sitter::{Parser, Tree};

use crate::languages;
use crate::models::Language;

/// Errors that can occur during parsing.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Unsupported file extension: {0}")]
    UnsupportedExtension(String),
    #[error("Failed to set parser language: {0}")]
    LanguageError(String),
    #[error("Failed to parse source code")]
    ParseFailed,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result of parsing a source file.
pub struct ParseResult {
    pub tree: Tree,
    pub source: String,
    pub language: Language,
    pub file_path: String,
}

/// Multi-language parser wrapping Tree-sitter.
pub struct NemesisParser {
    parser: Parser,
}

impl NemesisParser {
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
        }
    }

    /// Detect language from file extension.
    pub fn detect_language(file_path: &str) -> Result<Language, ParseError> {
        let ext = Path::new(file_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        Language::from_extension(ext)
            .ok_or_else(|| ParseError::UnsupportedExtension(ext.to_string()))
    }

    /// Parse a source string with an explicit language.
    pub fn parse_string(
        &mut self,
        source: &str,
        language: &Language,
        file_path: &str,
    ) -> Result<ParseResult, ParseError> {
        let grammar = languages::get_grammar(language);
        self.parser
            .set_language(&grammar)
            .map_err(|e| ParseError::LanguageError(e.to_string()))?;

        let tree = self
            .parser
            .parse(source, None)
            .ok_or(ParseError::ParseFailed)?;

        Ok(ParseResult {
            tree,
            source: source.to_string(),
            language: language.clone(),
            file_path: file_path.to_string(),
        })
    }

    /// Parse a file from disk, detecting language from extension.
    pub fn parse_file(&mut self, file_path: &str) -> Result<ParseResult, ParseError> {
        let language = Self::detect_language(file_path)?;
        let source = std::fs::read_to_string(file_path)?;
        self.parse_string(&source, &language, file_path)
    }
}

impl Default for NemesisParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language_python() {
        let lang = NemesisParser::detect_language("test.py").unwrap();
        assert_eq!(lang, Language::Python);
    }

    #[test]
    fn test_detect_language_typescript() {
        let lang = NemesisParser::detect_language("app.ts").unwrap();
        assert_eq!(lang, Language::TypeScript);
    }

    #[test]
    fn test_detect_language_rust() {
        let lang = NemesisParser::detect_language("main.rs").unwrap();
        assert_eq!(lang, Language::Rust);
    }

    #[test]
    fn test_detect_language_unsupported() {
        let result = NemesisParser::detect_language("file.java");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_python_string() {
        let mut parser = NemesisParser::new();
        let result = parser
            .parse_string("def hello():\n    pass\n", &Language::Python, "test.py")
            .unwrap();
        assert_eq!(result.language, Language::Python);
        let root = result.tree.root_node();
        assert_eq!(root.kind(), "module");
        assert!(root.child_count() > 0);
    }

    #[test]
    fn test_parse_typescript_string() {
        let mut parser = NemesisParser::new();
        let result = parser
            .parse_string(
                "function greet(name: string): void { console.log(name); }\n",
                &Language::TypeScript,
                "test.ts",
            )
            .unwrap();
        assert_eq!(result.language, Language::TypeScript);
        let root = result.tree.root_node();
        assert_eq!(root.kind(), "program");
    }

    #[test]
    fn test_parse_rust_string() {
        let mut parser = NemesisParser::new();
        let result = parser
            .parse_string("fn main() { println!(\"hello\"); }\n", &Language::Rust, "main.rs")
            .unwrap();
        assert_eq!(result.language, Language::Rust);
        let root = result.tree.root_node();
        assert_eq!(root.kind(), "source_file");
    }

    #[test]
    fn test_parse_preserves_source() {
        let mut parser = NemesisParser::new();
        let src = "x = 42\n";
        let result = parser
            .parse_string(src, &Language::Python, "t.py")
            .unwrap();
        assert_eq!(result.source, src);
        assert_eq!(result.file_path, "t.py");
    }
}
```

### Step 2 — Add thiserror dependency

Add to `nemesis-parse/Cargo.toml` under `[dependencies]`:
```toml
thiserror = "1"
```

### Step 3 — Validate

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && cargo test`
Expected: All parser tests pass (7 new tests)

### Step 4 — Commit

```bash
git add nemesis-parse/src/parser.rs nemesis-parse/src/lib.rs nemesis-parse/Cargo.toml
git commit -m "feat(parse): add multi-language Tree-sitter parser with auto-detection"
```

---

## Task 4: Python Extractor (extractor.rs — Python support)

Walk Python ASTs to extract classes, functions, imports, and variables. Build CONTAINS, HAS_METHOD, INHERITS, and IMPORTS edges.

**Files:**
- Create: `nemesis-parse/src/extractor.rs`
- Modify: `nemesis-parse/src/lib.rs` (add `mod extractor;`)

### Step 1 — Write the Rust code

```rust
// nemesis-parse/src/extractor.rs
use tree_sitter::Node;

use crate::models::*;
use crate::parser::ParseResult;

/// Generate a deterministic node ID from file path, kind, and name.
fn make_id(file: &str, kind: &str, name: &str, line: usize) -> String {
    format!("{kind}:{file}:{name}:{line}")
}

/// Extract text from source for a given tree-sitter node.
fn node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    &source[node.byte_range()]
}

/// Try to find a docstring (first expression_statement containing a string)
/// immediately inside a block node (Python-style).
fn extract_python_docstring(body_node: &Node, source: &str) -> Option<String> {
    if body_node.kind() != "block" {
        return None;
    }
    let first_child = body_node.named_child(0)?;
    if first_child.kind() == "expression_statement" {
        let expr = first_child.named_child(0)?;
        if expr.kind() == "string" || expr.kind() == "concatenated_string" {
            let raw = node_text(&expr, source);
            // Strip triple quotes
            let trimmed = raw
                .trim_start_matches("\"\"\"")
                .trim_start_matches("'''")
                .trim_end_matches("\"\"\"")
                .trim_end_matches("'''")
                .trim();
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Main extraction entry point: walks the parse result and produces nodes + edges.
pub fn extract(parse_result: &ParseResult) -> ExtractionResult {
    let source = &parse_result.source;
    let file_path = &parse_result.file_path;
    let lang_str = parse_result.language.as_str().to_string();

    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // File node
    let file_id = format!("file:{file_path}");
    nodes.push(CodeNode {
        id: file_id.clone(),
        kind: NodeKind::File,
        name: file_path.clone(),
        file: file_path.clone(),
        line_start: 1,
        line_end: source.lines().count(),
        language: lang_str.clone(),
        docstring: None,
        signature: None,
        type_hint: None,
        scope: None,
        source: None,
        alias: None,
        visibility: None,
        is_async: false,
        parent_class: None,
    });

    let root = parse_result.tree.root_node();

    match parse_result.language {
        Language::Python => {
            extract_python_nodes(&root, source, file_path, &lang_str, &file_id, &mut nodes, &mut edges);
        }
        Language::TypeScript | Language::Tsx => {
            extract_typescript_nodes(&root, source, file_path, &lang_str, &file_id, &mut nodes, &mut edges);
        }
        Language::Rust => {
            extract_rust_nodes(&root, source, file_path, &lang_str, &file_id, &mut nodes, &mut edges);
        }
    }

    ExtractionResult {
        file: file_path.clone(),
        language: lang_str,
        nodes,
        edges,
    }
}

// ---------------------------------------------------------------------------
// Python extraction
// ---------------------------------------------------------------------------

fn extract_python_nodes(
    node: &Node,
    source: &str,
    file_path: &str,
    lang: &str,
    parent_id: &str,
    nodes: &mut Vec<CodeNode>,
    edges: &mut Vec<CodeEdge>,
) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "class_definition" => {
                extract_python_class(&child, source, file_path, lang, parent_id, nodes, edges);
            }
            "function_definition" => {
                extract_python_function(&child, source, file_path, lang, parent_id, None, nodes, edges);
            }
            "import_statement" | "import_from_statement" => {
                extract_python_import(&child, source, file_path, lang, parent_id, nodes, edges);
            }
            "expression_statement" => {
                // Top-level assignments
                if let Some(assign) = child.named_child(0) {
                    if assign.kind() == "assignment" {
                        extract_python_variable(&assign, source, file_path, lang, parent_id, "module", nodes, edges);
                    }
                }
            }
            _ => {}
        }
    }
}

fn extract_python_class(
    node: &Node,
    source: &str,
    file_path: &str,
    lang: &str,
    parent_id: &str,
    nodes: &mut Vec<CodeNode>,
    edges: &mut Vec<CodeEdge>,
) {
    let name_node = match node.child_by_field_name("name") {
        Some(n) => n,
        None => return,
    };
    let name = node_text(&name_node, source).to_string();
    let line_start = node.start_position().row + 1;
    let line_end = node.end_position().row + 1;
    let class_id = make_id(file_path, "class", &name, line_start);

    // Docstring from body
    let body = node.child_by_field_name("body");
    let docstring = body.as_ref().and_then(|b| extract_python_docstring(b, source));

    nodes.push(CodeNode {
        id: class_id.clone(),
        kind: NodeKind::Class,
        name: name.clone(),
        file: file_path.to_string(),
        line_start,
        line_end,
        language: lang.to_string(),
        docstring,
        signature: None,
        type_hint: None,
        scope: None,
        source: None,
        alias: None,
        visibility: None,
        is_async: false,
        parent_class: None,
    });

    edges.push(CodeEdge {
        source_id: parent_id.to_string(),
        target_id: class_id.clone(),
        kind: EdgeKind::Contains,
        file: file_path.to_string(),
    });

    // Base classes -> INHERITS edges
    if let Some(bases) = node.child_by_field_name("superclasses") {
        let mut bc = bases.walk();
        for base_child in bases.named_children(&mut bc) {
            if base_child.kind() == "identifier" || base_child.kind() == "attribute" {
                let base_name = node_text(&base_child, source);
                edges.push(CodeEdge {
                    source_id: class_id.clone(),
                    target_id: format!("class:{file_path}:{base_name}:0"),
                    kind: EdgeKind::Inherits,
                    file: file_path.to_string(),
                });
            }
        }
    }

    // Walk class body for methods
    if let Some(body) = body {
        let mut bc = body.walk();
        for child in body.named_children(&mut bc) {
            if child.kind() == "function_definition" {
                extract_python_function(&child, source, file_path, lang, &class_id, Some(&name), nodes, edges);
            }
        }
    }
}

fn extract_python_function(
    node: &Node,
    source: &str,
    file_path: &str,
    lang: &str,
    parent_id: &str,
    parent_class: Option<&str>,
    nodes: &mut Vec<CodeNode>,
    edges: &mut Vec<CodeEdge>,
) {
    let name_node = match node.child_by_field_name("name") {
        Some(n) => n,
        None => return,
    };
    let name = node_text(&name_node, source).to_string();
    let line_start = node.start_position().row + 1;
    let line_end = node.end_position().row + 1;

    let is_method = parent_class.is_some();
    let kind = if is_method { NodeKind::Method } else { NodeKind::Function };
    let node_id = make_id(
        file_path,
        if is_method { "method" } else { "func" },
        &name,
        line_start,
    );

    // Build signature
    let params = node.child_by_field_name("parameters")
        .map(|p| node_text(&p, source).to_string());
    let return_type = node.child_by_field_name("return_type")
        .map(|r| node_text(&r, source).to_string());
    let signature = match (&params, &return_type) {
        (Some(p), Some(r)) => Some(format!("{name}{p} -> {r}")),
        (Some(p), None) => Some(format!("{name}{p}")),
        _ => None,
    };

    // Check async
    let is_async = source[node.byte_range().start.saturating_sub(6)..node.byte_range().start]
        .trim()
        .ends_with("async");

    // Docstring
    let body = node.child_by_field_name("body");
    let docstring = body.as_ref().and_then(|b| extract_python_docstring(b, source));

    // Visibility for methods
    let visibility = if is_method {
        if name.starts_with("__") && name.ends_with("__") {
            Some("dunder".to_string())
        } else if name.starts_with("__") {
            Some("private".to_string())
        } else if name.starts_with('_') {
            Some("protected".to_string())
        } else {
            Some("public".to_string())
        }
    } else {
        None
    };

    nodes.push(CodeNode {
        id: node_id.clone(),
        kind,
        name,
        file: file_path.to_string(),
        line_start,
        line_end,
        language: lang.to_string(),
        docstring,
        signature,
        type_hint: None,
        scope: None,
        source: None,
        alias: None,
        visibility,
        is_async,
        parent_class: parent_class.map(|s| s.to_string()),
    });

    let edge_kind = if is_method { EdgeKind::HasMethod } else { EdgeKind::Contains };
    edges.push(CodeEdge {
        source_id: parent_id.to_string(),
        target_id: node_id,
        kind: edge_kind,
        file: file_path.to_string(),
    });
}

fn extract_python_import(
    node: &Node,
    source: &str,
    file_path: &str,
    lang: &str,
    parent_id: &str,
    nodes: &mut Vec<CodeNode>,
    edges: &mut Vec<CodeEdge>,
) {
    let text = node_text(node, source).to_string();
    let line = node.start_position().row + 1;
    let import_id = make_id(file_path, "import", &text.replace(' ', "_"), line);

    // Extract module name
    let module_name = if node.kind() == "import_from_statement" {
        node.child_by_field_name("module_name")
            .map(|n| node_text(&n, source).to_string())
    } else {
        node.named_child(0).map(|n| node_text(&n, source).to_string())
    };

    nodes.push(CodeNode {
        id: import_id.clone(),
        kind: NodeKind::Import,
        name: module_name.clone().unwrap_or_default(),
        file: file_path.to_string(),
        line_start: line,
        line_end: line,
        language: lang.to_string(),
        docstring: None,
        signature: None,
        type_hint: None,
        scope: None,
        source: module_name,
        alias: None,
        visibility: None,
        is_async: false,
        parent_class: None,
    });

    edges.push(CodeEdge {
        source_id: parent_id.to_string(),
        target_id: import_id,
        kind: EdgeKind::Imports,
        file: file_path.to_string(),
    });
}

fn extract_python_variable(
    node: &Node,
    source: &str,
    file_path: &str,
    lang: &str,
    parent_id: &str,
    scope: &str,
    nodes: &mut Vec<CodeNode>,
    edges: &mut Vec<CodeEdge>,
) {
    let name_node = match node.child_by_field_name("left") {
        Some(n) if n.kind() == "identifier" => n,
        _ => return,
    };
    let name = node_text(&name_node, source).to_string();
    let line = node.start_position().row + 1;
    let var_id = make_id(file_path, "var", &name, line);

    let type_hint = node.child_by_field_name("type")
        .map(|t| node_text(&t, source).to_string());

    nodes.push(CodeNode {
        id: var_id.clone(),
        kind: NodeKind::Variable,
        name,
        file: file_path.to_string(),
        line_start: line,
        line_end: line,
        language: lang.to_string(),
        docstring: None,
        signature: None,
        type_hint,
        scope: Some(scope.to_string()),
        source: None,
        alias: None,
        visibility: None,
        is_async: false,
        parent_class: None,
    });

    edges.push(CodeEdge {
        source_id: parent_id.to_string(),
        target_id: var_id,
        kind: EdgeKind::Contains,
        file: file_path.to_string(),
    });
}

// ---------------------------------------------------------------------------
// TypeScript extraction
// ---------------------------------------------------------------------------

fn extract_typescript_nodes(
    node: &Node,
    source: &str,
    file_path: &str,
    lang: &str,
    parent_id: &str,
    nodes: &mut Vec<CodeNode>,
    edges: &mut Vec<CodeEdge>,
) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "class_declaration" => {
                let name_node = match child.child_by_field_name("name") {
                    Some(n) => n,
                    None => continue,
                };
                let name = node_text(&name_node, source).to_string();
                let line_start = child.start_position().row + 1;
                let line_end = child.end_position().row + 1;
                let class_id = make_id(file_path, "class", &name, line_start);

                nodes.push(CodeNode {
                    id: class_id.clone(),
                    kind: NodeKind::Class,
                    name,
                    file: file_path.to_string(),
                    line_start,
                    line_end,
                    language: lang.to_string(),
                    docstring: None,
                    signature: None,
                    type_hint: None,
                    scope: None,
                    source: None,
                    alias: None,
                    visibility: None,
                    is_async: false,
                    parent_class: None,
                });

                edges.push(CodeEdge {
                    source_id: parent_id.to_string(),
                    target_id: class_id.clone(),
                    kind: EdgeKind::Contains,
                    file: file_path.to_string(),
                });

                // Methods inside class body
                if let Some(body) = child.child_by_field_name("body") {
                    let mut bc = body.walk();
                    for member in body.named_children(&mut bc) {
                        if member.kind() == "method_definition" {
                            if let Some(mn) = member.child_by_field_name("name") {
                                let mname = node_text(&mn, source).to_string();
                                let mls = member.start_position().row + 1;
                                let mle = member.end_position().row + 1;
                                let mid = make_id(file_path, "method", &mname, mls);

                                nodes.push(CodeNode {
                                    id: mid.clone(),
                                    kind: NodeKind::Method,
                                    name: mname,
                                    file: file_path.to_string(),
                                    line_start: mls,
                                    line_end: mle,
                                    language: lang.to_string(),
                                    docstring: None,
                                    signature: None,
                                    type_hint: None,
                                    scope: None,
                                    source: None,
                                    alias: None,
                                    visibility: Some("public".to_string()),
                                    is_async: false,
                                    parent_class: None,
                                });

                                edges.push(CodeEdge {
                                    source_id: class_id.clone(),
                                    target_id: mid,
                                    kind: EdgeKind::HasMethod,
                                    file: file_path.to_string(),
                                });
                            }
                        }
                    }
                }
            }
            "function_declaration" => {
                let name_node = match child.child_by_field_name("name") {
                    Some(n) => n,
                    None => continue,
                };
                let name = node_text(&name_node, source).to_string();
                let line_start = child.start_position().row + 1;
                let line_end = child.end_position().row + 1;
                let func_id = make_id(file_path, "func", &name, line_start);

                nodes.push(CodeNode {
                    id: func_id.clone(),
                    kind: NodeKind::Function,
                    name,
                    file: file_path.to_string(),
                    line_start,
                    line_end,
                    language: lang.to_string(),
                    docstring: None,
                    signature: None,
                    type_hint: None,
                    scope: None,
                    source: None,
                    alias: None,
                    visibility: None,
                    is_async: false,
                    parent_class: None,
                });

                edges.push(CodeEdge {
                    source_id: parent_id.to_string(),
                    target_id: func_id,
                    kind: EdgeKind::Contains,
                    file: file_path.to_string(),
                });
            }
            "interface_declaration" => {
                let name_node = match child.child_by_field_name("name") {
                    Some(n) => n,
                    None => continue,
                };
                let name = node_text(&name_node, source).to_string();
                let line_start = child.start_position().row + 1;
                let line_end = child.end_position().row + 1;
                let iface_id = make_id(file_path, "iface", &name, line_start);

                nodes.push(CodeNode {
                    id: iface_id.clone(),
                    kind: NodeKind::Interface,
                    name,
                    file: file_path.to_string(),
                    line_start,
                    line_end,
                    language: lang.to_string(),
                    docstring: None,
                    signature: None,
                    type_hint: None,
                    scope: None,
                    source: None,
                    alias: None,
                    visibility: None,
                    is_async: false,
                    parent_class: None,
                });

                edges.push(CodeEdge {
                    source_id: parent_id.to_string(),
                    target_id: iface_id,
                    kind: EdgeKind::Contains,
                    file: file_path.to_string(),
                });
            }
            "import_statement" => {
                let text = node_text(&child, source);
                let line = child.start_position().row + 1;
                let import_id = make_id(file_path, "import", &text.replace(' ', "_"), line);

                let import_source = child.child_by_field_name("source")
                    .map(|s| node_text(&s, source).trim_matches(|c| c == '\'' || c == '"').to_string());

                nodes.push(CodeNode {
                    id: import_id.clone(),
                    kind: NodeKind::Import,
                    name: import_source.clone().unwrap_or_default(),
                    file: file_path.to_string(),
                    line_start: line,
                    line_end: line,
                    language: lang.to_string(),
                    docstring: None,
                    signature: None,
                    type_hint: None,
                    scope: None,
                    source: import_source,
                    alias: None,
                    visibility: None,
                    is_async: false,
                    parent_class: None,
                });

                edges.push(CodeEdge {
                    source_id: parent_id.to_string(),
                    target_id: import_id,
                    kind: EdgeKind::Imports,
                    file: file_path.to_string(),
                });
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Rust extraction
// ---------------------------------------------------------------------------

fn extract_rust_nodes(
    node: &Node,
    source: &str,
    file_path: &str,
    lang: &str,
    parent_id: &str,
    nodes: &mut Vec<CodeNode>,
    edges: &mut Vec<CodeEdge>,
) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "function_item" => {
                let name_node = match child.child_by_field_name("name") {
                    Some(n) => n,
                    None => continue,
                };
                let name = node_text(&name_node, source).to_string();
                let line_start = child.start_position().row + 1;
                let line_end = child.end_position().row + 1;
                let func_id = make_id(file_path, "func", &name, line_start);

                // Check for pub visibility
                let vis = {
                    let src_slice = &source[child.byte_range()];
                    if src_slice.starts_with("pub ") { Some("pub".to_string()) } else { None }
                };

                nodes.push(CodeNode {
                    id: func_id.clone(),
                    kind: NodeKind::Function,
                    name,
                    file: file_path.to_string(),
                    line_start,
                    line_end,
                    language: lang.to_string(),
                    docstring: None,
                    signature: None,
                    type_hint: None,
                    scope: None,
                    source: None,
                    alias: None,
                    visibility: vis,
                    is_async: false,
                    parent_class: None,
                });

                edges.push(CodeEdge {
                    source_id: parent_id.to_string(),
                    target_id: func_id,
                    kind: EdgeKind::Contains,
                    file: file_path.to_string(),
                });
            }
            "struct_item" => {
                let name_node = match child.child_by_field_name("name") {
                    Some(n) => n,
                    None => continue,
                };
                let name = node_text(&name_node, source).to_string();
                let line_start = child.start_position().row + 1;
                let line_end = child.end_position().row + 1;
                let struct_id = make_id(file_path, "class", &name, line_start);

                nodes.push(CodeNode {
                    id: struct_id.clone(),
                    kind: NodeKind::Class,
                    name,
                    file: file_path.to_string(),
                    line_start,
                    line_end,
                    language: lang.to_string(),
                    docstring: None,
                    signature: None,
                    type_hint: None,
                    scope: None,
                    source: None,
                    alias: None,
                    visibility: None,
                    is_async: false,
                    parent_class: None,
                });

                edges.push(CodeEdge {
                    source_id: parent_id.to_string(),
                    target_id: struct_id,
                    kind: EdgeKind::Contains,
                    file: file_path.to_string(),
                });
            }
            "trait_item" => {
                let name_node = match child.child_by_field_name("name") {
                    Some(n) => n,
                    None => continue,
                };
                let name = node_text(&name_node, source).to_string();
                let line_start = child.start_position().row + 1;
                let line_end = child.end_position().row + 1;
                let trait_id = make_id(file_path, "iface", &name, line_start);

                nodes.push(CodeNode {
                    id: trait_id.clone(),
                    kind: NodeKind::Interface,
                    name,
                    file: file_path.to_string(),
                    line_start,
                    line_end,
                    language: lang.to_string(),
                    docstring: None,
                    signature: None,
                    type_hint: None,
                    scope: None,
                    source: None,
                    alias: None,
                    visibility: None,
                    is_async: false,
                    parent_class: None,
                });

                edges.push(CodeEdge {
                    source_id: parent_id.to_string(),
                    target_id: trait_id,
                    kind: EdgeKind::Contains,
                    file: file_path.to_string(),
                });
            }
            "impl_item" => {
                // Extract methods from impl blocks
                if let Some(type_node) = child.child_by_field_name("type") {
                    let type_name = node_text(&type_node, source).to_string();
                    let impl_target_id = format!("class:{file_path}:{type_name}:0");

                    if let Some(body) = child.child_by_field_name("body") {
                        let mut bc = body.walk();
                        for member in body.named_children(&mut bc) {
                            if member.kind() == "function_item" {
                                if let Some(fn_name) = member.child_by_field_name("name") {
                                    let mname = node_text(&fn_name, source).to_string();
                                    let mls = member.start_position().row + 1;
                                    let mle = member.end_position().row + 1;
                                    let mid = make_id(file_path, "method", &mname, mls);

                                    let vis = {
                                        let src_slice = &source[member.byte_range()];
                                        if src_slice.starts_with("pub ") {
                                            Some("pub".to_string())
                                        } else {
                                            Some("private".to_string())
                                        }
                                    };

                                    nodes.push(CodeNode {
                                        id: mid.clone(),
                                        kind: NodeKind::Method,
                                        name: mname,
                                        file: file_path.to_string(),
                                        line_start: mls,
                                        line_end: mle,
                                        language: lang.to_string(),
                                        docstring: None,
                                        signature: None,
                                        type_hint: None,
                                        scope: None,
                                        source: None,
                                        alias: None,
                                        visibility: vis,
                                        is_async: false,
                                        parent_class: Some(type_name.clone()),
                                    });

                                    edges.push(CodeEdge {
                                        source_id: impl_target_id.clone(),
                                        target_id: mid,
                                        kind: EdgeKind::HasMethod,
                                        file: file_path.to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            "use_declaration" => {
                let text = node_text(&child, source);
                let line = child.start_position().row + 1;
                let import_id = make_id(file_path, "import", &text.replace(' ', "_"), line);

                nodes.push(CodeNode {
                    id: import_id.clone(),
                    kind: NodeKind::Import,
                    name: text.to_string(),
                    file: file_path.to_string(),
                    line_start: line,
                    line_end: line,
                    language: lang.to_string(),
                    docstring: None,
                    signature: None,
                    type_hint: None,
                    scope: None,
                    source: Some(text.to_string()),
                    alias: None,
                    visibility: None,
                    is_async: false,
                    parent_class: None,
                });

                edges.push(CodeEdge {
                    source_id: parent_id.to_string(),
                    target_id: import_id,
                    kind: EdgeKind::Imports,
                    file: file_path.to_string(),
                });
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::NemesisParser;
    use crate::models::Language;

    fn parse_and_extract(source: &str, lang: Language, path: &str) -> ExtractionResult {
        let mut parser = NemesisParser::new();
        let result = parser.parse_string(source, &lang, path).unwrap();
        extract(&result)
    }

    // ---- Python tests ----

    #[test]
    fn test_python_file_node() {
        let r = parse_and_extract("x = 1\n", Language::Python, "test.py");
        assert_eq!(r.nodes[0].kind, NodeKind::File);
        assert_eq!(r.nodes[0].name, "test.py");
    }

    #[test]
    fn test_python_class_extraction() {
        let src = "class Foo:\n    pass\n";
        let r = parse_and_extract(src, Language::Python, "test.py");
        let classes: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Class).collect();
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].name, "Foo");
        assert_eq!(classes[0].line_start, 1);
    }

    #[test]
    fn test_python_class_with_docstring() {
        let src = "class Foo:\n    \"\"\"A foo class.\"\"\"\n    pass\n";
        let r = parse_and_extract(src, Language::Python, "test.py");
        let cls = r.nodes.iter().find(|n| n.kind == NodeKind::Class).unwrap();
        assert_eq!(cls.docstring.as_deref(), Some("A foo class."));
    }

    #[test]
    fn test_python_function_extraction() {
        let src = "def hello(name: str) -> str:\n    return f\"Hello {name}\"\n";
        let r = parse_and_extract(src, Language::Python, "test.py");
        let funcs: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Function).collect();
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "hello");
        assert!(funcs[0].signature.as_ref().unwrap().contains("hello"));
    }

    #[test]
    fn test_python_method_extraction() {
        let src = "class Calc:\n    def add(self, a, b):\n        return a + b\n";
        let r = parse_and_extract(src, Language::Python, "test.py");
        let methods: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Method).collect();
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].name, "add");
        assert_eq!(methods[0].visibility.as_deref(), Some("public"));
    }

    #[test]
    fn test_python_private_method() {
        let src = "class Foo:\n    def __secret(self):\n        pass\n";
        let r = parse_and_extract(src, Language::Python, "test.py");
        let methods: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Method).collect();
        assert_eq!(methods[0].visibility.as_deref(), Some("private"));
    }

    #[test]
    fn test_python_import_extraction() {
        let src = "import os\nfrom pathlib import Path\n";
        let r = parse_and_extract(src, Language::Python, "test.py");
        let imports: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Import).collect();
        assert_eq!(imports.len(), 2);
    }

    #[test]
    fn test_python_inheritance_edge() {
        let src = "class Child(Parent):\n    pass\n";
        let r = parse_and_extract(src, Language::Python, "test.py");
        let inherits: Vec<_> = r.edges.iter().filter(|e| e.kind == EdgeKind::Inherits).collect();
        assert_eq!(inherits.len(), 1);
        assert!(inherits[0].source_id.contains("Child"));
    }

    #[test]
    fn test_python_contains_edges() {
        let src = "class Foo:\n    pass\ndef bar():\n    pass\n";
        let r = parse_and_extract(src, Language::Python, "test.py");
        let contains: Vec<_> = r.edges.iter().filter(|e| e.kind == EdgeKind::Contains).collect();
        assert!(contains.len() >= 2);
    }

    #[test]
    fn test_python_has_method_edge() {
        let src = "class Foo:\n    def bar(self):\n        pass\n";
        let r = parse_and_extract(src, Language::Python, "test.py");
        let hm: Vec<_> = r.edges.iter().filter(|e| e.kind == EdgeKind::HasMethod).collect();
        assert_eq!(hm.len(), 1);
    }

    // ---- TypeScript tests ----

    #[test]
    fn test_typescript_function_extraction() {
        let src = "function greet(name: string): void {\n  console.log(name);\n}\n";
        let r = parse_and_extract(src, Language::TypeScript, "test.ts");
        let funcs: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Function).collect();
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "greet");
    }

    #[test]
    fn test_typescript_class_extraction() {
        let src = "class Animal {\n  speak() {\n    return \"...\";\n  }\n}\n";
        let r = parse_and_extract(src, Language::TypeScript, "test.ts");
        let classes: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Class).collect();
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].name, "Animal");
    }

    #[test]
    fn test_typescript_interface_extraction() {
        let src = "interface Shape {\n  area(): number;\n}\n";
        let r = parse_and_extract(src, Language::TypeScript, "test.ts");
        let ifaces: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Interface).collect();
        assert_eq!(ifaces.len(), 1);
        assert_eq!(ifaces[0].name, "Shape");
    }

    #[test]
    fn test_typescript_method_has_method_edge() {
        let src = "class Dog {\n  bark() {\n    return \"woof\";\n  }\n}\n";
        let r = parse_and_extract(src, Language::TypeScript, "test.ts");
        let hm: Vec<_> = r.edges.iter().filter(|e| e.kind == EdgeKind::HasMethod).collect();
        assert_eq!(hm.len(), 1);
    }

    // ---- Rust tests ----

    #[test]
    fn test_rust_function_extraction() {
        let src = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let r = parse_and_extract(src, Language::Rust, "lib.rs");
        let funcs: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Function).collect();
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "add");
        assert_eq!(funcs[0].visibility.as_deref(), Some("pub"));
    }

    #[test]
    fn test_rust_struct_extraction() {
        let src = "struct Point {\n    x: f64,\n    y: f64,\n}\n";
        let r = parse_and_extract(src, Language::Rust, "lib.rs");
        let structs: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Class).collect();
        assert_eq!(structs.len(), 1);
        assert_eq!(structs[0].name, "Point");
    }

    #[test]
    fn test_rust_trait_extraction() {
        let src = "trait Drawable {\n    fn draw(&self);\n}\n";
        let r = parse_and_extract(src, Language::Rust, "lib.rs");
        let traits: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Interface).collect();
        assert_eq!(traits.len(), 1);
        assert_eq!(traits[0].name, "Drawable");
    }

    #[test]
    fn test_rust_impl_method_extraction() {
        let src = "struct Foo;\nimpl Foo {\n    pub fn bar(&self) {}\n    fn baz(&self) {}\n}\n";
        let r = parse_and_extract(src, Language::Rust, "lib.rs");
        let methods: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Method).collect();
        assert_eq!(methods.len(), 2);
        let bar = methods.iter().find(|m| m.name == "bar").unwrap();
        assert_eq!(bar.visibility.as_deref(), Some("pub"));
        assert_eq!(bar.parent_class.as_deref(), Some("Foo"));
    }

    #[test]
    fn test_rust_use_extraction() {
        let src = "use std::io::Read;\n";
        let r = parse_and_extract(src, Language::Rust, "lib.rs");
        let imports: Vec<_> = r.nodes.iter().filter(|n| n.kind == NodeKind::Import).collect();
        assert_eq!(imports.len(), 1);
    }
}
```

### Step 2 — Validate

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && cargo test`
Expected: ~20 extractor tests pass

### Step 3 — Commit

```bash
git add nemesis-parse/src/extractor.rs nemesis-parse/src/lib.rs
git commit -m "feat(parse): add AST extractor for Python, TypeScript, and Rust"
```

---

## Task 5: Full PyO3 Bindings (lib.rs update)

Expose `parse_file()`, `parse_string()`, `extract_nodes()`, `extract_edges()`, and `extract_all()` to Python.

**Files:**
- Modify: `nemesis-parse/src/lib.rs`

### Step 1 — Write the updated lib.rs

```rust
// nemesis-parse/src/lib.rs
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

/// nemesis-parse — Tree-sitter AST parser for Nemesis.
#[pymodule]
fn nemesis_parse(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
```

### Step 2 — Validate compilation

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && cargo check`
Expected: `Finished` without errors

### Step 3 — Build the Python extension

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && maturin develop --release 2>&1 | tail -3`
Expected: Successfully built wheel

### Step 4 — Quick smoke test from Python

Run: `cd /home/andreas/projects/nemesis && python3 -c "import nemesis_parse; print(nemesis_parse.version()); print(nemesis_parse.supported_languages())"`
Expected: `0.1.0` and `['python', 'typescript', 'tsx', 'rust']`

### Step 5 — Commit

```bash
git add nemesis-parse/src/lib.rs
git commit -m "feat(parse): expose full PyO3 API — parse_file, parse_string, extract_nodes, extract_edges"
```

---

## Task 6: Build and install the Rust extension into the venv

Ensures the extension is importable as `nemesis._nemesis_parse` for the Python bridge.

**Files:**
- Modify: `nemesis-parse/pyproject.toml` (if needed)

### Step 1 — Build with maturin develop

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && maturin develop --release`
Expected: Wheel installed into venv

### Step 2 — Validate Python import

Run: `cd /home/andreas/projects/nemesis && python3 -c "import nemesis_parse; print(nemesis_parse.version())"`
Expected: `0.1.0`

### Step 3 — Validate parse_string from Python

Run:
```bash
cd /home/andreas/projects/nemesis && python3 -c "
import nemesis_parse, json
result = json.loads(nemesis_parse.parse_string('def hello(): pass', 'python', 'test.py'))
print(f\"nodes: {len(result['nodes'])}, edges: {len(result['edges'])}\")
print([n['name'] for n in result['nodes']])
"
```
Expected: `nodes: 2, edges: 1` and `['test.py', 'hello']`

### Step 4 — Commit

```bash
git commit --allow-empty -m "chore(parse): verify maturin build and Python extension import"
```

---

## Task 7: Python Bridge — Data Classes (nemesis/parser/models.py)

Python-side dataclasses mirroring the Rust models, used by the bridge.

**Files:**
- Create: `nemesis/parser/models.py`
- Create: `tests/test_parser/__init__.py`
- Create: `tests/test_parser/test_models.py`

### Step 1 — Write the failing test

```python
# tests/test_parser/__init__.py
```

```python
# tests/test_parser/test_models.py
"""Tests for parser Python data models."""

from nemesis.parser.models import CodeNode, CodeEdge, ExtractionResult, NodeKind, EdgeKind


class TestNodeKind:
    def test_all_kinds_exist(self) -> None:
        kinds = [NodeKind.FILE, NodeKind.MODULE, NodeKind.CLASS, NodeKind.FUNCTION,
                 NodeKind.METHOD, NodeKind.INTERFACE, NodeKind.VARIABLE, NodeKind.IMPORT]
        assert len(kinds) == 8

    def test_kind_values(self) -> None:
        assert NodeKind.FILE == "File"
        assert NodeKind.FUNCTION == "Function"


class TestEdgeKind:
    def test_all_kinds_exist(self) -> None:
        kinds = [EdgeKind.CONTAINS, EdgeKind.HAS_METHOD, EdgeKind.INHERITS,
                 EdgeKind.IMPLEMENTS, EdgeKind.CALLS, EdgeKind.IMPORTS,
                 EdgeKind.RETURNS, EdgeKind.ACCEPTS]
        assert len(kinds) == 8

    def test_kind_values(self) -> None:
        assert EdgeKind.CONTAINS == "CONTAINS"
        assert EdgeKind.HAS_METHOD == "HAS_METHOD"


class TestCodeNode:
    def test_creation(self) -> None:
        node = CodeNode(
            id="func:test.py:hello:1",
            kind=NodeKind.FUNCTION,
            name="hello",
            file="test.py",
            line_start=1,
            line_end=3,
            language="python",
        )
        assert node.name == "hello"
        assert node.kind == NodeKind.FUNCTION
        assert node.docstring is None
        assert node.is_async is False

    def test_from_dict(self) -> None:
        data = {
            "id": "class:t.py:Foo:1",
            "kind": "Class",
            "name": "Foo",
            "file": "t.py",
            "line_start": 1,
            "line_end": 5,
            "language": "python",
            "docstring": "A foo.",
            "is_async": False,
        }
        node = CodeNode.from_dict(data)
        assert node.name == "Foo"
        assert node.docstring == "A foo."


class TestCodeEdge:
    def test_creation(self) -> None:
        edge = CodeEdge(
            source_id="file:test.py",
            target_id="func:test.py:hello:1",
            kind=EdgeKind.CONTAINS,
            file="test.py",
        )
        assert edge.kind == EdgeKind.CONTAINS

    def test_from_dict(self) -> None:
        data = {
            "source_id": "a",
            "target_id": "b",
            "kind": "CONTAINS",
            "file": "t.py",
        }
        edge = CodeEdge.from_dict(data)
        assert edge.kind == EdgeKind.CONTAINS


class TestExtractionResult:
    def test_creation(self) -> None:
        result = ExtractionResult(
            file="test.py",
            language="python",
            nodes=[],
            edges=[],
        )
        assert result.file == "test.py"
        assert len(result.nodes) == 0

    def test_from_dict(self) -> None:
        data = {
            "file": "t.py",
            "language": "python",
            "nodes": [
                {"id": "file:t.py", "kind": "File", "name": "t.py",
                 "file": "t.py", "line_start": 1, "line_end": 1,
                 "language": "python", "is_async": False},
            ],
            "edges": [],
        }
        result = ExtractionResult.from_dict(data)
        assert len(result.nodes) == 1
        assert result.nodes[0].kind == NodeKind.FILE

    def test_node_count_property(self) -> None:
        node = CodeNode(id="x", kind=NodeKind.FILE, name="x", file="x",
                        line_start=1, line_end=1, language="python")
        result = ExtractionResult(file="x", language="python",
                                  nodes=[node], edges=[])
        assert result.node_count == 1
        assert result.edge_count == 0
```

### Step 2 — Run test to verify it fails

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_models.py -v`
Expected: FAIL with ImportError

### Step 3 — Write implementation

```python
# nemesis/parser/models.py
"""Python data models mirroring the Rust nemesis-parse structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class NodeKind:
    """Code node type constants."""

    FILE = "File"
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    INTERFACE = "Interface"
    VARIABLE = "Variable"
    IMPORT = "Import"


class EdgeKind:
    """Code edge type constants."""

    CONTAINS = "CONTAINS"
    HAS_METHOD = "HAS_METHOD"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    RETURNS = "RETURNS"
    ACCEPTS = "ACCEPTS"


@dataclass
class CodeNode:
    """A structured code node extracted from an AST."""

    id: str
    kind: str
    name: str
    file: str
    line_start: int
    line_end: int
    language: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    type_hint: Optional[str] = None
    scope: Optional[str] = None
    source: Optional[str] = None
    alias: Optional[str] = None
    visibility: Optional[str] = None
    is_async: bool = False
    parent_class: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> CodeNode:
        """Create a CodeNode from a dictionary (e.g. from JSON)."""
        return cls(
            id=data["id"],
            kind=data["kind"],
            name=data["name"],
            file=data["file"],
            line_start=data["line_start"],
            line_end=data["line_end"],
            language=data["language"],
            docstring=data.get("docstring"),
            signature=data.get("signature"),
            type_hint=data.get("type_hint"),
            scope=data.get("scope"),
            source=data.get("source"),
            alias=data.get("alias"),
            visibility=data.get("visibility"),
            is_async=data.get("is_async", False),
            parent_class=data.get("parent_class"),
        )


@dataclass
class CodeEdge:
    """A directed edge between two code nodes."""

    source_id: str
    target_id: str
    kind: str
    file: str

    @classmethod
    def from_dict(cls, data: dict) -> CodeEdge:
        """Create a CodeEdge from a dictionary (e.g. from JSON)."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            kind=data["kind"],
            file=data["file"],
        )


@dataclass
class ExtractionResult:
    """Complete extraction result for a single file."""

    file: str
    language: str
    nodes: list[CodeNode] = field(default_factory=list)
    edges: list[CodeEdge] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @classmethod
    def from_dict(cls, data: dict) -> ExtractionResult:
        """Create an ExtractionResult from a dictionary (e.g. from JSON)."""
        nodes = [CodeNode.from_dict(n) for n in data.get("nodes", [])]
        edges = [CodeEdge.from_dict(e) for e in data.get("edges", [])]
        return cls(
            file=data["file"],
            language=data["language"],
            nodes=nodes,
            edges=edges,
        )
```

### Step 4 — Run test to verify pass

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_models.py -v`
Expected: PASS (12 tests)

### Step 5 — Commit

```bash
git add nemesis/parser/models.py tests/test_parser/
git commit -m "feat(parser): add Python data models for code nodes, edges, and extraction results"
```

---

## Task 8: Python Bridge (nemesis/parser/bridge.py)

Python wrapper around the Rust extension with error handling, caching, and Pythonic API.

**Files:**
- Create: `nemesis/parser/bridge.py`
- Create: `tests/test_parser/test_bridge.py`

### Step 1 — Write the failing test

```python
# tests/test_parser/test_bridge.py
"""Tests for the Python bridge to nemesis-parse Rust extension."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemesis.parser.bridge import ParserBridge, ParserError
from nemesis.parser.models import CodeNode, ExtractionResult, NodeKind, EdgeKind


class TestParserBridgeInit:
    def test_bridge_creation(self) -> None:
        bridge = ParserBridge()
        assert bridge is not None

    def test_bridge_has_native_module(self) -> None:
        bridge = ParserBridge()
        assert bridge.native_available is True or bridge.native_available is False

    def test_supported_languages(self) -> None:
        bridge = ParserBridge()
        langs = bridge.supported_languages()
        assert "python" in langs


class TestParserBridgeDetect:
    def test_detect_python(self) -> None:
        bridge = ParserBridge()
        assert bridge.detect_language("app.py") == "python"

    def test_detect_typescript(self) -> None:
        bridge = ParserBridge()
        assert bridge.detect_language("app.ts") == "typescript"

    def test_detect_rust(self) -> None:
        bridge = ParserBridge()
        assert bridge.detect_language("main.rs") == "rust"

    def test_detect_unsupported(self) -> None:
        bridge = ParserBridge()
        with pytest.raises(ParserError):
            bridge.detect_language("file.java")


class TestParserBridgeParseString:
    def test_parse_python_string(self) -> None:
        bridge = ParserBridge()
        result = bridge.parse_string("def hello(): pass\n", "python", "test.py")
        assert isinstance(result, ExtractionResult)
        assert result.file == "test.py"
        assert result.language == "python"
        assert result.node_count >= 2  # File + Function

    def test_parse_python_class(self) -> None:
        bridge = ParserBridge()
        src = "class Foo:\n    def bar(self):\n        pass\n"
        result = bridge.parse_string(src, "python", "test.py")
        class_nodes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        method_nodes = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        assert len(class_nodes) == 1
        assert len(method_nodes) == 1
        assert class_nodes[0].name == "Foo"

    def test_parse_typescript_string(self) -> None:
        bridge = ParserBridge()
        src = "function greet(name: string): void { console.log(name); }\n"
        result = bridge.parse_string(src, "typescript", "test.ts")
        assert result.language == "typescript"
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert len(funcs) == 1

    def test_parse_rust_string(self) -> None:
        bridge = ParserBridge()
        src = "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
        result = bridge.parse_string(src, "rust", "lib.rs")
        assert result.language == "rust"
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert len(funcs) == 1

    def test_parse_empty_source(self) -> None:
        bridge = ParserBridge()
        result = bridge.parse_string("", "python", "empty.py")
        assert result.node_count >= 1  # At least File node

    def test_parse_invalid_language_raises(self) -> None:
        bridge = ParserBridge()
        with pytest.raises(ParserError):
            bridge.parse_string("code", "java", "test.java")


class TestParserBridgeParseFile:
    def test_parse_python_file(self, sample_python_file: Path) -> None:
        bridge = ParserBridge()
        result = bridge.parse_file(str(sample_python_file))
        assert result.language == "python"
        class_nodes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        assert len(class_nodes) >= 1
        assert any(n.name == "Calculator" for n in class_nodes)

    def test_parse_nonexistent_file_raises(self) -> None:
        bridge = ParserBridge()
        with pytest.raises(ParserError):
            bridge.parse_file("/nonexistent/path/file.py")

    def test_parse_file_extracts_methods(self, sample_python_file: Path) -> None:
        bridge = ParserBridge()
        result = bridge.parse_file(str(sample_python_file))
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_names = {m.name for m in methods}
        assert "add" in method_names
        assert "subtract" in method_names

    def test_parse_file_extracts_edges(self, sample_python_file: Path) -> None:
        bridge = ParserBridge()
        result = bridge.parse_file(str(sample_python_file))
        contains = [e for e in result.edges if e.kind == EdgeKind.CONTAINS]
        has_method = [e for e in result.edges if e.kind == EdgeKind.HAS_METHOD]
        assert len(contains) >= 1
        assert len(has_method) >= 1


class TestParserBridgeEdgeCases:
    def test_parse_syntax_error_still_produces_partial(self) -> None:
        """Tree-sitter is error-tolerant; partial results are expected."""
        bridge = ParserBridge()
        src = "def broken(:\n    pass\n"
        result = bridge.parse_string(src, "python", "broken.py")
        # Should still produce a File node at minimum
        assert result.node_count >= 1

    def test_parse_large_source(self) -> None:
        bridge = ParserBridge()
        # Generate a file with 100 functions
        lines = []
        for i in range(100):
            lines.append(f"def func_{i}():\n    pass\n")
        src = "\n".join(lines)
        result = bridge.parse_string(src, "python", "big.py")
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert len(funcs) == 100
```

### Step 2 — Run test to verify it fails

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_bridge.py -v`
Expected: FAIL with ImportError

### Step 3 — Write implementation

```python
# nemesis/parser/bridge.py
"""Python bridge to the nemesis-parse Rust extension.

Provides a Pythonic API around the native Tree-sitter parser.
Falls back to helpful error messages if the extension is not compiled.
"""

from __future__ import annotations

import json
from pathlib import Path

from nemesis.parser.models import CodeEdge, CodeNode, ExtractionResult


class ParserError(Exception):
    """Raised when parsing or extraction fails."""


# Try to import the native Rust extension
_native = None
try:
    import nemesis_parse as _native  # type: ignore[import-not-found]
except ImportError:
    _native = None


class ParserBridge:
    """High-level Python interface to the nemesis-parse Rust extension.

    Usage::

        bridge = ParserBridge()
        result = bridge.parse_file("src/main.py")
        for node in result.nodes:
            print(f"{node.kind}: {node.name}")
    """

    def __init__(self) -> None:
        self._native = _native

    @property
    def native_available(self) -> bool:
        """Whether the Rust native extension is available."""
        return self._native is not None

    def _require_native(self) -> None:
        if self._native is None:
            raise ParserError(
                "nemesis-parse native extension not available. "
                "Build it with: cd nemesis-parse && maturin develop --release"
            )

    def supported_languages(self) -> list[str]:
        """Return the list of supported programming languages."""
        if self._native is not None:
            return self._native.supported_languages()
        return ["python", "typescript", "tsx", "rust"]

    def detect_language(self, file_path: str) -> str:
        """Detect the programming language from a file extension.

        Args:
            file_path: Path to the file (only extension is used).

        Returns:
            Language identifier string.

        Raises:
            ParserError: If the file extension is not supported.
        """
        if self._native is not None:
            try:
                return self._native.detect_language(file_path)
            except ValueError as e:
                raise ParserError(str(e)) from e

        # Fallback detection
        ext = Path(file_path).suffix.lstrip(".")
        mapping = {"py": "python", "ts": "typescript", "tsx": "tsx", "rs": "rust"}
        if ext not in mapping:
            raise ParserError(f"Unsupported file extension: {ext}")
        return mapping[ext]

    def parse_string(
        self,
        source: str,
        language: str,
        file_path: str,
    ) -> ExtractionResult:
        """Parse a source string and extract code nodes and edges.

        Args:
            source: The source code string.
            language: Language identifier (python, typescript, tsx, rust).
            file_path: Virtual file path for node IDs.

        Returns:
            ExtractionResult with nodes and edges.

        Raises:
            ParserError: If parsing fails or language is unsupported.
        """
        self._require_native()
        try:
            json_str = self._native.parse_string(source, language, file_path)
            data = json.loads(json_str)
            return ExtractionResult.from_dict(data)
        except (ValueError, json.JSONDecodeError) as e:
            raise ParserError(f"Parse failed: {e}") from e

    def parse_file(self, file_path: str) -> ExtractionResult:
        """Parse a file from disk, auto-detecting the language.

        Args:
            file_path: Absolute or relative path to the source file.

        Returns:
            ExtractionResult with nodes and edges.

        Raises:
            ParserError: If the file doesn't exist, can't be read, or language
                        is unsupported.
        """
        self._require_native()
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"File not found: {file_path}")

        try:
            json_str = self._native.parse_file(str(path))
            data = json.loads(json_str)
            return ExtractionResult.from_dict(data)
        except (ValueError, json.JSONDecodeError, OSError) as e:
            raise ParserError(f"Parse failed for {file_path}: {e}") from e

    def extract_nodes(
        self,
        source: str,
        language: str,
        file_path: str,
    ) -> list[CodeNode]:
        """Extract only code nodes from a source string.

        Args:
            source: The source code string.
            language: Language identifier.
            file_path: Virtual file path for node IDs.

        Returns:
            List of CodeNode objects.
        """
        self._require_native()
        try:
            json_str = self._native.extract_nodes(source, language, file_path)
            data = json.loads(json_str)
            return [CodeNode.from_dict(n) for n in data]
        except (ValueError, json.JSONDecodeError) as e:
            raise ParserError(f"Node extraction failed: {e}") from e

    def extract_edges(
        self,
        source: str,
        language: str,
        file_path: str,
    ) -> list[CodeEdge]:
        """Extract only code edges from a source string.

        Args:
            source: The source code string.
            language: Language identifier.
            file_path: Virtual file path for node IDs.

        Returns:
            List of CodeEdge objects.
        """
        self._require_native()
        try:
            json_str = self._native.extract_edges(source, language, file_path)
            data = json.loads(json_str)
            return [CodeEdge.from_dict(e) for e in data]
        except (ValueError, json.JSONDecodeError) as e:
            raise ParserError(f"Edge extraction failed: {e}") from e
```

### Step 4 — Run test to verify pass

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_bridge.py -v`
Expected: PASS (18 tests)

Hinweis: Die Tests benoetigen die kompilierte Rust-Extension. Vor dem Test: `cd nemesis-parse && maturin develop --release`

### Step 5 — Commit

```bash
git add nemesis/parser/bridge.py tests/test_parser/test_bridge.py
git commit -m "feat(parser): add Python bridge with parse_file, parse_string, extract_nodes, extract_edges"
```

---

## Task 9: Update parser __init__.py with public API

Clean public exports from the parser subpackage.

**Files:**
- Modify: `nemesis/parser/__init__.py`
- Create: `tests/test_parser/test_parser_init.py`

### Step 1 — Write the failing test

```python
# tests/test_parser/test_parser_init.py
"""Tests for parser subpackage public API."""


def test_parser_exports_bridge() -> None:
    from nemesis.parser import ParserBridge
    assert ParserBridge is not None


def test_parser_exports_error() -> None:
    from nemesis.parser import ParserError
    assert ParserError is not None


def test_parser_exports_models() -> None:
    from nemesis.parser import CodeNode, CodeEdge, ExtractionResult
    assert CodeNode is not None
    assert CodeEdge is not None
    assert ExtractionResult is not None


def test_parser_exports_kinds() -> None:
    from nemesis.parser import NodeKind, EdgeKind
    assert NodeKind.FILE == "File"
    assert EdgeKind.CONTAINS == "CONTAINS"
```

### Step 2 — Run test to verify it fails

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_parser_init.py -v`
Expected: FAIL with ImportError

### Step 3 — Write implementation

```python
# nemesis/parser/__init__.py
"""Parser module — PyO3 bridge to nemesis-parse Rust crate.

Public API::

    from nemesis.parser import ParserBridge, ParserError
    from nemesis.parser import CodeNode, CodeEdge, ExtractionResult
    from nemesis.parser import NodeKind, EdgeKind
"""

from nemesis.parser.bridge import ParserBridge, ParserError
from nemesis.parser.models import (
    CodeEdge,
    CodeNode,
    EdgeKind,
    ExtractionResult,
    NodeKind,
)

__all__ = [
    "ParserBridge",
    "ParserError",
    "CodeNode",
    "CodeEdge",
    "ExtractionResult",
    "NodeKind",
    "EdgeKind",
]
```

### Step 4 — Run test to verify pass

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_parser_init.py -v`
Expected: PASS (4 tests)

### Step 5 — Commit

```bash
git add nemesis/parser/__init__.py tests/test_parser/test_parser_init.py
git commit -m "feat(parser): export clean public API from parser subpackage"
```

---

## Task 10: Test Fixtures for Parser Tests

Add shared fixtures (conftest) for parser test directory.

**Files:**
- Create: `tests/test_parser/conftest.py`

### Step 1 — Write the fixtures

```python
# tests/test_parser/conftest.py
"""Shared fixtures for parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_typescript_file(tmp_project: Path) -> Path:
    """Create a sample TypeScript file for testing."""
    code = '''import { Request, Response } from "express";

interface User {
  id: number;
  name: string;
}

class UserService {
  private users: User[] = [];

  getUser(id: number): User | undefined {
    return this.users.find(u => u.id === id);
  }

  addUser(user: User): void {
    this.users.push(user);
  }
}

function createApp(): void {
  const service = new UserService();
  console.log("App started");
}
'''
    file_path = tmp_project / "app.ts"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_rust_file(tmp_project: Path) -> Path:
    """Create a sample Rust file for testing."""
    code = '''use std::fmt;

pub trait Greetable {
    fn greet(&self) -> String;
}

pub struct Person {
    pub name: String,
    age: u32,
}

impl Person {
    pub fn new(name: &str, age: u32) -> Self {
        Person {
            name: name.to_string(),
            age,
        }
    }

    fn is_adult(&self) -> bool {
        self.age >= 18
    }
}

impl Greetable for Person {
    fn greet(&self) -> String {
        format!("Hello, I am {}", self.name)
    }
}

pub fn create_person(name: &str) -> Person {
    Person::new(name, 30)
}
'''
    file_path = tmp_project / "person.rs"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_complex_python(tmp_project: Path) -> Path:
    """Create a complex Python file with inheritance and imports."""
    code = '''"""Complex module for testing."""

import os
from pathlib import Path
from typing import List, Optional

MAX_SIZE: int = 1024


class BaseService:
    """Base class for services."""

    def __init__(self, name: str):
        self.name = name

    def _internal_method(self):
        pass

    def __repr__(self):
        return f"BaseService({self.name})"


class UserService(BaseService):
    """Service for user operations."""

    def __init__(self, name: str, db_url: str):
        super().__init__(name)
        self.db_url = db_url

    def get_user(self, user_id: int) -> Optional[dict]:
        """Get a user by ID."""
        return None

    async def fetch_users(self) -> List[dict]:
        """Async fetch all users."""
        return []


def create_service(name: str = "default") -> UserService:
    """Factory function."""
    return UserService(name, "sqlite:///db.sqlite")
'''
    file_path = tmp_project / "services.py"
    file_path.write_text(code)
    return file_path
```

### Step 2 — Validate fixtures load

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/ --collect-only 2>&1 | tail -5`
Expected: Shows collected tests without errors

### Step 3 — Commit

```bash
git add tests/test_parser/conftest.py
git commit -m "test(parser): add TypeScript, Rust, and complex Python fixtures for parser tests"
```

---

## Task 11: Multi-Language Integration Tests

End-to-end tests that parse real files in all three languages through the bridge.

**Files:**
- Create: `tests/test_parser/test_integration.py`

### Step 1 — Write the integration tests

```python
# tests/test_parser/test_integration.py
"""Integration tests for multi-language parsing through the Python bridge."""

from pathlib import Path

import pytest

from nemesis.parser import ParserBridge, NodeKind, EdgeKind, ExtractionResult


@pytest.fixture
def bridge() -> ParserBridge:
    return ParserBridge()


class TestPythonIntegration:
    """Full parsing integration for Python files."""

    def test_parse_complex_python_file(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        assert result.language == "python"
        assert result.node_count > 5

    def test_extracts_classes_with_inheritance(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        classes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        class_names = {c.name for c in classes}
        assert "BaseService" in class_names
        assert "UserService" in class_names

    def test_extracts_inherits_edge(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        inherits = [e for e in result.edges if e.kind == EdgeKind.INHERITS]
        assert len(inherits) >= 1
        # UserService inherits BaseService
        assert any("UserService" in e.source_id for e in inherits)

    def test_extracts_imports(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        imports = [n for n in result.nodes if n.kind == NodeKind.IMPORT]
        assert len(imports) >= 2

    def test_extracts_methods_with_visibility(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_map = {m.name: m for m in methods}
        assert "get_user" in method_map
        assert method_map["get_user"].visibility == "public"
        assert "_internal_method" in method_map
        assert method_map["_internal_method"].visibility == "protected"

    def test_extracts_standalone_functions(
        self, bridge: ParserBridge, sample_complex_python: Path
    ) -> None:
        result = bridge.parse_file(str(sample_complex_python))
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        func_names = {f.name for f in funcs}
        assert "create_service" in func_names


class TestTypeScriptIntegration:
    """Full parsing integration for TypeScript files."""

    def test_parse_typescript_file(
        self, bridge: ParserBridge, sample_typescript_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_typescript_file))
        assert result.language == "typescript"
        assert result.node_count > 3

    def test_extracts_interface(
        self, bridge: ParserBridge, sample_typescript_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_typescript_file))
        ifaces = [n for n in result.nodes if n.kind == NodeKind.INTERFACE]
        assert len(ifaces) >= 1
        assert any(i.name == "User" for i in ifaces)

    def test_extracts_class_with_methods(
        self, bridge: ParserBridge, sample_typescript_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_typescript_file))
        classes = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        assert any(c.name == "UserService" for c in classes)
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_names = {m.name for m in methods}
        assert "getUser" in method_names
        assert "addUser" in method_names

    def test_extracts_function(
        self, bridge: ParserBridge, sample_typescript_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_typescript_file))
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        assert any(f.name == "createApp" for f in funcs)


class TestRustIntegration:
    """Full parsing integration for Rust files."""

    def test_parse_rust_file(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        assert result.language == "rust"
        assert result.node_count > 3

    def test_extracts_struct_as_class(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        structs = [n for n in result.nodes if n.kind == NodeKind.CLASS]
        assert any(s.name == "Person" for s in structs)

    def test_extracts_trait_as_interface(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        traits = [n for n in result.nodes if n.kind == NodeKind.INTERFACE]
        assert any(t.name == "Greetable" for t in traits)

    def test_extracts_impl_methods(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        methods = [n for n in result.nodes if n.kind == NodeKind.METHOD]
        method_names = {m.name for m in methods}
        assert "new" in method_names
        assert "is_adult" in method_names

    def test_extracts_pub_function(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        funcs = [n for n in result.nodes if n.kind == NodeKind.FUNCTION]
        create_fn = next((f for f in funcs if f.name == "create_person"), None)
        assert create_fn is not None
        assert create_fn.visibility == "pub"

    def test_has_method_edges(
        self, bridge: ParserBridge, sample_rust_file: Path
    ) -> None:
        result = bridge.parse_file(str(sample_rust_file))
        hm = [e for e in result.edges if e.kind == EdgeKind.HAS_METHOD]
        assert len(hm) >= 2  # new, is_adult from impl Person
```

### Step 2 — Run tests

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/test_integration.py -v`
Expected: PASS (~17 tests)

Hinweis: Benoetigt kompilierte Rust-Extension (`maturin develop --release`).

### Step 3 — Commit

```bash
git add tests/test_parser/test_integration.py
git commit -m "test(parser): add multi-language integration tests for Python, TypeScript, and Rust"
```

---

## Task 12: Ruff-Konformitaet und finaler Gesamttest

Linting, Formatting, und finaler Testlauf ueber alle Parser-Tests.

**Files:**
- Modify: alle bestehenden `.py` Dateien im `nemesis/parser/` und `tests/test_parser/` (falls noetig)

### Step 1 — Ruff Check

Run: `cd /home/andreas/projects/nemesis && python3 -m ruff check nemesis/parser/ tests/test_parser/`
Expected: `All checks passed!`

### Step 2 — Ruff Format

Run: `cd /home/andreas/projects/nemesis && python3 -m ruff format nemesis/parser/ tests/test_parser/`

### Step 3 — Cargo clippy

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && cargo clippy -- -D warnings`
Expected: No warnings

### Step 4 — Full Rust test suite

Run: `cd /home/andreas/projects/nemesis/nemesis-parse && cargo test`
Expected: All Rust tests pass (~36 tests)

### Step 5 — Full Python test suite

Run: `cd /home/andreas/projects/nemesis && python3 -m pytest tests/test_parser/ -v --tb=short`
Expected: All Python tests pass (~51 tests)

### Step 6 — Commit

```bash
git add -A
git commit -m "style(parse): apply ruff formatting and clippy fixes to parser module"
```

---

## Zusammenfassung

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 1 | Rust Data Models (models.rs) | `nemesis-parse/src/models.rs` | 5 Rust-Tests |
| 2 | Language Grammar Bindings | `nemesis-parse/src/languages/{mod,python,typescript,rust_lang}.rs` | 4 Rust-Tests |
| 3 | Multi-Language Parser (parser.rs) | `nemesis-parse/src/parser.rs` | 7 Rust-Tests |
| 4 | AST Extractor (extractor.rs) | `nemesis-parse/src/extractor.rs` | 20 Rust-Tests |
| 5 | PyO3 Bindings (lib.rs) | `nemesis-parse/src/lib.rs` | cargo check |
| 6 | Build & Install Extension | maturin develop | Smoke-Test |
| 7 | Python Data Models | `nemesis/parser/models.py` | 12 Python-Tests |
| 8 | Python Bridge (bridge.py) | `nemesis/parser/bridge.py` | 18 Python-Tests |
| 9 | Parser __init__.py | `nemesis/parser/__init__.py` | 4 Python-Tests |
| 10 | Test Fixtures | `tests/test_parser/conftest.py` | Fixture-Validierung |
| 11 | Multi-Language Integration Tests | `tests/test_parser/test_integration.py` | 17 Python-Tests |
| 12 | Ruff + Clippy + Gesamttest | Bestehende Dateien | Lint + ~88 Tests gesamt |

**Gesamt: 12 Tasks, ~36 Rust-Tests + ~51 Python-Tests = ~87 Tests, 12 Commits**
