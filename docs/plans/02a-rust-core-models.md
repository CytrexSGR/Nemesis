# Rust Parser — Arbeitspaket B1: Core Models + Grammars + Parser

> **Arbeitspaket B1** — Teil 1 von 4 des Rust Parser Plans

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rust-seitige Datenmodelle (models.rs), Tree-sitter Grammar-Bindings (languages/) und den Multi-Language-Parser (parser.rs) implementieren. (Tasks 1, 2, 3)

**Tech Stack:** Rust, Tree-sitter 0.24, tree-sitter-python 0.23, tree-sitter-typescript 0.23, tree-sitter-rust 0.23, serde, serde_json, thiserror

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

## Zusammenfassung B1

| Task | Beschreibung | Dateien | Tests |
|------|-------------|---------|-------|
| 1 | Rust Data Models (models.rs) | `nemesis-parse/src/models.rs` | 5 Rust-Tests |
| 2 | Language Grammar Bindings | `nemesis-parse/src/languages/{mod,python,typescript,rust_lang}.rs` | 4 Rust-Tests |
| 3 | Multi-Language Parser (parser.rs) | `nemesis-parse/src/parser.rs` | 7 Rust-Tests |

**Gesamt B1: 3 Tasks, 16 Rust-Tests, 3 Commits**

---

**Navigation:**
- Naechstes Paket: [B2 — Rust Extraction](02b-rust-extraction.md)
- Gesamtplan: [02-rust-parser.md](02-rust-parser.md)
