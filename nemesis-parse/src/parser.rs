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
            .parse_string(
                "fn main() { println!(\"hello\"); }\n",
                &Language::Rust,
                "main.rs",
            )
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
