use tree_sitter::Language as TsLanguage;

pub fn language_typescript() -> TsLanguage {
    tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
}

pub fn language_tsx() -> TsLanguage {
    tree_sitter_typescript::LANGUAGE_TSX.into()
}

#[allow(dead_code)]
pub const CLASS_QUERY: &str = r#"
(class_declaration
  name: (type_identifier) @class.name
  body: (class_body) @class.body) @class.def
"#;

#[allow(dead_code)]
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

#[allow(dead_code)]
pub const INTERFACE_QUERY: &str = r#"
(interface_declaration
  name: (type_identifier) @iface.name
  body: (interface_body) @iface.body) @iface.def
"#;

#[allow(dead_code)]
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
        parser
            .set_language(&lang)
            .expect("TypeScript grammar must load");
    }

    #[test]
    fn test_tsx_grammar_loads() {
        let lang = language_tsx();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&lang).expect("TSX grammar must load");
    }
}
