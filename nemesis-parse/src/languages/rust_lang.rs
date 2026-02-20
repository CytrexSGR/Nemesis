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
