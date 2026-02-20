use tree_sitter::Language as TsLanguage;

pub fn language() -> TsLanguage {
    tree_sitter_python::LANGUAGE.into()
}

/// Tree-sitter query patterns for Python code extraction.
#[allow(dead_code)]
pub const CLASS_QUERY: &str = r#"
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases
  body: (block) @class.body) @class.def
"#;

#[allow(dead_code)]
pub const FUNCTION_QUERY: &str = r#"
(function_definition
  name: (identifier) @func.name
  parameters: (parameters) @func.params
  return_type: (type)? @func.return
  body: (block) @func.body) @func.def
"#;

#[allow(dead_code)]
pub const IMPORT_QUERY: &str = r#"
[
  (import_statement
    name: (dotted_name) @import.name) @import.stmt
  (import_from_statement
    module_name: (dotted_name) @import.module
    name: (dotted_name) @import.name) @import.stmt
]
"#;

#[allow(dead_code)]
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
