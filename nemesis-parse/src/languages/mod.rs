pub mod python;
pub mod rust_lang;
pub mod typescript;

use crate::models::Language;
use tree_sitter::Language as TsLanguage;

/// Return the tree-sitter Language grammar for a given Language enum.
pub fn get_grammar(lang: &Language) -> TsLanguage {
    match lang {
        Language::Python => python::language(),
        Language::TypeScript => typescript::language_typescript(),
        Language::Tsx => typescript::language_tsx(),
        Language::Rust => rust_lang::language(),
    }
}
