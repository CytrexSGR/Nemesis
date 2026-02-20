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
