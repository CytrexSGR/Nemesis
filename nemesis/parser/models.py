"""Python data models mirroring the Rust nemesis-parse structures."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    docstring: str | None = None
    signature: str | None = None
    type_hint: str | None = None
    scope: str | None = None
    source: str | None = None
    alias: str | None = None
    visibility: str | None = None
    is_async: bool = False
    parent_class: str | None = None

    @property
    def node_type(self) -> str:
        """Alias for kind — used by chunker and other consumers."""
        return self.kind

    @property
    def start_line(self) -> int:
        """Alias for line_start — used by chunker."""
        return self.line_start

    @property
    def end_line(self) -> int:
        """Alias for line_end — used by chunker."""
        return self.line_end

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

    @property
    def edge_type(self) -> str:
        """Alias for kind — used by graph adapter consumers."""
        return self.kind

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
