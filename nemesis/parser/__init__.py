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
