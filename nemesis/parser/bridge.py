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
    from nemesis import _nemesis_parse as _native  # type: ignore[import-not-found]
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
