"""Tests for Nemesis CLI skeleton and command logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from nemesis.core.cli import main
from nemesis.indexer.models import IndexResult
from nemesis.vector.store import SearchResult

# ---------------------------------------------------------------------------
# Existing skeleton tests
# ---------------------------------------------------------------------------


def test_cli_main_group():
    """The main CLI group exists and shows help."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Nemesis" in result.output


def test_cli_version_flag():
    """--version flag prints the version."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_index_command_exists():
    """The 'index' command is registered."""
    runner = CliRunner()
    result = runner.invoke(main, ["index", "--help"])
    assert result.exit_code == 0
    assert "Index" in result.output or "index" in result.output.lower()


def test_cli_query_command_exists():
    """The 'query' command is registered."""
    runner = CliRunner()
    result = runner.invoke(main, ["query", "--help"])
    assert result.exit_code == 0


def test_cli_watch_command_exists():
    """The 'watch' command is registered."""
    runner = CliRunner()
    result = runner.invoke(main, ["watch", "--help"])
    assert result.exit_code == 0


def test_cli_serve_command_exists():
    """The 'serve' command is registered."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Helper: mock NemesisEngine as context manager
# ---------------------------------------------------------------------------


def _make_mock_engine() -> MagicMock:
    """Create a mock NemesisEngine that works as a context manager."""
    engine = MagicMock()
    engine.__enter__ = MagicMock(return_value=engine)
    engine.__exit__ = MagicMock(return_value=False)
    return engine


# ---------------------------------------------------------------------------
# TestIndexCommand
# ---------------------------------------------------------------------------


class TestIndexCommand:
    """Tests for the 'index' command with real logic."""

    @patch("nemesis.core.cli.NemesisEngine")
    def test_index_calls_pipeline(self, mock_engine_cls: MagicMock, tmp_path: Path):
        """index command calls pipeline.index_project on the engine."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        engine.pipeline.index_project.return_value = IndexResult(
            files_indexed=0,
            nodes_created=0,
            edges_created=0,
            chunks_created=0,
            embeddings_created=0,
            duration_ms=0.0,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert result.exit_code == 0
        engine.pipeline.index_project.assert_called_once()
        call_args = engine.pipeline.index_project.call_args
        assert call_args[0][0] == Path(str(tmp_path))

    @patch("nemesis.core.cli.NemesisEngine")
    def test_index_with_languages(self, mock_engine_cls: MagicMock, tmp_path: Path):
        """index command passes languages to config."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        engine.pipeline.index_project.return_value = IndexResult(
            files_indexed=0,
            nodes_created=0,
            edges_created=0,
            chunks_created=0,
            embeddings_created=0,
            duration_ms=0.0,
        )

        runner = CliRunner()
        result = runner.invoke(
            main, ["index", str(tmp_path), "-l", "python", "-l", "typescript"]
        )

        assert result.exit_code == 0
        # NemesisConfig is created with languages — verify via the engine constructor
        config_arg = mock_engine_cls.call_args[0][0]
        assert "python" in config_arg.languages
        assert "typescript" in config_arg.languages

    @patch("nemesis.core.cli.NemesisEngine")
    def test_index_shows_results(self, mock_engine_cls: MagicMock, tmp_path: Path):
        """index command prints file count and statistics."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        engine.pipeline.index_project.return_value = IndexResult(
            files_indexed=5,
            nodes_created=42,
            edges_created=17,
            chunks_created=30,
            embeddings_created=30,
            duration_ms=123.456,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert result.exit_code == 0
        assert "Indexed 5 files" in result.output
        assert "Nodes: 42" in result.output
        assert "Edges: 17" in result.output
        assert "Chunks: 30" in result.output
        assert "Duration: 123ms" in result.output


# ---------------------------------------------------------------------------
# TestQueryCommand
# ---------------------------------------------------------------------------


class TestQueryCommand:
    """Tests for the 'query' command with real logic."""

    @patch("nemesis.core.cli.NemesisEngine")
    def test_query_shows_results(self, mock_engine_cls: MagicMock, tmp_path: Path):
        """query command shows formatted search results with scores."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        engine.embedder.embed_single.return_value = [0.1, 0.2, 0.3]
        engine.vector_store.search.return_value = [
            SearchResult(
                id="chunk-1",
                text="def hello(): pass",
                score=0.9512,
                metadata={
                    "file": "main.py",
                    "start_line": 10,
                    "end_line": 15,
                },
            ),
        ]

        runner = CliRunner()
        result = runner.invoke(
            main, ["query", "find hello function", "-p", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "Score: 0.9512" in result.output
        assert "main.py" in result.output
        assert "Lines: 10-15" in result.output
        assert "def hello(): pass" in result.output

    @patch("nemesis.core.cli.NemesisEngine")
    def test_query_no_results(self, mock_engine_cls: MagicMock, tmp_path: Path):
        """query command shows 'No results found.' for empty search."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        engine.embedder.embed_single.return_value = [0.1, 0.2, 0.3]
        engine.vector_store.search.return_value = []

        runner = CliRunner()
        result = runner.invoke(
            main, ["query", "something obscure", "-p", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "No results found." in result.output


# ---------------------------------------------------------------------------
# TestWatchCommand
# ---------------------------------------------------------------------------


class TestWatchCommand:
    """Tests for the 'watch' command."""

    @patch("nemesis.core.cli.NemesisEngine")
    @patch("nemesis.core.watcher.FileWatcher")
    @patch("nemesis.indexer.delta._get_extensions", return_value={".py"})
    @patch("time.sleep", side_effect=KeyboardInterrupt)
    def test_watch_displays_info(
        self,
        _mock_sleep: MagicMock,
        _mock_get_ext: MagicMock,
        mock_watcher_cls: MagicMock,
        mock_engine_cls: MagicMock,
        tmp_path: Path,
    ):
        """watch command displays startup info before entering the loop."""
        engine = MagicMock()
        mock_engine_cls.return_value = engine
        watcher = MagicMock()
        mock_watcher_cls.return_value = watcher

        runner = CliRunner()
        result = runner.invoke(main, ["watch", str(tmp_path)])

        assert result.exit_code == 0
        assert "Watching" in result.output
        assert "Languages:" in result.output
        assert "Extensions:" in result.output
        assert "Press Ctrl+C to stop." in result.output
        assert "Stopping watcher..." in result.output
        watcher.start.assert_called_once()
        watcher.stop.assert_called_once()
        engine.close.assert_called_once()


# ---------------------------------------------------------------------------
# TestServeCommand
# ---------------------------------------------------------------------------


class TestServeCommand:
    """Tests for the 'serve' command."""

    def test_serve_default(self):
        """serve command output mentions MCP server and stdio transport."""
        runner = CliRunner()
        result = runner.invoke(main, ["serve"])

        assert result.exit_code == 0
        assert "MCP server" in result.output
        assert "stdio" in result.output

    def test_serve_placeholder_message(self):
        """serve command mentions that MCP server is not yet implemented."""
        runner = CliRunner()
        result = runner.invoke(main, ["serve"])

        assert result.exit_code == 0
        assert "not yet implemented" in result.output
