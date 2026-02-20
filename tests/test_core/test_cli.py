"""Tests for Nemesis CLI skeleton and command logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from nemesis.core.cli import main
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


def test_cli_projects_command_exists():
    """The 'projects' command is registered."""
    runner = CliRunner()
    result = runner.invoke(main, ["projects", "--help"])
    assert result.exit_code == 0


def test_cli_remove_command_exists():
    """The 'remove' command is registered."""
    runner = CliRunner()
    result = runner.invoke(main, ["remove", "--help"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Helper: mock NemesisEngine as context manager
# ---------------------------------------------------------------------------


def _make_mock_engine() -> MagicMock:
    """Create a mock NemesisEngine that works as a context manager."""
    engine = MagicMock()
    engine.__enter__ = MagicMock(return_value=engine)
    engine.__exit__ = MagicMock(return_value=False)
    # Provide a registry mock with default resolve
    engine.registry = MagicMock()
    engine.registry.resolve.return_value = None
    return engine


# ---------------------------------------------------------------------------
# TestIndexCommand
# ---------------------------------------------------------------------------


class TestIndexCommand:
    """Tests for the 'index' command with real logic."""

    @patch("nemesis.tools.tools.index_project")
    @patch("nemesis.core.cli.NemesisEngine")
    def test_index_calls_tool_funcs(
        self, mock_engine_cls: MagicMock, mock_index_project: MagicMock, tmp_path: Path
    ):
        """index command calls tool_funcs.index_project."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        mock_index_project.return_value = {
            "project": tmp_path.name,
            "files_indexed": 0,
            "nodes_created": 0,
            "edges_created": 0,
            "chunks_created": 0,
            "embeddings_created": 0,
            "duration_ms": 0.0,
            "errors": [],
            "success": True,
        }

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert result.exit_code == 0
        mock_index_project.assert_called_once()
        call_args = mock_index_project.call_args
        assert call_args[0][0] is engine  # first positional arg is engine
        assert call_args[0][1] == str(tmp_path)  # second positional arg is path

    @patch("nemesis.tools.tools.index_project")
    @patch("nemesis.core.cli.NemesisEngine")
    def test_index_with_languages(
        self, mock_engine_cls: MagicMock, mock_index_project: MagicMock, tmp_path: Path
    ):
        """index command passes languages to config and tool function."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        mock_index_project.return_value = {
            "project": tmp_path.name,
            "files_indexed": 0,
            "nodes_created": 0,
            "edges_created": 0,
            "chunks_created": 0,
            "embeddings_created": 0,
            "duration_ms": 0.0,
            "errors": [],
            "success": True,
        }

        runner = CliRunner()
        result = runner.invoke(
            main, ["index", str(tmp_path), "-l", "python", "-l", "typescript"]
        )

        assert result.exit_code == 0
        # NemesisConfig is created with languages -- verify via the engine constructor
        config_arg = mock_engine_cls.call_args[0][0]
        assert "python" in config_arg.languages
        assert "typescript" in config_arg.languages

        # tool_funcs.index_project should receive the languages
        call_kwargs = mock_index_project.call_args[1]
        assert "python" in call_kwargs["languages"]
        assert "typescript" in call_kwargs["languages"]

    @patch("nemesis.tools.tools.index_project")
    @patch("nemesis.core.cli.NemesisEngine")
    def test_index_shows_results(
        self, mock_engine_cls: MagicMock, mock_index_project: MagicMock, tmp_path: Path
    ):
        """index command prints file count and statistics."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        mock_index_project.return_value = {
            "project": "myproject",
            "files_indexed": 5,
            "nodes_created": 42,
            "edges_created": 17,
            "chunks_created": 30,
            "embeddings_created": 30,
            "duration_ms": 123.456,
            "errors": [],
            "success": True,
        }

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path)])

        assert result.exit_code == 0
        assert "Project: myproject" in result.output
        assert "Indexed 5 files" in result.output
        assert "Nodes: 42" in result.output
        assert "Edges: 17" in result.output
        assert "Chunks: 30" in result.output
        assert "Duration: 123ms" in result.output

    @patch("nemesis.tools.tools.index_project")
    @patch("nemesis.core.cli.NemesisEngine")
    def test_index_with_custom_name(
        self, mock_engine_cls: MagicMock, mock_index_project: MagicMock, tmp_path: Path
    ):
        """index command passes --name to tool_funcs.index_project."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        mock_index_project.return_value = {
            "project": "custom-name",
            "files_indexed": 0,
            "nodes_created": 0,
            "edges_created": 0,
            "chunks_created": 0,
            "embeddings_created": 0,
            "duration_ms": 0.0,
            "errors": [],
            "success": True,
        }

        runner = CliRunner()
        result = runner.invoke(main, ["index", str(tmp_path), "-n", "custom-name"])

        assert result.exit_code == 0
        call_kwargs = mock_index_project.call_args[1]
        assert call_kwargs["name"] == "custom-name"


# ---------------------------------------------------------------------------
# TestQueryCommand
# ---------------------------------------------------------------------------


class TestQueryCommand:
    """Tests for the 'query' command with real logic."""

    @patch("nemesis.core.cli.NemesisEngine")
    def test_query_shows_results(self, mock_engine_cls: MagicMock):
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
        result = runner.invoke(main, ["query", "find hello function"])

        assert result.exit_code == 0
        assert "Score: 0.9512" in result.output
        assert "main.py" in result.output
        assert "Lines: 10-15" in result.output
        assert "def hello(): pass" in result.output

    @patch("nemesis.core.cli.NemesisEngine")
    def test_query_no_results(self, mock_engine_cls: MagicMock):
        """query command shows 'No results found.' for empty search."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        engine.embedder.embed_single.return_value = [0.1, 0.2, 0.3]
        engine.vector_store.search.return_value = []

        runner = CliRunner()
        result = runner.invoke(main, ["query", "something obscure"])

        assert result.exit_code == 0
        assert "No results found." in result.output

    @patch("nemesis.core.cli.NemesisEngine")
    def test_query_with_project_filter(self, mock_engine_cls: MagicMock):
        """query command passes --project to vector store search."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        engine.embedder.embed_single.return_value = [0.1, 0.2, 0.3]
        engine.vector_store.search.return_value = []

        runner = CliRunner()
        result = runner.invoke(main, ["query", "test query", "-p", "myproject"])

        assert result.exit_code == 0
        # Verify project_id was passed to search
        engine.vector_store.search.assert_called_once()
        call_kwargs = engine.vector_store.search.call_args[1]
        assert call_kwargs["project_id"] == "myproject"


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

    @patch("nemesis.core.cli.NemesisEngine")
    @patch("nemesis.core.watcher.FileWatcher")
    @patch("nemesis.indexer.delta._get_extensions", return_value={".py"})
    @patch("time.sleep", side_effect=KeyboardInterrupt)
    def test_watch_with_name(
        self,
        _mock_sleep: MagicMock,
        _mock_get_ext: MagicMock,
        mock_watcher_cls: MagicMock,
        mock_engine_cls: MagicMock,
        tmp_path: Path,
    ):
        """watch command uses --name as project_id."""
        engine = MagicMock()
        mock_engine_cls.return_value = engine
        watcher = MagicMock()
        mock_watcher_cls.return_value = watcher

        runner = CliRunner()
        result = runner.invoke(main, ["watch", str(tmp_path), "-n", "myproject"])

        assert result.exit_code == 0
        assert "project: myproject" in result.output


# ---------------------------------------------------------------------------
# TestProjectsCommand
# ---------------------------------------------------------------------------


class TestProjectsCommand:
    """Tests for the 'projects' command."""

    @patch("nemesis.tools.tools.list_projects")
    @patch("nemesis.core.cli.NemesisEngine")
    def test_projects_lists_registered(
        self, mock_engine_cls: MagicMock, mock_list_projects: MagicMock
    ):
        """projects command lists registered projects with details."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        mock_list_projects.return_value = {
            "count": 1,
            "projects": {
                "eve": {
                    "path": "/home/user/eve",
                    "languages": ["python"],
                    "files": 100,
                    "indexed_at": "2026-01-15T10:00:00Z",
                },
            },
        }

        runner = CliRunner()
        result = runner.invoke(main, ["projects"])

        assert result.exit_code == 0
        assert "eve" in result.output
        assert "/home/user/eve" in result.output
        assert "python" in result.output
        assert "100" in result.output
        assert "2026-01-15" in result.output

    @patch("nemesis.tools.tools.list_projects")
    @patch("nemesis.core.cli.NemesisEngine")
    def test_projects_empty(
        self, mock_engine_cls: MagicMock, mock_list_projects: MagicMock
    ):
        """projects command shows message when no projects registered."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        mock_list_projects.return_value = {"count": 0, "projects": {}}

        runner = CliRunner()
        result = runner.invoke(main, ["projects"])

        assert result.exit_code == 0
        assert "No projects registered" in result.output


# ---------------------------------------------------------------------------
# TestRemoveCommand
# ---------------------------------------------------------------------------


class TestRemoveCommand:
    """Tests for the 'remove' command."""

    @patch("nemesis.tools.tools.remove_project")
    @patch("nemesis.core.cli.NemesisEngine")
    def test_remove_succeeds(
        self, mock_engine_cls: MagicMock, mock_remove_project: MagicMock
    ):
        """remove command removes a project successfully."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        mock_remove_project.return_value = {"project": "eve", "success": True}

        runner = CliRunner()
        result = runner.invoke(main, ["remove", "eve"])

        assert result.exit_code == 0
        assert "Removed" in result.output
        assert "eve" in result.output

    @patch("nemesis.tools.tools.remove_project")
    @patch("nemesis.core.cli.NemesisEngine")
    def test_remove_not_found(
        self, mock_engine_cls: MagicMock, mock_remove_project: MagicMock
    ):
        """remove command fails when project not found."""
        engine = _make_mock_engine()
        mock_engine_cls.return_value = engine
        mock_remove_project.return_value = {
            "error": "Project 'nonexistent' not found",
            "success": False,
        }

        runner = CliRunner()
        result = runner.invoke(main, ["remove", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# TestServeCommand
# ---------------------------------------------------------------------------


class TestServeCommand:
    """Tests for the 'serve' command."""

    def test_serve_default(self):
        """serve command output mentions MCP server and stdio transport."""
        runner = CliRunner()
        with patch("nemesis.core.server.run_stdio_server", return_value=None):
            with patch("asyncio.run", return_value=None):
                result = runner.invoke(main, ["serve"])

        assert result.exit_code == 0
        assert "MCP server" in result.output
        assert "stdio" in result.output

    def test_serve_calls_run_stdio_server(self):
        """serve command calls asyncio.run with run_stdio_server."""
        runner = CliRunner()
        with patch("asyncio.run") as mock_run:
            result = runner.invoke(main, ["serve"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
