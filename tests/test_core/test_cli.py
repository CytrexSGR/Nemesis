"""Tests for Nemesis CLI skeleton."""

from click.testing import CliRunner


def test_cli_main_group():
    """The main CLI group exists and shows help."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Nemesis" in result.output


def test_cli_version_flag():
    """--version flag prints the version."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_index_command_exists():
    """The 'index' command is registered."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["index", "--help"])
    assert result.exit_code == 0
    assert "Index" in result.output or "index" in result.output.lower()


def test_cli_query_command_exists():
    """The 'query' command is registered."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["query", "--help"])
    assert result.exit_code == 0


def test_cli_watch_command_exists():
    """The 'watch' command is registered."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["watch", "--help"])
    assert result.exit_code == 0


def test_cli_serve_command_exists():
    """The 'serve' command is registered."""
    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
