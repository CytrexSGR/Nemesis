"""Integration test for project setup — verifies everything works together."""

from pathlib import Path

import pytest


def test_nemesis_version_consistency():
    """Version in __init__.py matches pyproject.toml."""
    import tomllib

    from nemesis import __version__

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    assert __version__ == pyproject["project"]["version"]


def test_all_subpackages_importable():
    """Every subpackage defined in the project can be imported."""
    subpackages = [
        "nemesis",
        "nemesis.core",
        "nemesis.core.config",
        "nemesis.core.cli",
        "nemesis.indexer",
        "nemesis.parser",
        "nemesis.graph",
        "nemesis.vector",
        "nemesis.memory",
        "nemesis.tools",
    ]
    for pkg in subpackages:
        __import__(pkg)


def test_config_creates_with_defaults():
    """NemesisConfig can be instantiated with all defaults."""
    from nemesis.core.config import NemesisConfig

    config = NemesisConfig()
    assert config.project_name == "nemesis"
    assert config.graph_backend == "kuzu"
    assert config.vector_provider == "openai"


def test_cli_entrypoint():
    """The nemesis CLI entry point is callable."""
    from click.testing import CliRunner

    from nemesis.core.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_pyproject_has_all_dependencies():
    """pyproject.toml lists all required runtime dependencies."""
    import tomllib

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    deps = pyproject["project"]["dependencies"]
    dep_names = [d.split(">=")[0].split("[")[0].strip() for d in deps]

    required = ["kuzu", "lancedb", "openai", "click", "pydantic", "pydantic-settings", "watchdog", "mcp"]
    for req in required:
        assert req in dep_names, f"Missing dependency: {req}"


def test_pyproject_dev_dependencies():
    """pyproject.toml lists all required dev dependencies."""
    import tomllib

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]
    dep_names = [d.split(">=")[0].strip() for d in dev_deps]

    required = ["pytest", "pytest-asyncio", "pytest-cov", "ruff", "maturin"]
    for req in required:
        assert req in dep_names, f"Missing dev dependency: {req}"


@pytest.mark.xfail(reason="Cargo.toml wird in Paket A3 erstellt")
def test_rust_crate_cargo_toml():
    """nemesis-parse/Cargo.toml exists and has correct package name."""
    import tomllib

    cargo_path = Path(__file__).parent.parent / "nemesis-parse" / "Cargo.toml"
    assert cargo_path.exists(), "nemesis-parse/Cargo.toml not found"

    with open(cargo_path, "rb") as f:
        cargo = tomllib.load(f)

    assert cargo["package"]["name"] == "nemesis-parse"
    assert "pyo3" in str(cargo["dependencies"])
    assert "tree-sitter" in str(cargo["dependencies"])


def test_gitignore_exists():
    """.gitignore exists and covers Python + Rust."""
    gitignore_path = Path(__file__).parent.parent / ".gitignore"
    assert gitignore_path.exists()

    content = gitignore_path.read_text()
    assert "__pycache__" in content
    assert "target/" in content
    assert ".venv" in content or "venv/" in content
    assert ".env" in content
