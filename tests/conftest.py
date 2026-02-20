"""Shared test fixtures for Nemesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with .nemesis subdirectory."""
    project_dir = tmp_path / "test-project"
    project_dir.mkdir()
    nemesis_dir = project_dir / ".nemesis"
    nemesis_dir.mkdir()
    return project_dir


@pytest.fixture
def nemesis_config(tmp_project: Path):
    """Create a NemesisConfig pointing to the tmp_project."""
    from nemesis.core.config import NemesisConfig

    return NemesisConfig(
        project_name="test-project",
        project_root=tmp_project,
        openai_api_key="sk-test-fake-key",
    )


@pytest.fixture
def sample_python_file(tmp_project: Path) -> Path:
    """Create a sample Python file for parser testing."""
    code = '''"""A sample module for testing."""

from typing import List


class Calculator:
    """A simple calculator class."""

    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return round(a + b, self.precision)

    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return round(a - b, self.precision)


def create_calculator(precision: int = 2) -> Calculator:
    """Factory function for Calculator."""
    return Calculator(precision=precision)


async def fetch_data(url: str) -> List[dict]:
    """Async function for testing async detection."""
    return []
'''
    file_path = tmp_project / "sample.py"
    file_path.write_text(code)
    return file_path
