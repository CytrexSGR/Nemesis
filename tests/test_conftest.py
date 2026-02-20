"""Tests for test infrastructure — verify fixtures work."""


def test_tmp_project_fixture(tmp_project):
    """tmp_project fixture creates a temporary project directory."""
    assert tmp_project.exists()
    assert tmp_project.is_dir()


def test_tmp_project_has_nemesis_dir(tmp_project):
    """tmp_project fixture creates a .nemesis subdirectory."""
    nemesis_dir = tmp_project / ".nemesis"
    assert nemesis_dir.exists()
    assert nemesis_dir.is_dir()


def test_nemesis_config_fixture(nemesis_config):
    """nemesis_config fixture returns a valid NemesisConfig."""
    from nemesis.core.config import NemesisConfig

    assert isinstance(nemesis_config, NemesisConfig)
    assert nemesis_config.project_root.exists()


def test_sample_python_file_fixture(sample_python_file):
    """sample_python_file fixture creates a Python file for testing."""
    assert sample_python_file.exists()
    assert sample_python_file.suffix == ".py"
    content = sample_python_file.read_text()
    assert "class Calculator" in content
    assert "def add" in content
