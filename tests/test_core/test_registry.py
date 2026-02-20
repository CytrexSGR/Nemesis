"""Tests for ProjectRegistry."""

import json
from pathlib import Path

import pytest

from nemesis.core.registry import ProjectInfo, ProjectRegistry


@pytest.fixture
def registry(tmp_path):
    return ProjectRegistry(tmp_path / "registry.json")


class TestProjectRegistry:
    def test_register_creates_entry(self, registry):
        info = registry.register(
            path=Path("/home/user/projects/eve"),
            languages=["python", "typescript"],
        )
        assert info.name == "eve"
        assert info.path == Path("/home/user/projects/eve")
        assert info.languages == ["python", "typescript"]

    def test_register_custom_name(self, registry):
        info = registry.register(
            path=Path("/home/user/projects/Eve-Online-Copilot"),
            name="eve",
            languages=["python"],
        )
        assert info.name == "eve"

    def test_register_default_name_from_dirname(self, registry):
        info = registry.register(
            path=Path("/home/user/projects/my-cool-project"),
            languages=["python"],
        )
        assert info.name == "my-cool-project"

    def test_register_persists_to_file(self, registry, tmp_path):
        registry.register(path=Path("/tmp/proj"), languages=["python"])
        data = json.loads((tmp_path / "registry.json").read_text())
        assert "proj" in data["projects"]

    def test_unregister_removes_entry(self, registry):
        registry.register(path=Path("/tmp/proj"), languages=["python"])
        registry.unregister("proj")
        assert registry.get("proj") is None

    def test_unregister_unknown_raises(self, registry):
        with pytest.raises(KeyError):
            registry.unregister("nope")

    def test_list_projects(self, registry):
        registry.register(path=Path("/tmp/a"), languages=["python"])
        registry.register(path=Path("/tmp/b"), languages=["rust"])
        projects = registry.list_projects()
        assert len(projects) == 2
        assert "a" in projects
        assert "b" in projects

    def test_resolve_finds_project_by_subpath(self, registry):
        registry.register(path=Path("/home/user/projects/eve"), languages=["python"])
        result = registry.resolve(Path("/home/user/projects/eve/services/main.py"))
        assert result == "eve"

    def test_resolve_returns_none_for_unknown(self, registry):
        assert registry.resolve(Path("/unknown/path/file.py")) is None

    def test_get_returns_project_info(self, registry):
        registry.register(path=Path("/tmp/proj"), languages=["python"])
        info = registry.get("proj")
        assert info is not None
        assert info.name == "proj"

    def test_get_returns_none_for_unknown(self, registry):
        assert registry.get("nope") is None

    def test_update_indexed_at(self, registry):
        registry.register(path=Path("/tmp/proj"), languages=["python"])
        registry.update_stats("proj", files=42)
        info = registry.get("proj")
        assert info.files == 42
        assert info.indexed_at is not None

    def test_registry_loads_from_existing_file(self, tmp_path):
        data = {
            "projects": {
                "old": {
                    "path": "/tmp/old",
                    "languages": ["python"],
                    "indexed_at": None,
                    "files": 0,
                }
            }
        }
        (tmp_path / "registry.json").write_text(json.dumps(data))
        reg = ProjectRegistry(tmp_path / "registry.json")
        assert reg.get("old") is not None
