"""Project Registry — manages registered projects in ~/.nemesis/registry.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ProjectInfo:
    """Info about a registered project."""

    name: str
    path: Path
    languages: list[str] = field(default_factory=list)
    indexed_at: str | None = None
    files: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        return d

    @classmethod
    def from_dict(cls, name: str, data: dict) -> ProjectInfo:
        return cls(
            name=name,
            path=Path(data["path"]),
            languages=data.get("languages", []),
            indexed_at=data.get("indexed_at"),
            files=data.get("files", 0),
        )


class ProjectRegistry:
    """Manages registered projects persisted to a JSON file."""

    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path
        self._projects: dict[str, ProjectInfo] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            for name, proj_data in data.get("projects", {}).items():
                self._projects[name] = ProjectInfo.from_dict(name, proj_data)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"projects": {name: info.to_dict() for name, info in self._projects.items()}}
        self._path.write_text(json.dumps(data, indent=2))

    def register(
        self,
        path: Path,
        languages: list[str],
        name: str | None = None,
    ) -> ProjectInfo:
        project_name = name or path.name
        if "::" in project_name:
            raise ValueError(f"Project name cannot contain '::': {project_name}")
        info = ProjectInfo(name=project_name, path=path.resolve(), languages=languages)
        self._projects[project_name] = info
        self._save()
        return info

    def unregister(self, name: str) -> None:
        if name not in self._projects:
            raise KeyError(f"Project '{name}' not found in registry")
        del self._projects[name]
        self._save()

    def list_projects(self) -> dict[str, ProjectInfo]:
        return dict(self._projects)

    def get(self, name: str) -> ProjectInfo | None:
        return self._projects.get(name)

    def resolve(self, file_path: Path) -> str | None:
        resolved = file_path.resolve()
        for name, info in self._projects.items():
            try:
                resolved.relative_to(info.path)
                return name
            except ValueError:
                continue
        return None

    def update_stats(self, name: str, files: int) -> None:
        info = self._projects.get(name)
        if info is None:
            raise KeyError(f"Project '{name}' not found")
        info.files = files
        info.indexed_at = datetime.now(timezone.utc).isoformat()
        self._save()
