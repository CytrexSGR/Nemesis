# Nemesis Multi-Project Design

## Entscheidung

**Ansatz B: Namespaced Single-Engine** — Eine KuzuDB, eine LanceDB, alle Projekte in einer Instanz mit Prefix-Isolation.

## Zentrales Datenverzeichnis

Statt `.nemesis/` pro Projekt ein zentrales Verzeichnis:

```
~/.nemesis/
├── graph/          # Eine KuzuDB für alle Projekte
├── vectors/        # Eine LanceDB für alle Projekte
└── registry.json   # Registrierte Projekte
```

`registry.json` Beispiel:

```json
{
  "projects": {
    "eve": {
      "path": "/home/andreas/projects/Eve-Online-Copilot",
      "languages": ["python", "typescript"],
      "indexed_at": "2026-02-20T16:55:00Z",
      "files": 1862
    },
    "nemesis": {
      "path": "/home/andreas/projects/nemesis",
      "languages": ["python"],
      "indexed_at": "2026-02-20T14:00:00Z",
      "files": 33
    }
  }
}
```

Projekt-IDs werden automatisch vom Verzeichnisnamen abgeleitet oder per `--name` überschrieben. Pfad-Resolution: ein Dateipfad wird gegen die Registry geprüft, welches Projekt ihn enthält.

## Graph-Namespacing

Node-IDs bekommen ein Projekt-Prefix mit `::` als Separator:

```
eve::func:services/market-service/app/main.py:get_prices:15
nemesis::func:nemesis/core/engine.py:initialize:42
```

- `::` kommt in Dateipfaden nicht vor
- Dateipfade werden **relativ zum Projekt-Root** gespeichert (portabel)
- Cross-Project-Query: kein Filter auf Prefix → alle Projekte
- Projekt-Filter: `WHERE id STARTS WITH 'eve::'`

Betroffene Stellen:
- `KuzuAdapter.add_node()` / `add_edge()` — Prefix beim Schreiben
- `KuzuAdapter.get_node()` / `get_neighbors()` — Prefix beim Lesen
- `KuzuAdapter.get_nodes_for_file()` — Filter nach Projekt-Prefix
- `IndexingPipeline` — generiert IDs mit Prefix
- `delete_file_data()` — löscht nur Nodes mit passendem Prefix

## VectorStore-Namespacing

LanceDB-Schema bekommt ein `project_id`-Feld:

```python
schema = pa.schema([
    pa.field("id", pa.utf8()),
    pa.field("text", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), list_size=dimensions)),
    pa.field("metadata", pa.utf8()),
    pa.field("project_id", pa.utf8()),   # NEU
])
```

- `search_code(query)` ohne Projekt → alle Projekte, merged nach Score
- `search_code(query, project="eve")` → `WHERE project_id = 'eve'` Pre-Filter
- Projekt löschen: `DELETE FROM chunks WHERE project_id = 'eve'`
- Re-Index: Chunks löschen + neu indexieren, atomic durch LanceDB-Transaktionen

## Config & Engine

**NemesisConfig:**

```python
class NemesisConfig(BaseSettings):
    data_dir: Path = Path.home() / ".nemesis"
    # project_root, graph_path, vector_path entfallen
    # graph:   data_dir / "graph"
    # vectors: data_dir / "vectors"
```

**ProjectRegistry:**

```python
class ProjectRegistry:
    def register(self, path: Path, name: str = None, languages: list[str]) -> str
    def unregister(self, name: str) -> None
    def resolve(self, file_path: Path) -> str | None
    def list_projects(self) -> dict[str, ProjectInfo]
    def get(self, name: str) -> ProjectInfo | None
```

**NemesisEngine:**

```python
class NemesisEngine:
    def initialize(self):
        self._graph = KuzuAdapter(db_path=str(self.config.data_dir / "graph"))
        self._vector_store = VectorStore(path=str(self.config.data_dir / "vectors"))
        self._registry = ProjectRegistry(self.config.data_dir / "registry.json")
```

**Pipeline** bekommt `project_id` als Parameter:

```python
def index_project(self, path: Path, project_id: str, languages: list[str]) -> IndexResult
def index_file(self, path: Path, project_id: str, project_root: Path) -> IndexResult
```

## MCP Tools

Tools bekommen optionalen `project`-Parameter:

```python
search_code(query: str, project: str = None, limit: int = 10)
get_context(file_path: str, depth: int = 2)  # Auto-Resolve über Registry
index_project(path: str, name: str = None, languages: list[str] = None)
update_project(path: str)
list_projects()                              # NEU
remove_project(name: str)                    # NEU
```

## CLI

```bash
nemesis index /path/to/project -l python -l typescript
nemesis index . --name eve
nemesis query "ESI client" --project eve
nemesis query "ESI client"                   # alle Projekte
nemesis projects                             # Liste
nemesis remove eve                           # Projekt entfernen
```

## Server-Config

Kein `NEMESIS_PROJECT_ROOT` mehr nötig:

```json
{
  "nemesis": {
    "command": "nemesis",
    "args": ["serve"],
    "env": {
      "NEMESIS_OPENAI_API_KEY": "sk-..."
    }
  }
}
```

## Migration

Neu indexieren statt migrieren:

1. `~/.nemesis/` erstellen mit neuem Schema
2. `nemesis index /home/andreas/projects/nemesis --name nemesis`
3. `nemesis index /home/andreas/projects/Eve-Online-Copilot --name eve`
4. Alte `.nemesis/`-Verzeichnisse in den Projekten löschen

Kein Migrations-Code, keine Abwärtskompatibilität.
