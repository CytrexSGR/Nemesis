# 05e — Full Project Index + Delta Update

> **Arbeitspaket E5** — Teil 5 von 5 des Indexing Pipeline Plans

**Goal:** Full Project Index (alle Code-Dateien rekursiv finden und indexieren) sowie Delta Project Update (nur geaenderte Dateien verarbeiten) implementieren.

**Tech Stack:** Python, nemesis.indexer.pipeline (IndexingPipeline), nemesis.indexer.delta (detect_changes, delete_file_data)

**Abhängigkeiten:** E1 (05a), E2 (05b), E3 (05c), E4 (05d)

**Tasks in diesem Paket:** 9, 10

---

## Task 9: Pipeline — Full Project Index

**Files:**
- `nemesis/indexer/pipeline.py` (erweitern)
- `tests/test_indexer/test_pipeline.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_indexer/test_pipeline.py — APPEND folgende Tests

def test_index_project_indexes_all_files(tmp_path):
    """index_project indexiert alle Code-Dateien im Verzeichnis."""
    from nemesis.indexer.pipeline import IndexingPipeline

    # Erstelle Test-Dateien
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def main():\n    pass\n")
    (src / "utils.py").write_text("def helper():\n    return 1\n")
    (tmp_path / "readme.md").write_text("# Readme\n")  # Soll ignoriert werden

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_project(tmp_path, languages=["python"])

    # Parser wurde fuer beide .py Dateien aufgerufen
    assert parser.parse_file.call_count == 2
    assert result.files_indexed == 2
    assert result.success is True


def test_index_project_skips_ignored_dirs(tmp_path):
    """index_project ignoriert __pycache__, node_modules, etc."""
    from nemesis.indexer.pipeline import IndexingPipeline

    (tmp_path / "good.py").write_text("x = 1\n")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "bad.py").write_text("y = 2\n")

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_project(tmp_path, languages=["python"])

    # Nur good.py, nicht __pycache__/bad.py
    assert parser.parse_file.call_count == 1
    assert result.files_indexed == 1


def test_index_project_multiple_languages(tmp_path):
    """index_project unterstuetzt mehrere Sprachen gleichzeitig."""
    from nemesis.indexer.pipeline import IndexingPipeline

    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / "index.ts").write_text("const x = 1;\n")

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_project(tmp_path, languages=["python", "typescript"])

    assert parser.parse_file.call_count == 2
    assert result.files_indexed == 2


def test_index_project_empty_dir(tmp_path):
    """index_project mit leerem Verzeichnis gibt leeres Ergebnis."""
    from nemesis.indexer.pipeline import IndexingPipeline

    parser = MagicMock()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_project(tmp_path, languages=["python"])

    assert result.files_indexed == 0
    assert result.success is True
    parser.parse_file.assert_not_called()


def test_index_project_continues_on_single_file_error(tmp_path):
    """index_project macht weiter wenn eine einzelne Datei fehlschlaegt."""
    from nemesis.indexer.pipeline import IndexingPipeline

    (tmp_path / "good.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_text("invalid syntax ???\n")

    parser = MagicMock()
    call_count = 0

    def mock_parse(path):
        nonlocal call_count
        call_count += 1
        if "bad.py" in str(path):
            raise RuntimeError("Parse error")
        node = MockCodeNode(
            id="func-1", node_type="Function", name="x",
            start_line=1, end_line=1, source="x = 1\n",
        )
        return MockParseResult(
            nodes=[node], edges=[], file_path=str(path), language="python",
        )

    parser.parse_file.side_effect = mock_parse
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.index_project(tmp_path, languages=["python"])

    # Eine Datei erfolgreich, eine fehlgeschlagen
    assert result.files_indexed == 1
    assert len(result.errors) == 1
    assert result.success is False


def test_index_project_progress_reporting(tmp_path):
    """index_project meldet Fortschritt ueber Callback."""
    from nemesis.indexer.pipeline import IndexingPipeline

    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    progress_calls = []

    def on_progress(msg, current, total):
        progress_calls.append((msg, current, total))

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
        on_progress=on_progress,
    )

    pipeline.index_project(tmp_path, languages=["python"])

    # Es wurden Fortschritts-Meldungen erzeugt
    assert len(progress_calls) > 0
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v -k "index_project"
# Erwartung: FAILED — index_project existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/pipeline.py — Methode index_project zur Klasse IndexingPipeline hinzufuegen

    def index_project(
        self,
        path: Path,
        languages: list[str],
        ignore_dirs: set[str] | None = None,
    ) -> IndexResult:
        """Full Index eines gesamten Projekts.

        Findet alle Code-Dateien im Verzeichnis (rekursiv) und
        indexiert jede einzeln. Fehler bei einzelnen Dateien werden
        gesammelt, der Prozess laeuft weiter.

        Args:
            path: Wurzelverzeichnis des Projekts.
            languages: Liste der zu indexierenden Sprachen
                       (z.B. ["python", "typescript"]).
            ignore_dirs: Verzeichnisnamen die ignoriert werden.
                         Default: __pycache__, node_modules, .git, etc.

        Returns:
            Aggregiertes IndexResult ueber alle Dateien.
        """
        from nemesis.indexer.delta import _get_extensions, _collect_code_files

        start_time = time.monotonic()

        extensions = _get_extensions(languages)
        files = _collect_code_files(path, extensions, ignore_dirs)

        total_files = len(files)
        total_nodes = 0
        total_edges = 0
        total_chunks = 0
        total_embeddings = 0
        files_indexed = 0
        all_errors: list[str] = []

        self._report_progress(f"Found {total_files} files to index", 0, total_files)

        for i, file_path in enumerate(files):
            self._report_progress(
                f"Indexing {file_path.name} ({i + 1}/{total_files})",
                i + 1,
                total_files,
            )

            result = self.index_file(file_path)

            files_indexed += result.files_indexed
            total_nodes += result.nodes_created
            total_edges += result.edges_created
            total_chunks += result.chunks_created
            total_embeddings += result.embeddings_created
            all_errors.extend(result.errors)

        duration_ms = (time.monotonic() - start_time) * 1000

        self._report_progress(
            f"Done: {files_indexed}/{total_files} files indexed",
            total_files,
            total_files,
        )

        return IndexResult(
            files_indexed=files_indexed,
            nodes_created=total_nodes,
            edges_created=total_edges,
            chunks_created=total_chunks,
            embeddings_created=total_embeddings,
            duration_ms=duration_ms,
            errors=all_errors,
        )
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/pipeline.py tests/test_indexer/test_pipeline.py
git commit -m "feat(indexer): add full project indexing with progress reporting"
```

---

## Task 10: Pipeline — Delta Project Update

**Files:**
- `nemesis/indexer/pipeline.py` (erweitern)
- `tests/test_indexer/test_pipeline.py` (erweitern)

### Step 1 — Write Test

```python
# tests/test_indexer/test_pipeline.py — APPEND folgende Tests

def test_update_project_processes_all_change_types(tmp_path):
    """update_project verarbeitet ADDED, MODIFIED und DELETED korrekt."""
    from nemesis.indexer.pipeline import IndexingPipeline
    from nemesis.indexer.models import ChangeType, FileChange
    from nemesis.indexer.delta import compute_file_hash

    # Bestehende Datei die geaendert wurde
    modified_file = tmp_path / "modified.py"
    modified_file.write_text("x = 2  # changed\n")

    # Neue Datei
    new_file = tmp_path / "new.py"
    new_file.write_text("y = 1\n")

    parser, _ = _make_mock_parser()
    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {
        str(modified_file): "old_hash",
        str(tmp_path / "deleted.py"): "deleted_hash",
    }
    graph.get_chunk_ids_for_file.return_value = []
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.update_project(tmp_path, languages=["python"])

    # ADDED (new.py) und MODIFIED (modified.py) wurden indexiert
    # DELETED (deleted.py) wurde geloescht
    assert result.files_indexed >= 1
    # delete_nodes_for_file wurde aufgerufen (fuer modified und deleted)
    assert graph.delete_nodes_for_file.call_count >= 1


def test_update_project_no_changes(tmp_path):
    """update_project mit unveraenderten Dateien gibt leeres Ergebnis."""
    from nemesis.indexer.pipeline import IndexingPipeline
    from nemesis.indexer.delta import compute_file_hash

    py_file = tmp_path / "stable.py"
    py_file.write_text("x = 1\n")
    current_hash = compute_file_hash(py_file)

    parser = MagicMock()
    graph = _make_mock_graph()
    graph.get_file_hashes.return_value = {str(py_file): current_hash}
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.update_project(tmp_path, languages=["python"])

    # Nichts zu tun
    assert result.files_indexed == 0
    assert result.success is True
    parser.parse_file.assert_not_called()


def test_update_project_handles_deleted_files(tmp_path):
    """update_project loescht Daten fuer geloeschte Dateien."""
    from nemesis.indexer.pipeline import IndexingPipeline

    # Keine Dateien auf Disk, aber Graph kennt eine
    parser = MagicMock()
    graph = _make_mock_graph()
    deleted_path = str(tmp_path / "gone.py")
    graph.get_file_hashes.return_value = {deleted_path: "hash123"}
    graph.get_chunk_ids_for_file.return_value = ["chunk-old"]
    vector_store = _make_mock_vector_store()
    embedder = _make_mock_embedder()

    pipeline = IndexingPipeline(
        parser=parser,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = pipeline.update_project(tmp_path, languages=["python"])

    # Alte Daten wurden geloescht
    graph.delete_nodes_for_file.assert_called_once_with(deleted_path)
    vector_store.delete_embeddings.assert_called_once_with(["chunk-old"])
    # Kein Parse-Aufruf (Datei existiert nicht mehr)
    parser.parse_file.assert_not_called()
    assert result.success is True
```

### Step 2 — Run (RED)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v -k "update_project"
# Erwartung: FAILED — update_project existiert nicht
```

### Step 3 — Implement

```python
# nemesis/indexer/pipeline.py — Methode update_project zur Klasse IndexingPipeline hinzufuegen

    def update_project(
        self,
        path: Path,
        languages: list[str],
        ignore_dirs: set[str] | None = None,
    ) -> IndexResult:
        """Delta Update: Nur geaenderte Dateien verarbeiten.

        Erkennt Aenderungen seit dem letzten Index und verarbeitet nur:
        - ADDED: Neu indexieren
        - MODIFIED: Alte Daten loeschen, neu indexieren
        - DELETED: Alte Daten loeschen

        Args:
            path: Wurzelverzeichnis des Projekts.
            languages: Liste der Sprachen.
            ignore_dirs: Verzeichnisnamen die ignoriert werden.

        Returns:
            Aggregiertes IndexResult.
        """
        from nemesis.indexer.delta import detect_changes, delete_file_data
        from nemesis.indexer.models import ChangeType

        start_time = time.monotonic()

        self._report_progress("Detecting changes...")
        changes = detect_changes(path, self._graph, languages, ignore_dirs)

        if not changes:
            duration_ms = (time.monotonic() - start_time) * 1000
            self._report_progress("No changes detected")
            return IndexResult(
                files_indexed=0,
                nodes_created=0,
                edges_created=0,
                chunks_created=0,
                embeddings_created=0,
                duration_ms=duration_ms,
                errors=[],
            )

        total_changes = len(changes)
        files_indexed = 0
        total_nodes = 0
        total_edges = 0
        total_chunks = 0
        total_embeddings = 0
        all_errors: list[str] = []

        added = [c for c in changes if c.change_type == ChangeType.ADDED]
        modified = [c for c in changes if c.change_type == ChangeType.MODIFIED]
        deleted = [c for c in changes if c.change_type == ChangeType.DELETED]

        self._report_progress(
            f"Changes: {len(added)} added, {len(modified)} modified, {len(deleted)} deleted",
            0,
            total_changes,
        )

        # Geloeschte Dateien: Nur alte Daten entfernen
        for i, change in enumerate(deleted):
            self._report_progress(
                f"Removing {change.path.name}",
                i + 1,
                total_changes,
            )
            try:
                delete_file_data(change.path, self._graph, self._vector_store)
            except Exception as e:
                all_errors.append(f"Error deleting data for {change.path}: {e}")

        # Geaenderte Dateien: Alte Daten loeschen, dann neu indexieren
        for i, change in enumerate(modified):
            self._report_progress(
                f"Re-indexing {change.path.name}",
                len(deleted) + i + 1,
                total_changes,
            )
            result = self.reindex_file(change.path)
            files_indexed += result.files_indexed
            total_nodes += result.nodes_created
            total_edges += result.edges_created
            total_chunks += result.chunks_created
            total_embeddings += result.embeddings_created
            all_errors.extend(result.errors)

        # Neue Dateien: Einfach indexieren
        for i, change in enumerate(added):
            self._report_progress(
                f"Indexing new {change.path.name}",
                len(deleted) + len(modified) + i + 1,
                total_changes,
            )
            result = self.index_file(change.path)
            files_indexed += result.files_indexed
            total_nodes += result.nodes_created
            total_edges += result.edges_created
            total_chunks += result.chunks_created
            total_embeddings += result.embeddings_created
            all_errors.extend(result.errors)

        duration_ms = (time.monotonic() - start_time) * 1000

        self._report_progress(
            f"Delta update done: {files_indexed} files processed",
            total_changes,
            total_changes,
        )

        return IndexResult(
            files_indexed=files_indexed,
            nodes_created=total_nodes,
            edges_created=total_edges,
            chunks_created=total_chunks,
            embeddings_created=total_embeddings,
            duration_ms=duration_ms,
            errors=all_errors,
        )
```

### Step 4 — Run (GREEN)

```bash
cd /home/andreas/projects/nemesis && python -m pytest tests/test_indexer/test_pipeline.py -v
# Erwartung: ALL PASSED
```

### Step 5 — Commit

```bash
git add nemesis/indexer/pipeline.py tests/test_indexer/test_pipeline.py
git commit -m "feat(indexer): add delta project update with change detection"
```

---

## Zusammenfassung E5

| Task | Datei(en) | Beschreibung |
|------|-----------|-------------|
| 9 | `pipeline.py` | index_project: Full Project Index mit Progress |
| 10 | `pipeline.py` | update_project: Delta Project Update |

---

**Navigation:**
- Zurueck: [05d — Single File Pipeline (E4)](05d-single-file-pipeline.md)
- Gesamtplan: [05 — Indexing Pipeline](05-indexing-pipeline.md)
