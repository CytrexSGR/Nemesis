"""Tests fuer nemesis.indexer.delta — Delta Detection + Deletion."""

from pathlib import Path
from unittest.mock import MagicMock

from nemesis.indexer.delta import (
    compute_file_hash,
    delete_file_data,
    detect_changes,
)
from nemesis.indexer.models import ChangeType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_graph():
    graph = MagicMock()
    return graph


# ---------------------------------------------------------------------------
# Task 5: Delta Detection
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_compute_file_hash(self, tmp_path: Path):
        """SHA-256 Hash ist ein 64-Zeichen Hex-String."""
        f = tmp_path / "hello.py"
        f.write_text("print('hello')\n")
        result = compute_file_hash(f)
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_compute_file_hash_deterministic(self, tmp_path: Path):
        """Gleicher Inhalt ergibt immer den gleichen Hash."""
        content = "def foo(): return 42\n"
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text(content)
        f2.write_text(content)
        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_compute_file_hash_different_content(self, tmp_path: Path):
        """Unterschiedlicher Inhalt ergibt unterschiedliche Hashes."""
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("x = 1\n")
        f2.write_text("x = 2\n")
        assert compute_file_hash(f1) != compute_file_hash(f2)


class TestDetectChanges:
    def test_detect_changes_new_file(self, tmp_path: Path):
        """Neue Datei wird als ADDED erkannt."""
        (tmp_path / "main.py").write_text("x = 1\n")
        graph = _make_mock_graph()
        graph.get_file_hashes.return_value = {}

        changes = detect_changes(tmp_path, graph, ["python"])

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.ADDED
        assert changes[0].old_hash is None
        assert changes[0].new_hash is not None

    def test_detect_changes_modified_file(self, tmp_path: Path):
        """Geaenderte Datei wird als MODIFIED erkannt."""
        f = tmp_path / "main.py"
        f.write_text("x = 2\n")
        current_hash = compute_file_hash(f)

        graph = _make_mock_graph()
        graph.get_file_hashes.return_value = {str(f): "old_hash_value"}

        changes = detect_changes(tmp_path, graph, ["python"])

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.MODIFIED
        assert changes[0].old_hash == "old_hash_value"
        assert changes[0].new_hash == current_hash

    def test_detect_changes_deleted_file(self, tmp_path: Path):
        """Datei im Graph aber nicht auf Disk wird als DELETED erkannt."""
        graph = _make_mock_graph()
        graph.get_file_hashes.return_value = {str(tmp_path / "gone.py"): "some_hash"}

        changes = detect_changes(tmp_path, graph, ["python"])

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.DELETED
        assert changes[0].old_hash == "some_hash"
        assert changes[0].new_hash is None

    def test_detect_changes_unchanged_file(self, tmp_path: Path):
        """Unveraenderte Datei erzeugt keinen Change-Eintrag."""
        f = tmp_path / "stable.py"
        f.write_text("y = 42\n")
        file_hash = compute_file_hash(f)

        graph = _make_mock_graph()
        graph.get_file_hashes.return_value = {str(f): file_hash}

        changes = detect_changes(tmp_path, graph, ["python"])

        assert len(changes) == 0

    def test_detect_changes_ignores_non_code_files(self, tmp_path: Path):
        """Nicht-Code-Dateien (.md, .json, .png) werden ignoriert."""
        (tmp_path / "readme.md").write_text("# Readme\n")
        (tmp_path / "config.json").write_text("{}\n")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")

        graph = _make_mock_graph()
        graph.get_file_hashes.return_value = {}

        changes = detect_changes(tmp_path, graph, ["python"])

        assert len(changes) == 0

    def test_detect_changes_multiple_languages(self, tmp_path: Path):
        """Python- und TypeScript-Dateien werden beide erkannt."""
        (tmp_path / "app.py").write_text("print('hi')\n")
        (tmp_path / "index.ts").write_text("console.log('hi');\n")

        graph = _make_mock_graph()
        graph.get_file_hashes.return_value = {}

        changes = detect_changes(tmp_path, graph, ["python", "typescript"])

        assert len(changes) == 2
        paths = {str(c.path.name) for c in changes}
        assert "app.py" in paths
        assert "index.ts" in paths
        assert all(c.change_type == ChangeType.ADDED for c in changes)


# ---------------------------------------------------------------------------
# Task 6: Delete File Data
# ---------------------------------------------------------------------------


class TestDeleteFileData:
    def test_delete_file_data_calls_graph_and_vector(self):
        """Ruft delete_nodes_for_file und delete_embeddings mit chunk_ids auf."""
        graph = _make_mock_graph()
        graph.get_chunk_ids_for_file.return_value = ["chunk-1", "chunk-2"]

        vector_store = MagicMock()

        delete_file_data("src/main.py", graph, vector_store)

        graph.get_chunk_ids_for_file.assert_called_once_with("src/main.py")
        graph.delete_nodes_for_file.assert_called_once_with("src/main.py")
        vector_store.delete_embeddings.assert_called_once_with(["chunk-1", "chunk-2"])

    def test_delete_file_data_no_chunks(self):
        """Funktioniert auch wenn keine Chunks vorhanden sind."""
        graph = _make_mock_graph()
        graph.get_chunk_ids_for_file.return_value = []

        vector_store = MagicMock()

        delete_file_data("src/empty.py", graph, vector_store)

        graph.get_chunk_ids_for_file.assert_called_once_with("src/empty.py")
        graph.delete_nodes_for_file.assert_called_once_with("src/empty.py")
        vector_store.delete_embeddings.assert_not_called()

    def test_delete_file_data_no_existing_data(self):
        """Funktioniert wenn die Datei gar nicht im Graph existiert."""
        graph = _make_mock_graph()
        graph.get_chunk_ids_for_file.return_value = []

        vector_store = MagicMock()

        delete_file_data("src/unknown.py", graph, vector_store)

        graph.get_chunk_ids_for_file.assert_called_once_with("src/unknown.py")
        graph.delete_nodes_for_file.assert_called_once_with("src/unknown.py")
        vector_store.delete_embeddings.assert_not_called()
