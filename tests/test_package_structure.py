"""Tests for package structure — all subpackages must be importable."""


def test_nemesis_package_exists():
    """The root nemesis package is importable and has a version."""
    import nemesis

    assert hasattr(nemesis, "__version__")
    assert nemesis.__version__ == "0.1.0"


def test_core_subpackage():
    """The core subpackage is importable."""
    import nemesis.core

    assert nemesis.core is not None


def test_indexer_subpackage():
    """The indexer subpackage is importable."""
    import nemesis.indexer

    assert nemesis.indexer is not None


def test_parser_subpackage():
    """The parser subpackage is importable."""
    import nemesis.parser

    assert nemesis.parser is not None


def test_graph_subpackage():
    """The graph subpackage is importable."""
    import nemesis.graph

    assert nemesis.graph is not None


def test_vector_subpackage():
    """The vector subpackage is importable."""
    import nemesis.vector

    assert nemesis.vector is not None


def test_memory_subpackage():
    """The memory subpackage is importable."""
    import nemesis.memory

    assert nemesis.memory is not None


def test_tools_subpackage():
    """The tools subpackage is importable."""
    import nemesis.tools

    assert nemesis.tools is not None


def test_all_subpackages_list():
    """All expected subpackages are present."""
    expected = {"core", "indexer", "parser", "graph", "vector", "memory", "tools"}

    for pkg in expected:
        __import__(f"nemesis.{pkg}")
