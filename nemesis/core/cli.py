"""Nemesis CLI — command-line interface for indexing, querying, and serving."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from nemesis import __version__
from nemesis.core.config import NemesisConfig
from nemesis.core.engine import NemesisEngine


@click.group()
@click.version_option(version=__version__, prog_name="Nemesis")
def main() -> None:
    """Nemesis — GraphRAG context engine for AI coding agents."""


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--languages", "-l", multiple=True, help="Languages to index.")
def index(path: str, languages: tuple[str, ...]) -> None:
    """Index a project directory."""
    config = NemesisConfig(project_root=Path(path))
    if languages:
        config = NemesisConfig(project_root=Path(path), languages=list(languages))

    click.echo(f"Indexing {path}...")

    try:
        with NemesisEngine(config) as engine:
            result = engine.pipeline.index_project(
                Path(path),
                languages=config.languages,
                ignore_dirs=set(config.ignore_patterns),
            )

        click.echo(f"Indexed {result.files_indexed} files")
        click.echo(f"  Nodes: {result.nodes_created}")
        click.echo(f"  Edges: {result.edges_created}")
        click.echo(f"  Chunks: {result.chunks_created}")
        click.echo(f"  Duration: {result.duration_ms:.0f}ms")

        if result.errors:
            click.echo(f"  Errors: {len(result.errors)}", err=True)
            for error in result.errors:
                click.echo(f"    - {error}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("query_text")
@click.option("--limit", "-n", default=10, help="Max results.")
@click.option(
    "--project-root",
    "-p",
    default=".",
    type=click.Path(exists=True),
    help="Project root.",
)
def query(query_text: str, limit: int, project_root: str) -> None:
    """Query the code graph with natural language."""
    config = NemesisConfig(project_root=Path(project_root))

    try:
        with NemesisEngine(config) as engine:
            embedding = engine.embedder.embed_single(query_text)
            results = engine.vector_store.search(embedding, limit=limit)

        if not results:
            click.echo("No results found.")
            return

        for i, r in enumerate(results, 1):
            click.echo(f"\n[{i}] Score: {r.score:.4f}")
            click.echo(f"    File: {r.metadata.get('file', 'unknown')}")
            if r.metadata.get("start_line"):
                click.echo(
                    f"    Lines: {r.metadata['start_line']}-"
                    f"{r.metadata.get('end_line', '?')}"
                )
            # Show truncated text preview
            preview = r.text[:200].replace("\n", " ")
            if len(r.text) > 200:
                preview += "..."
            click.echo(f"    {preview}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--languages", "-l", multiple=True, help="Languages to watch.")
def watch(path: str, languages: tuple[str, ...]) -> None:
    """Watch a project directory for changes and auto-reindex."""
    from nemesis.core.watcher import FileWatcher
    from nemesis.indexer.delta import _get_extensions

    config = NemesisConfig(project_root=Path(path))
    if languages:
        config = NemesisConfig(project_root=Path(path), languages=list(languages))

    extensions = _get_extensions(config.languages)

    click.echo(f"Watching {path} for changes...")
    click.echo(f"  Languages: {', '.join(config.languages)}")
    click.echo(f"  Extensions: {', '.join(sorted(extensions))}")
    click.echo("Press Ctrl+C to stop.")

    engine = NemesisEngine(config)
    engine.initialize()

    def on_change(file_path: Path) -> None:
        click.echo(f"  Changed: {file_path}")
        try:
            result = engine.pipeline.reindex_file(file_path)
            if result.success:
                click.echo(f"  Reindexed: {result.nodes_created} nodes")
            else:
                click.echo(f"  Errors: {result.errors}", err=True)
        except Exception as e:
            click.echo(f"  Reindex error: {e}", err=True)

    watcher = FileWatcher(
        root=Path(path),
        callback=on_change,
        extensions=extensions,
        ignore_dirs=set(config.ignore_patterns),
        debounce_ms=config.watcher_debounce_ms,
    )

    try:
        watcher.start()
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping watcher...")
    finally:
        watcher.stop()
        engine.close()


@main.command()
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio"]),
    help="MCP transport.",
)
def serve(transport: str) -> None:
    """Start the MCP server."""
    click.echo(f"Starting Nemesis MCP server ({transport} transport)...")
    click.echo("MCP server not yet implemented.")


if __name__ == "__main__":
    main()
