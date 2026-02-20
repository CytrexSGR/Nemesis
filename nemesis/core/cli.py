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
@click.option("--name", "-n", default=None, help="Custom project name (default: directory name).")
def index(path: str, languages: tuple[str, ...], name: str | None) -> None:
    """Index a project directory."""
    config = NemesisConfig()
    if languages:
        config = NemesisConfig(languages=list(languages))

    click.echo(f"Indexing {path}...")

    try:
        with NemesisEngine(config) as engine:
            from nemesis.tools import tools as tool_funcs

            result_dict = tool_funcs.index_project(
                engine,
                path,
                languages=list(languages) if languages else None,
                name=name,
            )

        click.echo(f"Project: {result_dict['project']}")
        click.echo(f"Indexed {result_dict['files_indexed']} files")
        click.echo(f"  Nodes: {result_dict['nodes_created']}")
        click.echo(f"  Edges: {result_dict['edges_created']}")
        click.echo(f"  Chunks: {result_dict['chunks_created']}")
        click.echo(f"  Embeddings: {result_dict['embeddings_created']}")
        click.echo(f"  Duration: {result_dict['duration_ms']:.0f}ms")

        if result_dict["errors"]:
            click.echo(f"  Errors: {len(result_dict['errors'])}", err=True)
            for error in result_dict["errors"]:
                click.echo(f"    - {error}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("query_text")
@click.option("--limit", "-n", default=10, help="Max results.")
@click.option("--project", "-p", default=None, help="Filter to specific project.")
def query(query_text: str, limit: int, project: str | None) -> None:
    """Query the code graph with natural language."""
    config = NemesisConfig()

    try:
        with NemesisEngine(config) as engine:
            embedding = engine.embedder.embed_single(query_text)
            results = engine.vector_store.search(embedding, limit=limit, project_id=project)

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
@click.option("--name", "-n", default=None, help="Project name for reindexing.")
def watch(path: str, languages: tuple[str, ...], name: str | None) -> None:
    """Watch a project directory for changes and auto-reindex."""
    from nemesis.core.watcher import FileWatcher
    from nemesis.indexer.delta import _get_extensions

    config = NemesisConfig()
    if languages:
        config = NemesisConfig(languages=list(languages))

    extensions = _get_extensions(config.languages)
    project_id = name or Path(path).name

    click.echo(f"Watching {path} for changes (project: {project_id})...")
    click.echo(f"  Languages: {', '.join(config.languages)}")
    click.echo(f"  Extensions: {', '.join(sorted(extensions))}")
    click.echo("Press Ctrl+C to stop.")

    engine = NemesisEngine(config)
    engine.initialize()

    def on_change(file_path: Path) -> None:
        click.echo(f"  Changed: {file_path}")
        try:
            result = engine.pipeline.reindex_file(
                file_path,
                project_id=project_id,
                project_root=Path(path),
            )
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
def projects() -> None:
    """List all registered projects."""
    config = NemesisConfig()

    try:
        with NemesisEngine(config) as engine:
            from nemesis.tools import tools as tool_funcs

            result = tool_funcs.list_projects(engine)

        if result["count"] == 0:
            click.echo("No projects registered.")
            return

        for name, info in result["projects"].items():
            click.echo(f"\n{name}")
            click.echo(f"  Path: {info['path']}")
            click.echo(f"  Languages: {', '.join(info['languages'])}")
            click.echo(f"  Files: {info['files']}")
            if info.get("indexed_at"):
                click.echo(f"  Indexed: {info['indexed_at']}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("name")
def remove(name: str) -> None:
    """Remove a project and its indexed data."""
    config = NemesisConfig()

    try:
        with NemesisEngine(config) as engine:
            from nemesis.tools import tools as tool_funcs

            result = tool_funcs.remove_project(engine, name)

        if result.get("success"):
            click.echo(f"Removed project '{name}'.")
        else:
            click.echo(f"Error: {result.get('error', 'Unknown error')}", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio"]),
    help="MCP transport.",
)
def serve(transport: str) -> None:
    """Start the MCP server."""
    import asyncio

    from nemesis.core.server import run_stdio_server

    # Log to stderr -- stdout is reserved for MCP JSON-RPC protocol
    click.echo(f"Starting Nemesis MCP server ({transport} transport)...", err=True)
    asyncio.run(run_stdio_server())


if __name__ == "__main__":
    main()
