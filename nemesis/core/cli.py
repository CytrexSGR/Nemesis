"""Nemesis CLI — command-line interface for indexing, querying, and serving."""

from __future__ import annotations

import click

from nemesis import __version__


@click.group()
@click.version_option(version=__version__, prog_name="Nemesis")
def main() -> None:
    """Nemesis — GraphRAG context engine for AI coding agents."""


@main.command()
@click.argument("path", default=".", type=click.Path(exists=False))
@click.option("--languages", "-l", multiple=True, help="Languages to index.")
def index(path: str, languages: tuple[str, ...]) -> None:
    """Index a project directory."""
    click.echo(f"Indexing {path}...")


@main.command()
@click.argument("query_text")
@click.option("--limit", "-n", default=10, help="Max results.")
def query(query_text: str, limit: int) -> None:
    """Query the code graph with natural language."""
    click.echo(f"Querying: {query_text}")


@main.command()
@click.argument("path", default=".", type=click.Path(exists=False))
def watch(path: str) -> None:
    """Watch a project directory for changes."""
    click.echo(f"Watching {path}...")


@main.command()
@click.option("--host", default="localhost", help="Server host.")
@click.option("--port", default=3333, help="Server port.")
def serve(host: str, port: int) -> None:
    """Start the MCP server."""
    click.echo(f"Starting MCP server on {host}:{port}...")


if __name__ == "__main__":
    main()
