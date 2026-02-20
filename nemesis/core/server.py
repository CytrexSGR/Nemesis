"""Nemesis MCP Server -- exposes tools, resources, and prompts via MCP protocol."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from nemesis.core.config import NemesisConfig
from nemesis.core.engine import NemesisEngine
from nemesis.tools import tools as tool_funcs

# Tool definitions with JSON Schema for parameters
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_code",
        "description": "Search code using natural language query with vector similarity",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_context",
        "description": "Get context for a file from the code graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "depth": {"type": "integer", "description": "Traversal depth", "default": 2},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "index_project",
        "description": "Index a project directory",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Project directory path"},
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Languages to index",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "update_project",
        "description": "Run delta update on a project directory",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Project directory path"},
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Languages to index",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "remember_rule",
        "description": "Store a coding rule in the knowledge graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Rule content"},
                "scope": {
                    "type": "string",
                    "description": "Scope: project, file, function",
                    "default": "project",
                },
                "source": {
                    "type": "string",
                    "description": "Source of the rule",
                    "default": "user",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "remember_decision",
        "description": "Store an architectural decision",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Decision title"},
                "reasoning": {"type": "string", "description": "Reasoning", "default": ""},
                "status": {
                    "type": "string",
                    "description": "Status: proposed, accepted, rejected",
                    "default": "accepted",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "get_memory",
        "description": "Retrieve all stored rules, decisions, and conventions",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_session_summary",
        "description": "Get a summary of the current session context",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]

# Mapping tool name -> function
_TOOL_DISPATCH: dict[str, Any] = {
    "search_code": tool_funcs.search_code,
    "get_context": tool_funcs.get_context,
    "index_project": tool_funcs.index_project,
    "update_project": tool_funcs.update_project,
    "remember_rule": tool_funcs.remember_rule,
    "remember_decision": tool_funcs.remember_decision,
    "get_memory": tool_funcs.get_memory,
    "get_session_summary": tool_funcs.get_session_summary,
}


def create_mcp_server(engine: NemesisEngine) -> Server:
    """Create and configure an MCP Server with all Nemesis tools.

    Args:
        engine: Initialized NemesisEngine instance.

    Returns:
        Configured mcp.server.Server ready to run.
    """
    server = Server("nemesis")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=td["name"],
                description=td["description"],
                inputSchema=td["inputSchema"],
            )
            for td in TOOL_DEFINITIONS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        func = _TOOL_DISPATCH.get(name)
        if func is None:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

        try:
            # Run in thread pool — sync wrappers can't run in async context
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: func(engine, **arguments))
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


async def run_stdio_server(config: NemesisConfig | None = None) -> None:
    """Run the MCP server with stdio transport.

    Args:
        config: Optional NemesisConfig. Uses defaults if not provided.
    """
    from mcp.server.stdio import stdio_server

    engine = NemesisEngine(config or NemesisConfig())

    # Initialize in thread pool — sync wrappers create own event loops
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.initialize)

    server = create_mcp_server(engine)

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await loop.run_in_executor(None, engine.close)
