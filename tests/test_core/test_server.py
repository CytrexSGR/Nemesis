"""Tests for the Nemesis MCP Server."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from nemesis.core.server import (
    _TOOL_DISPATCH,
    TOOL_DEFINITIONS,
    create_mcp_server,
)

# ---------------------------------------------------------------
# TestToolDefinitions
# ---------------------------------------------------------------


class TestToolDefinitions:
    """Tests for TOOL_DEFINITIONS and _TOOL_DISPATCH consistency."""

    def test_all_tools_defined(self):
        """TOOL_DEFINITIONS has exactly 8 tool entries."""
        assert len(TOOL_DEFINITIONS) == 8

    def test_all_tools_have_dispatch(self):
        """Every tool in TOOL_DEFINITIONS has a corresponding entry in _TOOL_DISPATCH."""
        definition_names = {td["name"] for td in TOOL_DEFINITIONS}
        dispatch_names = set(_TOOL_DISPATCH.keys())
        assert definition_names == dispatch_names


# ---------------------------------------------------------------
# TestCreateMcpServer
# ---------------------------------------------------------------


class TestCreateMcpServer:
    """Tests for the create_mcp_server factory function."""

    def test_creates_server(self):
        """create_mcp_server returns a Server instance."""
        from mcp.server import Server

        mock_engine = MagicMock()
        server = create_mcp_server(mock_engine)
        assert isinstance(server, Server)
        assert server.name == "nemesis"

    async def test_list_tools_returns_all(self):
        """The list_tools handler returns all 8 tools."""
        import mcp.types as types

        mock_engine = MagicMock()
        server = create_mcp_server(mock_engine)

        # The list_tools handler is registered for ListToolsRequest.
        # We invoke it directly via the request_handlers dict.
        handler = server.request_handlers[types.ListToolsRequest]
        result = await handler(None)

        # The handler wraps the result in ServerResult(ListToolsResult(tools=...))
        assert isinstance(result, types.ServerResult)
        tools_result = result.root
        assert isinstance(tools_result, types.ListToolsResult)
        assert len(tools_result.tools) == 8

        tool_names = {t.name for t in tools_result.tools}
        expected_names = {td["name"] for td in TOOL_DEFINITIONS}
        assert tool_names == expected_names

    async def test_call_tool_dispatches_correctly(self):
        """call_tool for 'get_memory' dispatches to tool_funcs.get_memory."""
        import mcp.types as types

        mock_engine = MagicMock()
        memory_result = {"rules": [], "decisions": [], "conventions": [], "total": 0}

        mock_dispatch = {"get_memory": MagicMock(return_value=memory_result)}
        with patch("nemesis.core.server._TOOL_DISPATCH", mock_dispatch):
            server = create_mcp_server(mock_engine)

        handler = server.request_handlers[types.CallToolRequest]

        # Build a mock CallToolRequest
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="get_memory", arguments={}),
        )
        result = await handler(request)

        assert isinstance(result, types.ServerResult)
        call_result = result.root
        assert isinstance(call_result, types.CallToolResult)
        assert call_result.isError is False


# ---------------------------------------------------------------
# TestCallTool
# ---------------------------------------------------------------


class TestCallTool:
    """Tests for call_tool edge cases."""

    async def test_call_unknown_tool(self):
        """Calling an unknown tool returns an error response."""
        import mcp.types as types

        mock_engine = MagicMock()
        server = create_mcp_server(mock_engine)

        handler = server.request_handlers[types.CallToolRequest]

        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="nonexistent_tool", arguments={}),
        )
        result = await handler(request)

        assert isinstance(result, types.ServerResult)
        call_result = result.root
        assert isinstance(call_result, types.CallToolResult)
        # The call_tool handler returns a list with a TextContent containing the error
        # The MCP framework wraps it into a CallToolResult
        content_text = call_result.content[0].text
        parsed = json.loads(content_text)
        assert "error" in parsed
        assert "nonexistent_tool" in parsed["error"]

    async def test_call_tool_with_arguments(self):
        """search_code with query='test' is called with the correct arguments."""
        import mcp.types as types

        mock_engine = MagicMock()
        search_result = {"query": "test", "results": [], "count": 0}
        mock_search = MagicMock(return_value=search_result)

        with patch.dict("nemesis.core.server._TOOL_DISPATCH", {"search_code": mock_search}):
            server = create_mcp_server(mock_engine)

            handler = server.request_handlers[types.CallToolRequest]
            request = types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="search_code",
                    arguments={"query": "test"},
                ),
            )
            result = await handler(request)

        mock_search.assert_called_once_with(mock_engine, query="test")
        call_result = result.root
        content_text = call_result.content[0].text
        parsed = json.loads(content_text)
        assert parsed["query"] == "test"

    async def test_call_tool_handles_exception(self):
        """When a tool function raises an exception, it is returned as an error dict."""
        import mcp.types as types

        mock_engine = MagicMock()
        mock_func = MagicMock(side_effect=ValueError("something broke"))

        with patch.dict("nemesis.core.server._TOOL_DISPATCH", {"get_memory": mock_func}):
            server = create_mcp_server(mock_engine)

            handler = server.request_handlers[types.CallToolRequest]
            request = types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(name="get_memory", arguments={}),
            )
            result = await handler(request)

        call_result = result.root
        content_text = call_result.content[0].text
        parsed = json.loads(content_text)
        assert "error" in parsed
        assert "something broke" in parsed["error"]


# ---------------------------------------------------------------
# TestRunStdioServer
# ---------------------------------------------------------------


class TestRunStdioServer:
    """Tests for the run_stdio_server coroutine."""

    async def test_run_creates_engine(self):
        """run_stdio_server initializes and closes the engine."""
        from nemesis.core.config import NemesisConfig

        mock_engine = MagicMock()
        mock_server = MagicMock()
        mock_server.run = AsyncMock()
        mock_server.create_initialization_options.return_value = MagicMock()

        mock_read = MagicMock()
        mock_write = MagicMock()

        with (
            patch("nemesis.core.server.NemesisEngine", return_value=mock_engine) as mock_engine_cls,
            patch("nemesis.core.server.create_mcp_server", return_value=mock_server),
            patch("mcp.server.stdio.stdio_server") as mock_stdio,
        ):
            # stdio_server is an async context manager
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_stdio.return_value = mock_cm

            from nemesis.core.server import run_stdio_server

            config = NemesisConfig()
            await run_stdio_server(config)

        mock_engine_cls.assert_called_once_with(config)
        mock_engine.initialize.assert_called_once()
        mock_engine.close.assert_called_once()
        mock_server.run.assert_awaited_once()

    async def test_run_uses_default_config(self):
        """When no config is provided, a default NemesisConfig is used."""
        mock_engine = MagicMock()
        mock_server = MagicMock()
        mock_server.run = AsyncMock()
        mock_server.create_initialization_options.return_value = MagicMock()

        mock_read = MagicMock()
        mock_write = MagicMock()

        with (
            patch("nemesis.core.server.NemesisEngine", return_value=mock_engine) as mock_engine_cls,
            patch("nemesis.core.server.create_mcp_server", return_value=mock_server),
            patch("mcp.server.stdio.stdio_server") as mock_stdio,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_stdio.return_value = mock_cm

            from nemesis.core.server import run_stdio_server

            await run_stdio_server()

        # Engine should have been created with a NemesisConfig instance (default)
        args = mock_engine_cls.call_args
        config_arg = args[0][0]
        from nemesis.core.config import NemesisConfig

        assert isinstance(config_arg, NemesisConfig)
        mock_engine.initialize.assert_called_once()
        mock_engine.close.assert_called_once()
