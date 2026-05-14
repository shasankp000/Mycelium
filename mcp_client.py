"""
mcp_client.py
=============
Thin async MCP client for Mycelium's sandbox layer.

Starts mcp_tools_server.py as a subprocess and communicates via stdio
transport using the MCP JSON-RPC protocol.  Exposes a simple synchronous
interface (call_tool) so sandbox_manager.py needs no async changes.

Lifecycle
---------
The client is created lazily and kept alive for the process lifetime
(process-level singleton via get_mcp_client()).  The subprocess is started
on the first call and reused for all subsequent tool calls.

If the MCP server subprocess crashes, the client re-spawns it on the next
call — so a single tool failure cannot permanently break the sandbox.

Usage
-----
    from mcp_client import get_mcp_client

    client = get_mcp_client()
    result = client.call_tool("web_search", {"query": "string theory"})
    # result: dict  — parsed JSON output from the tool
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Path to the MCP server script, resolved relative to this file.
_SERVER_SCRIPT = Path(__file__).parent / "mcp_tools_server.py"


class MCPClient:
    """
    Synchronous wrapper around the async MCP stdio client.

    A private event loop runs in a background daemon thread so that
    sandbox_manager.py (which is not async) can call tools with a simple
    blocking call_tool() method.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="mcp-client-loop",
        )
        self._thread.start()
        self._session: Any = None           # mcp.ClientSession
        self._init_lock = threading.Lock()
        self._available_tools: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available_tools(self) -> list[str]:
        """Names of tools exposed by the MCP server."""
        if not self._available_tools:
            self._ensure_session()
        return list(self._available_tools)

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server and return its parsed JSON output.

        Parameters
        ----------
        tool_name  : str  — must be one of available_tools
        arguments  : dict — forwarded verbatim to the MCP tool

        Returns
        -------
        dict — parsed JSON result from the tool, or {"error": ...} on failure
        """
        try:
            self._ensure_session()
            future = asyncio.run_coroutine_threadsafe(
                self._call_tool_async(tool_name, arguments),
                self._loop,
            )
            return future.result(timeout=15)
        except Exception as exc:
            logger.warning("MCPClient.call_tool('%s') failed: %s", tool_name, exc)
            # Session may be stale — reset so next call re-spawns
            self._session = None
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_session(self) -> None:
        """Initialise the MCP session if it is not already alive."""
        with self._init_lock:
            if self._session is not None:
                return
            future = asyncio.run_coroutine_threadsafe(
                self._init_session(), self._loop
            )
            try:
                future.result(timeout=10)
            except Exception as exc:
                logger.error("MCPClient: session init failed: %s", exc)
                raise

    async def _init_session(self) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(_SERVER_SCRIPT)],
        )
        # The stdio_client context manager must stay open for the session
        # lifetime.  We store the context + session as instance state.
        self._stdio_ctx = stdio_client(params)
        read_stream, write_stream = await self._stdio_ctx.__aenter__()

        self._session_ctx = ClientSession(read_stream, write_stream)
        self._session = await self._session_ctx.__aenter__()

        await self._session.initialize()

        tools_response = await self._session.list_tools()
        self._available_tools = [t.name for t in tools_response.tools]
        logger.info(
            "MCPClient: server ready — tools: %s",
            self._available_tools,
        )

    async def _call_tool_async(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = await self._session.call_tool(tool_name, arguments=arguments)
        # MCP returns a list of content blocks; we expect one TextContent
        for block in result.content:
            if hasattr(block, "text"):
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    return {"result": block.text}
        return {"error": "empty response from MCP server"}


# Process-level singleton
_client: Optional[MCPClient] = None
_client_lock = threading.Lock()


def get_mcp_client() -> MCPClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = MCPClient()
    return _client
