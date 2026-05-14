"""
mcp_tools_server.py
===================
Mycelium MCP tool server.

Exposes the four sandbox tools as proper MCP tools over stdio transport.
Started as a subprocess by MCPClient; communicates via stdin/stdout using
the MCP JSON-RPC protocol.

Tools
-----
  web_search        — DuckDuckGo text search (ddgs preferred, instant-answer fallback, HTML fallback)
  academic_search   — Semantic Scholar paper search
  knowledge_base    — Wikidata entity lookup via wbsearchentities REST API
  calculator        — Safe arithmetic / math expression evaluator

Changes (2026-05-14 — patch 2)
------------------------------
  web_search    : when all three backends drain, return
                  {"results": [], "status": "empty", "note": "..."}  instead of
                  {"results": [], "error": "..."}.  The 'error' key was absent
                  from the output dict so the conversation-agent prompt-builder
                  hit the json.dumps fallback and emitted "FAILED — unknown error".
                  'status': 'empty' gives the agent a clean, first-person-
                  compatible branch to render.

Changes (2026-05-14 — patch 1)
------------------------------
  web_search    : added region='wt-wt', safesearch='off' to DDGS call;
                  added DuckDuckGo instant-answer fallback before HTML scrape.
  knowledge_base: replaced SERVICE wikibase:mwapi SPARQL with wbsearchentities
                  REST API.

Usage (standalone test)
-----------------------
    python mcp_tools_server.py

The server reads MCP JSON-RPC messages from stdin and writes responses to
stdout.  Do not print anything else to stdout — it will corrupt the stream.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from typing import Any, Dict

import requests

# MCP SDK — `pip install mcp`
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types as mcp_types

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

_TOOL_TIMEOUT_S = 8.0

# ---------------------------------------------------------------------------
# Tool implementations (pure functions, no LLM)
# ---------------------------------------------------------------------------

def _web_search(query: str) -> Dict[str, Any]:
    """DuckDuckGo search — ddgs preferred, instant-answer fallback, HTML scrape last resort."""
    # ── primary: duckduckgo-search library ──────────────────────────────────
    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=5,
                    region="wt-wt",
                    safesearch="off",
                )
            )
        if results:
            snippets = [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                }
                for r in results
            ]
            return {"results": snippets, "source": "duckduckgo"}
        # fall through on empty list
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("_web_search ddgs error: %s", exc)

    # ── secondary: DuckDuckGo instant-answer API ────────────────────────────
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
                "no_redirect": "1",
            },
            headers={"User-Agent": "MyceliumPoC/0.4"},
            timeout=_TOOL_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        results_list = []
        if data.get("AbstractText"):
            results_list.append({
                "title": data.get("Heading", ""),
                "snippet": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
                "source": "duckduckgo_instant",
            })
        for t in data.get("RelatedTopics", [])[:4]:
            if isinstance(t, dict) and t.get("Text"):
                results_list.append({
                    "title": t.get("Text", "")[:80],
                    "snippet": t.get("Text", ""),
                    "url": t.get("FirstURL", ""),
                    "source": "duckduckgo_instant",
                })
        if results_list:
            return {"results": results_list, "source": "duckduckgo_instant"}
    except Exception as exc:
        logger.warning("_web_search instant-answer error: %s", exc)

    # ── tertiary: HTML scrape ───────────────────────────────────────────────
    try:
        import re
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (MyceliumPoC)"},
            timeout=_TOOL_TIMEOUT_S,
        )
        resp.raise_for_status()
        snippets_raw = re.findall(
            r'class="result__snippet">(.*?)</a>', resp.text, re.DOTALL
        )
        titles_raw = re.findall(
            r'class="result__a"[^>]*>(.*?)</a>', resp.text, re.DOTALL
        )
        results_list = [
            {
                "title": re.sub(r"<[^>]+>", "", t).strip(),
                "snippet": re.sub(r"<[^>]+>", "", s).strip(),
            }
            for t, s in zip(titles_raw[:5], snippets_raw[:5])
        ]
        if results_list:
            return {"results": results_list, "source": "duckduckgo_html"}
    except Exception as exc:
        logger.warning("_web_search html-scrape error: %s", exc)

    # ── all backends exhausted — return a clean 'empty' status ──────────────
    # Use 'status': 'empty' (not 'error') so the conversation-agent
    # prompt-builder can render a clean "returned no results" line rather
    # than falling through to the json.dumps catch-all.
    return {
        "results": [],
        "status": "empty",
        "note": "all web search backends returned no results for this query",
        "source": "web_search",
    }


def _academic_search(query: str) -> Dict[str, Any]:
    """Semantic Scholar public API — no key required."""
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": 4,
            "fields": "title,authors,year,abstract,externalIds",
        }
        resp = requests.get(url, params=params, timeout=_TOOL_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        papers = [
            {
                "title": p.get("title", ""),
                "year": p.get("year"),
                "authors": [
                    a.get("name", "") for a in (p.get("authors") or [])[:3]
                ],
                "abstract": (p.get("abstract") or "")[:300],
            }
            for p in data.get("data", [])
        ]
        return {"papers": papers, "source": "semantic_scholar"}
    except Exception as exc:
        return {"error": str(exc), "source": "academic_search"}


def _knowledge_base(query: str) -> Dict[str, Any]:
    """
    Wikidata entity lookup via the wbsearchentities REST API.

    Replaces the previous SERVICE wikibase:mwapi SPARQL approach which
    was silently returning empty bindings for non-browser User-Agent strings
    on the public SPARQL endpoint.
    """
    try:
        resp = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "limit": 5,
                "format": "json",
                "type": "item",
            },
            headers={"User-Agent": "MyceliumPoC/0.4 (https://github.com/shasankp000/Mycelium)"},
            timeout=_TOOL_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        search_results = data.get("search", [])
        entities = [
            {
                "id": item.get("id", ""),
                "label": item.get("label", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
            }
            for item in search_results
        ]
        return {"entities": entities, "source": "wikidata"}
    except Exception as exc:
        return {"error": str(exc), "source": "knowledge_base"}


def _calculator(expression: str) -> Dict[str, Any]:
    """Safe arithmetic evaluator using math module only."""
    allowed: Dict[str, Any] = {
        k: getattr(math, k) for k in dir(math) if not k.startswith("_")
    }
    allowed.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return {"result": result, "expression": expression, "source": "calculator"}
    except Exception as exc:
        return {"error": str(exc), "expression": expression, "source": "calculator"}


_TOOL_FN = {
    "web_search": _web_search,
    "academic_search": _academic_search,
    "knowledge_base": _knowledge_base,
    "calculator": _calculator,
}

# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

server = Server("mycelium-tools")


@server.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="web_search",
            description="General web search via DuckDuckGo. Use for recent news, current facts, or broad factual claims.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="academic_search",
            description="Peer-reviewed paper search via Semantic Scholar. Use for scientific, medical, or technical claims.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Academic search query"}
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="knowledge_base",
            description="Structured entity facts via Wikidata. Use for entity definitions, taxonomy, and factual attributes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Entity name to look up"}
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="calculator",
            description="Safe arithmetic evaluator. Use for numeric expressions, unit conversions, and mathematical computations only.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python-compatible math expression, e.g. 'sqrt(2) * pi'",
                    }
                },
                "required": ["expression"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[mcp_types.TextContent]:
    fn = _TOOL_FN.get(name)
    if fn is None:
        result = {"error": f"Unknown tool: {name}"}
    else:
        arg = arguments.get("query") or arguments.get("expression") or ""
        try:
            result = fn(arg)
        except Exception as exc:
            result = {"error": str(exc)}

    return [mcp_types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def _main() -> None:
    async with stdio_server() as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
