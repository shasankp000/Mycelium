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

Changes (2026-05-15 — patch 3)
------------------------------
  web_search    : exponential back-off retry (3 attempts, 1-2-4 s) on
                  DDGS RateException / requests.ConnectionError;
                  per-attempt User-Agent rotation so DDG doesn't fingerprint
                  repeated calls; requests.Session with keep-alive for
                  instant-answer and HTML fallbacks;
                  every code path now returns a consistent dict shape
                  {results/papers/entities/result, status, source, note?}
                  so conversation-agent never sees a bare 'error' key.

Changes (2026-05-14 — patch 2)
------------------------------
  web_search    : return {"results": [], "status": "empty", "note": "..."}
                  instead of {"results": [], "error": "..."} on all-backends-
                  exhausted path.

Changes (2026-05-14 — patch 1)
------------------------------
  web_search    : added region='wt-wt', safesearch='off';
                  DuckDuckGo instant-answer fallback before HTML scrape.
  knowledge_base: replaced SERVICE wikibase:mwapi SPARQL with
                  wbsearchentities REST API.
"""
from __future__ import annotations

import json
import logging
import math
import random
import sys
import time
from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types as mcp_types

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

_TOOL_TIMEOUT_S = 10.0
_DDGS_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Shared HTTP session (connection pooling + automatic retry on 5xx)
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

_SESSION = _make_session()

_USER_AGENTS = [
    "MyceliumPoC/0.5 (research bot; +https://github.com/shasankp000/Mycelium)",
    "Mozilla/5.0 (compatible; MyceliumBot/0.5; +https://github.com/shasankp000/Mycelium)",
    "MyceliumPoC/0.4 (sandbox search agent)",
]


def _ua() -> str:
    """Return a random User-Agent string to avoid DDG fingerprinting."""
    return random.choice(_USER_AGENTS)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _web_search(query: str) -> Dict[str, Any]:
    """
    DuckDuckGo search.

    Attempt order:
      1. duckduckgo-search library (DDGS) — retried up to 3× with
         exponential back-off on RateException / ConnectionError
      2. DuckDuckGo instant-answer JSON API
      3. DuckDuckGo HTML scrape

    All code paths return a consistent shape:
      {results: [...], status: 'ok'|'empty', source: str, note?: str}
    """
    # ── primary: DDGS library with retry ────────────────────────────────────
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import RateLimitException  # type: ignore[attr-defined]
    except ImportError:
        RateLimitException = Exception  # type: ignore[assignment,misc]
        DDGS = None  # type: ignore[assignment]

    if DDGS is not None:
        last_exc: Exception | None = None
        for attempt in range(1, _DDGS_MAX_RETRIES + 1):
            try:
                with DDGS(headers={"User-Agent": _ua()}) as ddgs:
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
                    return {"results": snippets, "status": "ok", "source": "duckduckgo"}
                # Empty result list — no point retrying
                break
            except (RateLimitException, requests.ConnectionError) as exc:
                last_exc = exc
                wait = 2 ** (attempt - 1)  # 1, 2, 4 s
                logger.warning(
                    "_web_search ddgs attempt %d/%d failed (%s) — retrying in %ds",
                    attempt, _DDGS_MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            except Exception as exc:
                logger.warning("_web_search ddgs error (non-retryable): %s", exc)
                break

        if last_exc:
            logger.warning("_web_search ddgs exhausted retries: %s", last_exc)

    # ── secondary: instant-answer API ───────────────────────────────────────
    try:
        resp = _SESSION.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
                "no_redirect": "1",
            },
            headers={"User-Agent": _ua()},
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
            })
        for t in data.get("RelatedTopics", [])[:4]:
            if isinstance(t, dict) and t.get("Text"):
                results_list.append({
                    "title": t.get("Text", "")[:80],
                    "snippet": t.get("Text", ""),
                    "url": t.get("FirstURL", ""),
                })
        if results_list:
            return {"results": results_list, "status": "ok", "source": "duckduckgo_instant"}
    except Exception as exc:
        logger.warning("_web_search instant-answer error: %s", exc)

    # ── tertiary: HTML scrape ────────────────────────────────────────────────
    try:
        import re
        resp = _SESSION.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": _ua()},
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
            return {"results": results_list, "status": "ok", "source": "duckduckgo_html"}
    except Exception as exc:
        logger.warning("_web_search html-scrape error: %s", exc)

    return {
        "results": [],
        "status": "empty",
        "note": "All web search backends returned no results for this query.",
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
        resp = _SESSION.get(url, params=params, timeout=_TOOL_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        papers = [
            {
                "title": p.get("title", ""),
                "year": p.get("year"),
                "authors": [a.get("name", "") for a in (p.get("authors") or [])[:3]],
                "abstract": (p.get("abstract") or "")[:300],
            }
            for p in data.get("data", [])
        ]
        return {"papers": papers, "status": "ok", "source": "semantic_scholar"}
    except Exception as exc:
        return {"papers": [], "status": "error", "note": str(exc), "source": "academic_search"}


def _knowledge_base(query: str) -> Dict[str, Any]:
    """Wikidata entity lookup via wbsearchentities REST API."""
    try:
        resp = _SESSION.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "limit": 5,
                "format": "json",
                "type": "item",
            },
            headers={"User-Agent": "MyceliumPoC/0.5 (https://github.com/shasankp000/Mycelium)"},
            timeout=_TOOL_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        entities = [
            {
                "id": item.get("id", ""),
                "label": item.get("label", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
            }
            for item in data.get("search", [])
        ]
        return {"entities": entities, "status": "ok", "source": "wikidata"}
    except Exception as exc:
        return {"entities": [], "status": "error", "note": str(exc), "source": "knowledge_base"}


def _calculator(expression: str) -> Dict[str, Any]:
    """Safe arithmetic evaluator using math module only."""
    allowed: Dict[str, Any] = {
        k: getattr(math, k) for k in dir(math) if not k.startswith("_")
    }
    allowed.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return {"result": result, "expression": expression, "status": "ok", "source": "calculator"}
    except Exception as exc:
        return {"result": None, "expression": expression, "status": "error", "note": str(exc), "source": "calculator"}


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
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="academic_search",
            description="Peer-reviewed paper search via Semantic Scholar. Use for scientific, medical, or technical claims.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Academic search query"}},
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name="knowledge_base",
            description="Structured entity facts via Wikidata. Use for entity definitions, taxonomy, and factual attributes.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Entity name to look up"}},
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
        result = {"error": f"Unknown tool: {name}", "status": "error", "source": name}
    else:
        arg = arguments.get("query") or arguments.get("expression") or ""
        try:
            result = fn(arg)
        except Exception as exc:
            result = {"error": str(exc), "status": "error", "source": name}

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
