"""
mcp_tools_server.py
===================
Mycelium MCP tool server — v2 (8 tools).

Exposes all tools registered in tool_registry.py as proper MCP tools
over stdio transport.  Started as a subprocess by MCPClient;
communicates via stdin/stdout using the MCP JSON-RPC protocol.

Tools
-----
  [original — unchanged internals]
  web_search          — DuckDuckGo text search (3-fallback chain)
  academic_search     — Semantic Scholar paper search
  knowledge_base      — Wikidata entity lookup
  calculator          — Safe arithmetic / math expression evaluator

  [new — predicate-type-aware]
  temporal_search     — Wikidata date-property lookup + year-scoped DDG search
  causal_search       — Semantic Scholar causal framing + Open Citations chain
  statistical_search  — World Bank / OECD Stats + DDG fallback
  domain_store_search — SQLite shard FTS5+cosine (STUB until TRM v2)
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types as mcp_types

import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from mycelium.pipeline.tool_registry import TOOL_REGISTRY, TOOL_BY_NAME

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

_TOOL_TIMEOUT_S = 10.0
_DDGS_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Shared HTTP session
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
    "MyceliumPoC/0.6 (research bot; +https://github.com/shasankp000/Mycelium)",
    "Mozilla/5.0 (compatible; MyceliumBot/0.6; +https://github.com/shasankp000/Mycelium)",
    "MyceliumPoC/0.5 (sandbox search agent)",
]


def _ua() -> str:
    return random.choice(_USER_AGENTS)


# ---------------------------------------------------------------------------
# DDGS retry helper (unchanged from v1)
# ---------------------------------------------------------------------------

def _build_ddgs_retryable() -> tuple:
    retryable = [requests.ConnectionError, requests.Timeout]
    try:
        import httpx
        retryable += [
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.HTTPStatusError,
            httpx.RemoteProtocolError,
        ]
    except ImportError:
        pass
    try:
        from duckduckgo_search.exceptions import RateLimitException
        retryable.append(RateLimitException)
    except ImportError:
        pass
    return tuple(set(retryable))


_DDGS_RETRYABLE = _build_ddgs_retryable()


def _ddgs_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Internal helper: run a DuckDuckGo search and return a list of
    {title, snippet, url} dicts.  Uses the same 3-fallback chain as v1.
    Returns [] on total failure (callers decide how to handle).
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

    if DDGS is not None:
        last_exc = None
        for attempt in range(1, _DDGS_MAX_RETRIES + 1):
            try:
                try:
                    ddgs_instance = DDGS(headers={"User-Agent": _ua()})
                except TypeError:
                    ddgs_instance = DDGS()
                with ddgs_instance as ddgs:
                    results = list(
                        ddgs.text(query, max_results=max_results,
                                  region="wt-wt", safesearch="off")
                    )
                if results:
                    return [
                        {
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("href", ""),
                        }
                        for r in results
                    ]
                break
            except _DDGS_RETRYABLE as exc:
                last_exc = exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "_ddgs_search attempt %d/%d failed (%s) — retrying in %ds",
                    attempt, _DDGS_MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            except Exception as exc:
                logger.warning("_ddgs_search non-retryable error: %s", exc)
                break
        if last_exc:
            logger.warning("_ddgs_search exhausted retries: %s", last_exc)

    # Instant-answer fallback
    try:
        resp = _SESSION.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query, "format": "json",
                "no_html": "1", "skip_disambig": "1", "no_redirect": "1",
            },
            headers={"User-Agent": _ua()},
            timeout=_TOOL_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        out = []
        if data.get("AbstractText"):
            out.append({
                "title": data.get("Heading", ""),
                "snippet": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
            })
        for t in data.get("RelatedTopics", [])[:4]:
            if isinstance(t, dict) and t.get("Text"):
                out.append({
                    "title": t.get("Text", "")[:80],
                    "snippet": t.get("Text", ""),
                    "url": t.get("FirstURL", ""),
                })
        if out:
            return out
    except Exception as exc:
        logger.warning("_ddgs_search instant-answer error: %s", exc)

    # HTML scrape last resort
    try:
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
        out = [
            {
                "title": re.sub(r"<[^>]+>", "", t).strip(),
                "snippet": re.sub(r"<[^>]+>", "", s).strip(),
                "url": "",
            }
            for t, s in zip(titles_raw[:max_results], snippets_raw[:max_results])
        ]
        if out:
            return out
    except Exception as exc:
        logger.warning("_ddgs_search html-scrape error: %s", exc)

    return []


# ---------------------------------------------------------------------------
# Original 4 tools (internals unchanged, now delegate to _ddgs_search)
# ---------------------------------------------------------------------------

def _web_search(query: str) -> Dict[str, Any]:
    results = _ddgs_search(query, max_results=5)
    if results:
        return {"results": results, "status": "ok", "source": "duckduckgo"}
    return {
        "results": [],
        "status": "empty",
        "note": "All web search backends returned no results.",
        "source": "web_search",
    }


def _academic_search(query: str) -> Dict[str, Any]:
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
                "authors": [
                    a.get("name", "") for a in (p.get("authors") or [])[:3]
                ],
                "abstract": (p.get("abstract") or "")[:300],
            }
            for p in data.get("data", [])
        ]
        return {"papers": papers, "status": "ok", "source": "semantic_scholar"}
    except Exception as exc:
        return {
            "papers": [],
            "status": "error",
            "note": str(exc),
            "source": "academic_search",
        }


def _knowledge_base(query: str) -> Dict[str, Any]:
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
            headers={"User-Agent": "MyceliumPoC/0.6 (https://github.com/shasankp000/Mycelium)"},
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
        return {
            "entities": [],
            "status": "error",
            "note": str(exc),
            "source": "knowledge_base",
        }


def _calculator(expression: str) -> Dict[str, Any]:
    allowed: Dict[str, Any] = {
        k: getattr(math, k) for k in dir(math) if not k.startswith("_")
    }
    allowed.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return {
            "result": result,
            "expression": expression,
            "status": "ok",
            "source": "calculator",
        }
    except Exception as exc:
        return {
            "result": None,
            "expression": expression,
            "status": "error",
            "note": str(exc),
            "source": "calculator",
        }


# ---------------------------------------------------------------------------
# NEW TOOL 1: temporal_search
# ---------------------------------------------------------------------------
# Strategy:
#   Pass 1 — Wikidata SPARQL for date-anchored entity properties.
#             Extracts P585 (point in time), P571 (inception),
#             P576 (dissolved), P17 (country) for the subject entity.
#   Pass 2 — Year-scoped DuckDuckGo snippet search as corroboration.
#
# The query is expected to contain a temporal signal (year, decade, era).
# If no year is detectable we still run both passes without year scoping.
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")

_WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

_TEMPORAL_SPARQL_TEMPLATE = """
SELECT ?item ?itemLabel ?date ?dateLabel ?propLabel WHERE {{
  ?item rdfs:label "{subject}"@en.
  ?item ?prop ?date.
  ?propEntity wikibase:directClaim ?prop;
              rdfs:label ?propLabel.
  FILTER(LANG(?propLabel) = "en")
  FILTER(DATATYPE(?date) = xsd:dateTime || DATATYPE(?date) = xsd:date)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 10
"""


def _temporal_search(query: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "source": "temporal_search",
        "wikidata_dates": [],
        "web_snippets": [],
        "status": "ok",
    }

    # Extract subject (first noun phrase heuristic: longest capitalised span)
    subject_match = re.search(r"([A-Z][A-Za-z\s]+(?:Act|Policy|War|Treaty|Law|Amendment|Revolution|Crisis)?)", query)
    subject = subject_match.group(1).strip() if subject_match else query[:40]

    # Pass 1: Wikidata SPARQL — date properties
    try:
        sparql = _TEMPORAL_SPARQL_TEMPLATE.format(subject=subject.replace('"', ''))
        resp = _SESSION.get(
            _WIKIDATA_SPARQL,
            params={"query": sparql, "format": "json"},
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "MyceliumPoC/0.6 (https://github.com/shasankp000/Mycelium)",
            },
            timeout=_TOOL_TIMEOUT_S,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        for b in bindings[:8]:
            results["wikidata_dates"].append({
                "entity": b.get("itemLabel", {}).get("value", ""),
                "property": b.get("propLabel", {}).get("value", ""),
                "date": b.get("date", {}).get("value", ""),
            })
    except Exception as exc:
        logger.warning("temporal_search SPARQL error: %s", exc)
        results["wikidata_note"] = str(exc)

    # Pass 2: Year-scoped DDG search
    years = _YEAR_RE.findall(query)
    scoped_query = query
    if years:
        scoped_query = f"{query} {years[0]}"
    snippets = _ddgs_search(scoped_query, max_results=4)
    results["web_snippets"] = snippets

    if not results["wikidata_dates"] and not results["web_snippets"]:
        results["status"] = "empty"
        results["note"] = "No temporal evidence found in Wikidata or web search."

    return results


# ---------------------------------------------------------------------------
# NEW TOOL 2: causal_search
# ---------------------------------------------------------------------------
# Strategy:
#   Pass 1 — Semantic Scholar with causal-framing keyword injection.
#             Reformulates query as "[query] causal effect mechanism".
#   Pass 2 — Open Citations REST API provenance chain for the top paper.
#             Fetches the direct references of the most-cited result.
#             This gives a lightweight provenance trace without a KG.
# ---------------------------------------------------------------------------

_OPENCITATIONS_REFERENCES = "https://opencitations.net/api/v1/references/"


def _causal_search(query: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "source": "causal_search",
        "papers": [],
        "provenance_chain": [],
        "status": "ok",
    }

    causal_query = f"{query} causal effect mechanism evidence"

    # Pass 1: Semantic Scholar
    top_doi: Optional[str] = None
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": causal_query,
            "limit": 5,
            "fields": "title,authors,year,abstract,externalIds,citationCount",
        }
        resp = _SESSION.get(url, params=params, timeout=_TOOL_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        papers_raw = sorted(
            data.get("data", []),
            key=lambda p: p.get("citationCount") or 0,
            reverse=True,
        )
        for p in papers_raw[:4]:
            doi = (p.get("externalIds") or {}).get("DOI")
            if doi and top_doi is None:
                top_doi = doi
            results["papers"].append({
                "title": p.get("title", ""),
                "year": p.get("year"),
                "authors": [
                    a.get("name", "") for a in (p.get("authors") or [])[:3]
                ],
                "abstract": (p.get("abstract") or "")[:300],
                "citations": p.get("citationCount", 0),
                "doi": doi or "",
            })
    except Exception as exc:
        logger.warning("causal_search academic error: %s", exc)
        results["academic_note"] = str(exc)

    # Pass 2: Open Citations provenance chain
    if top_doi:
        try:
            resp = _SESSION.get(
                f"{_OPENCITATIONS_REFERENCES}{top_doi}",
                headers={"User-Agent": _ua()},
                timeout=_TOOL_TIMEOUT_S,
            )
            if resp.status_code == 200:
                chain = resp.json()
                results["provenance_chain"] = [
                    {
                        "citing": entry.get("citing", ""),
                        "cited": entry.get("cited", ""),
                        "creation": entry.get("creation", ""),
                    }
                    for entry in chain[:10]
                ]
        except Exception as exc:
            logger.warning("causal_search opencitations error: %s", exc)
            results["provenance_note"] = str(exc)

    if not results["papers"] and not results["provenance_chain"]:
        results["status"] = "empty"
        results["note"] = "No causal evidence found."

    return results


# ---------------------------------------------------------------------------
# NEW TOOL 3: statistical_search
# ---------------------------------------------------------------------------
# Strategy:
#   Pass 1 — World Bank Open Data Indicators API.
#             Attempts to auto-map the query to a known WB indicator code
#             using a small keyword→code table.  Falls back to WB search.
#   Pass 2 — OECD Stats API (SDMX-JSON) for OECD-specific series.
#             Uses a small keyword→dataflow table.
#   Pass 3 — DDG snippet search as final fallback for unlisted series.
# ---------------------------------------------------------------------------

# Curated keyword → World Bank indicator code mapping.
# Covers the most common comparative claims in the predicate corpus.
_WB_INDICATOR_MAP: Dict[str, str] = {
    "gdp per capita":                   "NY.GDP.PCAP.CD",
    "gdp growth":                       "NY.GDP.MKTP.KD.ZG",
    "life expectancy":                  "SP.DYN.LE00.IN",
    "infant mortality":                 "SP.DYN.IMRT.IN",
    "co2 emissions per capita":         "EN.ATM.CO2E.PC",
    "renewable energy":                 "EG.FEC.RNEW.ZS",
    "population":                       "SP.POP.TOTL",
    "literacy rate":                    "SE.ADT.LITR.ZS",
    "unemployment":                     "SL.UEM.TOTL.ZS",
    "poverty headcount":                "SI.POV.DDAY",
    "electricity access":               "EG.ELC.ACCS.ZS",
    "maternal mortality":               "SH.STA.MMRT",
    "forest area":                      "AG.LND.FRST.ZS",
    "internet users":                   "IT.NET.USER.ZS",
    "military expenditure":             "MS.MIL.XPND.GD.ZS",
    "research development":             "GB.XPD.RSDV.GD.ZS",
    "nuclear energy":                   "EG.ELC.NUCL.ZS",
    "coal electricity":                 "EG.ELC.COAL.ZS",
    "death rate": "SP.DYN.CDRT.IN",
}

# Curated keyword → OECD dataflow ID mapping.
_OECD_DATAFLOW_MAP: Dict[str, str] = {
    "health expenditure":  "SHA",
    "pisa":                "PISA",
    "tax revenue":         "REV",
    "gender wage gap":     "EARNINGS",
    "social spending":     "SOCX_AGG",
    "air pollution":       "AIR_EMISSIONS",
    "road accidents":      "IRTAD",
    "trade":               "TIVA_2021_C1",
}

_WB_API = "https://api.worldbank.org/v2"
_OECD_API = "https://stats.oecd.org/SDMX-JSON/data"


def _wb_lookup(indicator: str, country: str = "WLD") -> List[Dict]:
    """Fetch the 5 most recent values for a World Bank indicator."""
    try:
        url = f"{_WB_API}/country/{country}/indicator/{indicator}"
        resp = _SESSION.get(
            url,
            params={"format": "json", "per_page": 5, "mrv": 5},
            timeout=_TOOL_TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        return [
            {
                "year": entry.get("date"),
                "value": entry.get("value"),
                "country": (entry.get("country") or {}).get("value", country),
                "indicator": indicator,
            }
            for entry in payload[1] or []
            if entry.get("value") is not None
        ]
    except Exception as exc:
        logger.warning("_wb_lookup error for %s: %s", indicator, exc)
        return []


def _oecd_lookup(dataflow: str) -> List[Dict]:
    """Fetch a compact slice of an OECD dataflow (latest 3 obs per member)."""
    try:
        url = f"{_OECD_API}/{dataflow}/all/all"
        resp = _SESSION.get(
            url,
            params={"contentType": "application/json", "lastNObservations": 1},
            timeout=_TOOL_TIMEOUT_S,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        # SDMX-JSON structure is deep; extract first few obs as flat dicts
        obs_map = (
            data.get("dataSets", [{}])[0]
            .get("series", {})
        )
        out = []
        for key, series_data in list(obs_map.items())[:5]:
            for period, obs in (series_data.get("observations") or {}).items():
                out.append({
                    "series_key": key,
                    "period": period,
                    "value": obs[0] if obs else None,
                    "dataflow": dataflow,
                })
        return out
    except Exception as exc:
        logger.warning("_oecd_lookup error for %s: %s", dataflow, exc)
        return []


def _statistical_search(query: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "source": "statistical_search",
        "world_bank": [],
        "oecd": [],
        "web_snippets": [],
        "status": "ok",
    }
    q_lower = query.lower()

    # Pass 1: World Bank
    wb_indicator: Optional[str] = None
    for kw, code in _WB_INDICATOR_MAP.items():
        if kw in q_lower:
            wb_indicator = code
            break
    if wb_indicator:
        results["world_bank"] = _wb_lookup(wb_indicator)

    # Pass 2: OECD
    oecd_flow: Optional[str] = None
    for kw, flow in _OECD_DATAFLOW_MAP.items():
        if kw in q_lower:
            oecd_flow = flow
            break
    if oecd_flow:
        results["oecd"] = _oecd_lookup(oecd_flow)

    # Pass 3: DDG fallback (always run as corroboration)
    snippets = _ddgs_search(f"{query} statistics data", max_results=3)
    results["web_snippets"] = snippets

    if (
        not results["world_bank"]
        and not results["oecd"]
        and not results["web_snippets"]
    ):
        results["status"] = "empty"
        results["note"] = "No statistical data found in World Bank, OECD, or web search."

    return results


# ---------------------------------------------------------------------------
# NEW TOOL 4: domain_store_search  (STUB — TRM v2 shim)
# ---------------------------------------------------------------------------
# This stub will be replaced by a real implementation when TRM v2 SQLite
# shards are built (mycelium/domain_store/retrieval.py).  Until then it:
#   - Logs a warning so the gap is visible in traces
#   - Returns a graceful empty response with a 'stub' status flag
#   - Allows the planner to route to it freely; the caller (run_workflow)
#     must check result["status"] == "stub" and fall back to web_search.
# ---------------------------------------------------------------------------

def _domain_store_search(query: str) -> Dict[str, Any]:
    logger.warning(
        "domain_store_search called but TRM v2 shards not yet built. "
        "Returning stub response. Query: %s",
        query,
    )
    return {
        "chunks": [],
        "status": "stub",
        "note": (
            "domain_store_search is a stub until TRM v2 SQLite shards are built. "
            "Caller should fall back to web_search or academic_search."
        ),
        "source": "domain_store_search",
        "query": query,
    }


# ---------------------------------------------------------------------------
# Tool dispatch table
# ---------------------------------------------------------------------------

_TOOL_FN: Dict[str, Any] = {
    "web_search":          _web_search,
    "academic_search":     _academic_search,
    "knowledge_base":      _knowledge_base,
    "calculator":          _calculator,
    "temporal_search":     _temporal_search,
    "causal_search":       _causal_search,
    "statistical_search":  _statistical_search,
    "domain_store_search": _domain_store_search,
}


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

server = Server("mycelium-tools")


@server.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    """Derive tool list from TOOL_REGISTRY so it stays in sync automatically."""
    tools = []
    for spec in TOOL_REGISTRY:
        if spec.input_arg == "expression":
            input_schema = {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A Python-compatible math expression, e.g. 'sqrt(2) * pi'"
                        ),
                    }
                },
                "required": ["expression"],
            }
        else:
            input_schema = {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            }
        tools.append(
            mcp_types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=input_schema,
            )
        )
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
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
            streams[0], streams[1], server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
