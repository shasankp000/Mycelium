"""sandbox_manager.py

Milestone 3: real LLM-backed sandbox orchestration.

The SandboxManager drives a tool-calling loop:
  1. Ask the LLM to produce a JSON plan of tool calls based on the task.
  2. Execute each tool call (web search, academic search, knowledge-base,
     calculator) and record a SandboxStep.
  3. Ask the LLM to synthesise a final summary from all steps.

Tool adapters are intentionally lightweight — they use only stdlib + the
packages already present in the repo (requests).  Heavy optional deps
(e.g. the `ddgs` package for DuckDuckGo) are imported lazily so the
server still boots if they are absent.
"""
from __future__ import annotations

import json
import logging
import math
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from llm_providers import LLMClient
from sandbox_models import SandboxResult, SandboxStep, SandboxTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------
_DEFAULT_MAX_STEPS = 6
_DEFAULT_WALL_TIMEOUT_S = 30.0  # seconds for the whole sandbox run
_TOOL_TIMEOUT_S = 8.0            # per individual tool call

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _web_search(query: str) -> Dict[str, Any]:
    """DuckDuckGo instant-answer search via ddgs (optional dep)."""
    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        snippets = [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ]
        return {"results": snippets, "source": "duckduckgo"}
    except ImportError:
        # Fallback: DuckDuckGo HTML scrape (no key needed)
        try:
            resp = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=_TOOL_TIMEOUT_S,
            )
            resp.raise_for_status()
            # Very basic extraction — good enough for PoC
            import re
            snippets_raw = re.findall(r'class="result__snippet">(.*?)</a>', resp.text, re.DOTALL)
            titles_raw   = re.findall(r'class="result__a"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
            results_list = [
                {"title": re.sub(r"<[^>]+>", "", t).strip(),
                 "snippet": re.sub(r"<[^>]+>", "", s).strip()}
                for t, s in zip(titles_raw[:4], snippets_raw[:4])
            ]
            return {"results": results_list, "source": "duckduckgo_html"}
        except Exception as exc:
            return {"error": str(exc), "source": "web_search"}
    except Exception as exc:
        return {"error": str(exc), "source": "web_search"}


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
        papers = []
        for p in data.get("data", []):
            papers.append({
                "title": p.get("title", ""),
                "year": p.get("year"),
                "authors": [a.get("name", "") for a in (p.get("authors") or [])[:3]],
                "abstract": (p.get("abstract") or "")[:300],
            })
        return {"papers": papers, "source": "semantic_scholar"}
    except Exception as exc:
        return {"error": str(exc), "source": "academic_search"}


def _knowledge_base(query: str) -> Dict[str, Any]:
    """Wikidata SPARQL — entity lookup by label."""
    try:
        sparql = f"""
SELECT ?item ?itemLabel ?description WHERE {{
  SERVICE wikibase:mwapi {{
    bd:serviceParam wikibase:endpoint "www.wikidata.org";
                    wikibase:api "EntitySearch";
                    mwapi:search "{query}";
                    mwapi:language "en".
    ?item wikibase:apiOutputItem mwapi:item.
  }}
  OPTIONAL {{ ?item schema:description ?description FILTER(LANG(?description) = "en") }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 4
"""
        resp = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": sparql, "format": "json"},
            headers={"Accept": "application/sparql-results+json",
                     "User-Agent": "MyceliumPoC/0.3"},
            timeout=_TOOL_TIMEOUT_S,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        entities = [
            {
                "label": b.get("itemLabel", {}).get("value", ""),
                "description": b.get("description", {}).get("value", ""),
                "id": b.get("item", {}).get("value", "").split("/")[-1],
            }
            for b in bindings
        ]
        return {"entities": entities, "source": "wikidata"}
    except Exception as exc:
        return {"error": str(exc), "source": "knowledge_base"}


def _calculator(expression: str) -> Dict[str, Any]:
    """Safe arithmetic evaluator."""
    allowed_names: Dict[str, Any] = {
        k: getattr(math, k) for k in dir(math) if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return {"result": result, "expression": expression, "source": "calculator"}
    except Exception as exc:
        return {"error": str(exc), "expression": expression, "source": "calculator"}


_TOOL_REGISTRY: Dict[str, Any] = {
    "web_search": _web_search,
    "academic_search": _academic_search,
    "knowledge_base": _knowledge_base,
    "calculator": _calculator,
}

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
You are the Mycelium sandbox orchestrator. Given a user query and optional hypotheses,
produce a JSON array of tool calls to gather evidence. Each element must be:
  {"tool": "<tool_name>", "input": "<query string>", "commentary": "<why this call>"}

Available tools: web_search, academic_search, knowledge_base, calculator.
- web_search: general web search, good for recent news and factual claims.
- academic_search: peer-reviewed papers via Semantic Scholar.
- knowledge_base: structured entity facts via Wikidata.
- calculator: safe arithmetic / unit conversion expressions only.

Rules:
- Return ONLY a valid JSON array. No markdown, no explanation outside the array.
- Maximum {max_steps} tool calls.
- Prefer academic_search for scientific/medical claims.
- Use calculator only for numeric expressions.
- Diversify tools when the query spans multiple consequence types.
"""

_PLAN_USER_TMPL = """\
User query: {query}

Hypotheses / context:
{hypotheses}

Domains: {domains}

Produce the tool-call plan now.
"""

_SUMMARY_SYSTEM = """\
You are the Mycelium sandbox summariser. Given a user query and a list of tool results,
write a concise evidence summary (3-5 sentences). Rules:
- Only reference facts actually present in the tool results.
- Note source names (e.g. "Semantic Scholar", "Wikidata").
- Flag if evidence is sparse or contradictory.
- Do not invent facts.
"""

_SUMMARY_USER_TMPL = """\
User query: {query}

Tool results:
{results_block}

Write the evidence summary now.
"""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SandboxManager:
    """LLM-backed sandbox that orchestrates real tool calls for each run."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        max_steps: int = _DEFAULT_MAX_STEPS,
        wall_timeout_s: float = _DEFAULT_WALL_TIMEOUT_S,
    ) -> None:
        self._llm = llm or LLMClient()
        self._max_steps = max_steps
        self._wall_timeout_s = wall_timeout_s

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, task: SandboxTask) -> SandboxResult:
        wall_start = time.monotonic()
        started = datetime.utcnow()
        steps: List[SandboxStep] = []

        plan = self._make_plan(task)
        if not plan:
            return self._stub_result(task, started)

        for call in plan[: self._max_steps]:
            if time.monotonic() - wall_start > self._wall_timeout_s * 0.85:
                logger.warning("SandboxManager: wall timeout approaching, stopping early")
                break

            tool_name = call.get("tool", "")
            tool_input = call.get("input", "")
            commentary = call.get("commentary", "")

            step = self._execute_step(tool_name, tool_input, commentary)
            steps.append(step)

        summary = self._synthesise_summary(task.user_query, steps)
        finished = datetime.utcnow()

        return SandboxResult(
            trace_id=task.trace_id,
            started_at=started,
            finished_at=finished,
            steps=steps,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_plan(self, task: SandboxTask) -> List[Dict[str, Any]]:
        hypotheses_block = "\n".join(f"- {h}" for h in task.hypotheses) or "(none)"
        domains_str = ", ".join(task.domains) or "general"
        system = _PLAN_SYSTEM.format(max_steps=self._max_steps)
        user_msg = _PLAN_USER_TMPL.format(
            query=task.user_query,
            hypotheses=hypotheses_block,
            domains=domains_str,
        )
        try:
            raw = self._llm.generate(user_msg, system=system)
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            plan = json.loads(raw)
            if isinstance(plan, list):
                return plan
            logger.warning("SandboxManager: LLM plan was not a list: %s", raw[:200])
            return []
        except Exception as exc:
            logger.warning("SandboxManager: plan generation failed — %s", exc)
            return []

    def _execute_step(
        self, tool_name: str, tool_input: str, commentary: str
    ) -> SandboxStep:
        fn = _TOOL_REGISTRY.get(tool_name)
        step_start = datetime.utcnow()
        if fn is None:
            output: Dict[str, Any] = {"error": f"Unknown tool: {tool_name}"}
            status = "error"
        else:
            try:
                output = fn(tool_input)
                status = "error" if "error" in output else "ok"
            except Exception as exc:
                output = {"error": str(exc)}
                status = "error"
                logger.warning("SandboxManager: tool '%s' raised — %s", tool_name, exc)

        step_end = datetime.utcnow()
        return SandboxStep(
            tool=tool_name,
            input={"query": tool_input},
            output=output,
            commentary=commentary,
            status=status,
            started_at=step_start,
            finished_at=step_end,
        )

    def _synthesise_summary(self, query: str, steps: List[SandboxStep]) -> str:
        if not steps:
            return "No tool calls were executed; no evidence was gathered."

        results_lines: List[str] = []
        for i, s in enumerate(steps, 1):
            out_str = json.dumps(s.output, ensure_ascii=False)[:600]
            results_lines.append(
                f"[{i}] tool={s.tool} status={s.status}\n    input={s.input.get('query','')}\n    output={out_str}"
            )
        results_block = "\n\n".join(results_lines)

        user_msg = _SUMMARY_USER_TMPL.format(
            query=query, results_block=results_block
        )
        try:
            return self._llm.generate(user_msg, system=_SUMMARY_SYSTEM)
        except Exception as exc:
            logger.warning("SandboxManager: summary LLM call failed — %s", exc)
            ok = sum(1 for s in steps if s.status == "ok")
            return (
                f"Sandbox executed {len(steps)} tool call(s), {ok} succeeded. "
                "LLM summary unavailable; see individual steps for raw evidence."
            )

    @staticmethod
    def _stub_result(task: SandboxTask, started: datetime) -> SandboxResult:
        return SandboxResult(
            trace_id=task.trace_id,
            started_at=started,
            finished_at=datetime.utcnow(),
            steps=[],
            summary=(
                "Sandbox plan generation failed (LLM unreachable or returned "
                "invalid JSON). No evidence was gathered for this trace."
            ),
        )


# Process-level singleton
_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    global _manager
    if _manager is None:
        _manager = SandboxManager()
    return _manager
