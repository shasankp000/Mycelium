"""conversation_agent.py  (Milestone 5 + Lexis Phase 2)

Conversational agent that explains a Mycelium run in plain language.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Optional

from mycelium.pipeline.llm_providers import LLMClient

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are Mycelium — an experimental multi-expert AI reasoning system.
You are speaking directly to the user in first person.
Never refer to yourself in third person. Never say "Mycelium did X" or
"the system did X". Always say "I did X", "my router", "my expert", etc.

You will receive:
  1. A pipeline summary of what I (Mycelium) just did to process the query.
  2. Raw sandbox tool results from my external research tools.

Your response must:
- Be 2-4 paragraphs. Concise but complete.
- First paragraph: explain what I did — how I routed the query, which
  domains I identified, which expert I engaged, and what my validation
  found.
- Second paragraph (if sandbox ran): synthesise the key findings from my
  tool results. Cite the tool that produced each fact.
  If all my tool calls returned empty or failed, say clearly that I
  couldn't find external evidence right now and flag INSUFFICIENT_EVIDENCE.
  Own this as my limitation — do not say an external system failed.
- If my expert decision was CREATE_NEW_PATCH, explain that I don't have a
  trained domain expert for this topic yet, so I used my general reasoning
  path and sandbox research to compensate.
- If latency data is available, briefly mention the slowest phase in my
  pipeline.
- Only use facts from the pipeline summary and tool results. Do not invent
  facts.
- Write in second person when addressing the user, first person when
  describing yourself.
- Do not mention internal implementation details.
"""


def _try_condense_history(turns: list[dict]) -> str:
    try:
        from lexis_condenser import condense_history
        return condense_history(turns, fallback_on_error=True)
    except Exception as exc:
        log.debug("lexis_condenser unavailable (%s); using raw history.", exc)
        return "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')}"
            for t in turns
        )


class ConversationAgent:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self._llm = llm or LLMClient()

    def _build_prompt(
        self,
        user_query: str,
        run: object,
        sandbox_result: Optional[object] = None,
        condensed_history: Optional[str] = None,
    ) -> str:
        try:
            run_dict = run.model_dump()
        except AttributeError:
            run_dict = dict(run) if not isinstance(run, dict) else run

        lines: list[str] = []

        if condensed_history:
            lines.append("=== Conversation Context (condensed) ===")
            lines.append(condensed_history)
            lines.append("")

        lines.append(f"User query: {user_query}")
        lines.append("")
        lines.append("=== Pipeline Summary ===")
        lines.append(f"Layer 0 route: {run_dict.get('layer0', {}).get('route', 'n/a')}")
        routing = run_dict.get("routing", {})
        lines.append(f"Routing classification: {routing.get('classification', 'n/a')}")
        lines.append(f"Selected domains: {', '.join(routing.get('selected_domains', []) or ['n/a'])}")
        exp = run_dict.get("expert_decision", {})
        lines.append(f"Expert decision type: {exp.get('decision_type', 'n/a')}")
        lines.append(f"Selected experts: {', '.join(exp.get('selected_experts', []) or ['n/a'])}")
        conf = exp.get("expert_confidence")
        if conf is not None:
            lines.append(f"Expert confidence: {float(conf):.2%}")
        phase3 = run_dict.get("phase3", {})
        vd = phase3.get("validation_decision", {})
        lines.append(f"Validation result: {vd.get('result_class', 'n/a')}")
        latencies = phase3.get("phase_latencies_ms", {})
        if latencies:
            slowest = max(latencies.items(), key=lambda kv: kv[1])
            lines.append(f"Slowest phase: {slowest[0]} ({slowest[1]} ms)")

        if sandbox_result is not None:
            try:
                sr_dict = sandbox_result.model_dump()
            except AttributeError:
                sr_dict = dict(sandbox_result) if not isinstance(sandbox_result, dict) else sandbox_result

            steps = sr_dict.get("steps", []) or []
            lines.append("")
            lines.append("=== Sandbox Tool Results ===")

            if not steps:
                lines.append("No sandbox steps were executed.")
            else:
                ok_steps = [s for s in steps if s.get("status") == "ok"]
                err_steps = [s for s in steps if s.get("status") not in ("ok", "empty")]
                empty_steps = [s for s in steps if s.get("status") == "empty"]
                lines.append(f"{len(steps)} tool call(s) total: {len(ok_steps)} succeeded, {len(empty_steps)} returned no results, {len(err_steps)} failed.")
                for i, step in enumerate(steps[:5], 1):
                    tool = step.get("tool", "unknown")
                    inp = (step.get("input") or {}).get("query", "")
                    status = step.get("status", "unknown")
                    out = step.get("output", {})
                    if status == "empty":
                        out_str = f"returned no results — {out.get('note', 'all backends exhausted')}"
                    elif status != "ok":
                        out_str = f"FAILED — {out.get('error', 'unknown error')}"
                    elif "results" in out:
                        snippets = [r.get("snippet") or r.get("body") or "" for r in (out["results"][:3])]
                        out_str = " | ".join(s[:250] for s in snippets if s) or "(empty results list)"
                    elif "papers" in out:
                        papers = out["papers"][:3]
                        out_str = "; ".join(f"{p.get('title', '')} ({p.get('year', '?')}): {(p.get('abstract') or '')[:150]}" for p in papers) or "(no papers found)"
                    elif "entities" in out:
                        entities = out["entities"][:3]
                        out_str = "; ".join(f"{e.get('label', '')}: {e.get('description', '')}" for e in entities) or "(no entities found)"
                    elif "result" in out:
                        out_str = str(out["result"])
                    elif "error" in out:
                        out_str = f"ERROR — {out['error']}"
                    else:
                        out_str = json.dumps(out, ensure_ascii=False)[:300]
                    lines.append(f"  [{i}] {tool}('{inp}') [{status}]: {out_str}")

        return "\n".join(lines)

    def answer(
        self,
        user_query: str,
        run: object,
        sandbox_result: Optional[object] = None,
        stream: bool = False,
        history: Optional[list[dict]] = None,
    ) -> str:
        if stream:
            import asyncio
            import concurrent.futures

            async def _collect() -> str:
                chunks: list[str] = []
                async for token in self.stream_answer(user_query, run, sandbox_result, history=history):
                    chunks.append(token)
                return "".join(chunks)

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(asyncio.run, _collect())
                    return future.result()
            else:
                return asyncio.run(_collect())

        condensed_history: Optional[str] = None
        if history:
            condensed_history = _try_condense_history(history)
            log.debug("History condensed: %d chars -> %d chars", sum(len(t.get("content", "")) for t in history), len(condensed_history))

        prompt = self._build_prompt(user_query, run, sandbox_result, condensed_history)

        try:
            return self._llm.generate(prompt, system=_SYSTEM_PROMPT)
        except Exception as exc:
            try:
                run_dict = run.model_dump()
            except AttributeError:
                run_dict = dict(run) if not isinstance(run, dict) else run

            fallback_lines = [
                f"I processed your query but my language model is currently unavailable ({exc}).",
                "",
                "Here is what I found in my pipeline:",
            ]
            route = run_dict.get("layer0", {}).get("route", "n/a")
            cls_ = run_dict.get("routing", {}).get("classification", "n/a")
            domains = ", ".join(run_dict.get("routing", {}).get("selected_domains", []) or ["n/a"])
            dec = run_dict.get("expert_decision", {}).get("decision_type", "n/a")
            experts = ", ".join(run_dict.get("expert_decision", {}).get("selected_experts", []) or ["n/a"])
            vr = run_dict.get("phase3", {}).get("validation_decision", {}).get("result_class", "n/a")
            fallback_lines += [
                f"  Route: {route}  |  Classification: {cls_}  |  Domains: {domains}",
                f"  Expert decision: {dec}  |  Experts: {experts}  |  Validation: {vr}",
            ]
            if sandbox_result is not None:
                try:
                    sr_dict2 = sandbox_result.model_dump()
                except AttributeError:
                    sr_dict2 = dict(sandbox_result) if not isinstance(sandbox_result, dict) else sandbox_result
                steps2 = sr_dict2.get("steps", []) or []
                if steps2:
                    fallback_lines.append("")
                    fallback_lines.append("My sandbox tool results (raw):")
                    for s in steps2:
                        tool = s.get("tool", "?")
                        status = s.get("status", "?")
                        out = s.get("output", {})
                        if status == "empty":
                            preview = out.get("note", "no results returned")
                        elif "results" in out:
                            preview = " | ".join((r.get("snippet") or r.get("body") or "")[:120] for r in out["results"][:2] if r.get("snippet") or r.get("body")) or "(empty)"
                        elif "entities" in out:
                            preview = "; ".join(f"{e.get('label', '')}: {e.get('description', '')}" for e in out["entities"][:2]) or "(empty)"
                        elif "papers" in out:
                            preview = "; ".join(f"{p.get('title', '')} ({p.get('year', '?')})" for p in out["papers"][:2]) or "(empty)"
                        elif "error" in out:
                            preview = f"error: {out['error']}"
                        else:
                            preview = json.dumps(out, ensure_ascii=False)[:150]
                        fallback_lines.append(f"  {tool} [{status}]: {preview}")
            return "\n".join(fallback_lines)

    async def stream_answer(
        self,
        user_query: str,
        run: object,
        sandbox_result: Optional[object] = None,
        history: Optional[list[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        condensed_history: Optional[str] = None
        if history:
            condensed_history = _try_condense_history(history)
            log.debug("History condensed (stream): %d chars -> %d chars", sum(len(t.get("content", "")) for t in history), len(condensed_history))

        prompt = self._build_prompt(user_query, run, sandbox_result, condensed_history)

        try:
            async for token in self._llm.stream(prompt, system=_SYSTEM_PROMPT):
                yield token
        except (AttributeError, NotImplementedError):
            try:
                result = self._llm.generate(prompt, system=_SYSTEM_PROMPT)
                yield result
            except Exception as exc:
                try:
                    run_dict = run.model_dump()
                except AttributeError:
                    run_dict = dict(run) if not isinstance(run, dict) else run
                yield (f"I processed your query but my language model is currently unavailable ({exc}). Route: {run_dict.get('layer0', {}).get('route', 'n/a')}.")
        except Exception as exc:
            try:
                run_dict = run.model_dump()
            except AttributeError:
                run_dict = dict(run) if not isinstance(run, dict) else run
            yield (f"I processed your query but streaming failed ({exc}). Route: {run_dict.get('layer0', {}).get('route', 'n/a')}.")


_agent: Optional[ConversationAgent] = None


def get_conversation_agent() -> ConversationAgent:
    global _agent
    if _agent is None:
        _agent = ConversationAgent()
    return _agent
