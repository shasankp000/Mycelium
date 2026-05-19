"""conversation_agent.py  (Milestone 5)

Conversational agent that explains a Mycelium run in plain language.

Changes (2026-05-19 — M5 patch)
---------------------------------
- M5 §8.2: answer() gains a ``stream: bool = False`` parameter.
  When stream=True the call is dispatched to stream_answer() so existing
  callers are unaffected while the broadcast API's SSE endpoint can use
  the new async generator path.
- New async def stream_answer() returns AsyncGenerator[str, None] backed
  by _llm.stream().  Falls back gracefully to a single-chunk yield via
  _llm.generate() if the LLMClient does not implement .stream().

Changes (2026-05-14 — patch 2)
--------------------------------
- _SYSTEM_PROMPT rewritten: agent speaks in first person as Mycelium.
- prompt-builder: 'empty' status branch for clean no-results lines.
- Fallback string updated to first-person voice.

Changes (2026-05-14 — patch 1)
--------------------------------
- Single LLM round-trip per request.
- Graceful degradation with readable fallback on LLM failure.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator, Optional

from llm_providers import LLMClient

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
  tool results. Cite the tool that produced each fact (e.g. "my web search
  found...", "Wikidata returned...", "Semantic Scholar found...").
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
- Do not mention internal implementation details (class names, JSON keys,
  file names, etc.).
"""


class ConversationAgent:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self._llm = llm or LLMClient()

    # ------------------------------------------------------------------
    # Internal: build the prompt shared by answer() and stream_answer()
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        user_query: str,
        run: object,
        sandbox_result: Optional[object] = None,
    ) -> str:
        """Assemble the prompt string from pipeline + sandbox data."""
        try:
            run_dict = run.model_dump()  # type: ignore[attr-defined]
        except AttributeError:
            run_dict = dict(run) if not isinstance(run, dict) else run  # type: ignore

        lines = [f"User query: {user_query}", ""]

        # --- pipeline summary ---
        lines.append("=== Pipeline Summary ===")
        lines.append(f"Layer 0 route: {run_dict.get('layer0', {}).get('route', 'n/a')}")
        routing = run_dict.get("routing", {})
        lines.append(f"Routing classification: {routing.get('classification', 'n/a')}")
        lines.append(
            f"Selected domains: {', '.join(routing.get('selected_domains', []) or ['n/a'])}"
        )
        exp = run_dict.get("expert_decision", {})
        lines.append(f"Expert decision type: {exp.get('decision_type', 'n/a')}")
        lines.append(
            f"Selected experts: {', '.join(exp.get('selected_experts', []) or ['n/a'])}"
        )
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

        # --- sandbox tool results ---
        if sandbox_result is not None:
            try:
                sr_dict = sandbox_result.model_dump()  # type: ignore[attr-defined]
            except AttributeError:
                sr_dict = dict(sandbox_result) if not isinstance(sandbox_result, dict) else sandbox_result  # type: ignore

            steps = sr_dict.get("steps", []) or []
            lines.append("")
            lines.append("=== Sandbox Tool Results ===")

            if not steps:
                lines.append("No sandbox steps were executed.")
            else:
                ok_steps = [s for s in steps if s.get("status") == "ok"]
                err_steps = [s for s in steps if s.get("status") not in ("ok", "empty")]
                empty_steps = [s for s in steps if s.get("status") == "empty"]
                lines.append(
                    f"{len(steps)} tool call(s) total: "
                    f"{len(ok_steps)} succeeded, {len(empty_steps)} returned no results, "
                    f"{len(err_steps)} failed."
                )
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
                        snippets = [
                            r.get("snippet") or r.get("body") or ""
                            for r in (out["results"][:3])
                        ]
                        out_str = " | ".join(s[:250] for s in snippets if s) or "(empty results list)"
                    elif "papers" in out:
                        papers = out["papers"][:3]
                        out_str = "; ".join(
                            f"{p.get('title', '')} ({p.get('year', '?')}): {(p.get('abstract') or '')[:150]}"
                            for p in papers
                        ) or "(no papers found)"
                    elif "entities" in out:
                        entities = out["entities"][:3]
                        out_str = "; ".join(
                            f"{e.get('label', '')}: {e.get('description', '')}"
                            for e in entities
                        ) or "(no entities found)"
                    elif "result" in out:
                        out_str = str(out["result"])
                    elif "error" in out:
                        out_str = f"ERROR — {out['error']}"
                    else:
                        out_str = json.dumps(out, ensure_ascii=False)[:300]

                    lines.append(f"  [{i}] {tool}('{inp}') [{status}]: {out_str}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Synchronous answer (unchanged public API)
    # ------------------------------------------------------------------
    def answer(
        self,
        user_query: str,
        run: object,
        sandbox_result: Optional[object] = None,
        stream: bool = False,
    ) -> str:
        """Generate a plain-language explanation in a single LLM call.

        Parameters
        ----------
        user_query:
            The original user question.
        run:
            A MyceliumRunSummary (or compatible dict/object).
        sandbox_result:
            Optional SandboxResult.  When provided, its raw steps are
            included in the prompt so the agent synthesises evidence AND
            explanation together — one Ollama round-trip total.
        stream:
            M5 §8.2 — when True, dispatches internally to stream_answer()
            and collects all tokens before returning so existing callers
            that expect a str are unaffected.  New callers that want true
            token-by-token streaming should call stream_answer() directly.
        """
        if stream:
            import asyncio
            import concurrent.futures

            async def _collect() -> str:
                chunks: list[str] = []
                async for token in self.stream_answer(user_query, run, sandbox_result):
                    chunks.append(token)
                return "".join(chunks)

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already inside an async context — run in a sibling thread
                # so we can call asyncio.run() without nesting event loops.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(asyncio.run, _collect())
                    return future.result()
            else:
                return asyncio.run(_collect())

        prompt = self._build_prompt(user_query, run, sandbox_result)

        try:
            return self._llm.generate(prompt, system=_SYSTEM_PROMPT)
        except Exception as exc:
            # Graceful fallback — build a readable first-person response from
            # raw data so the frontend is never left with an opaque error.
            try:
                run_dict = run.model_dump()  # type: ignore[attr-defined]
            except AttributeError:
                run_dict = dict(run) if not isinstance(run, dict) else run  # type: ignore

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
                    sr_dict2 = sandbox_result.model_dump()  # type: ignore[attr-defined]
                except AttributeError:
                    sr_dict2 = dict(sandbox_result) if not isinstance(sandbox_result, dict) else sandbox_result  # type: ignore
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
                            preview = " | ".join(
                                (r.get("snippet") or r.get("body") or "")[:120]
                                for r in out["results"][:2]
                                if r.get("snippet") or r.get("body")
                            ) or "(empty)"
                        elif "entities" in out:
                            preview = "; ".join(
                                f"{e.get('label', '')}: {e.get('description', '')}"
                                for e in out["entities"][:2]
                            ) or "(empty)"
                        elif "papers" in out:
                            preview = "; ".join(
                                f"{p.get('title', '')} ({p.get('year', '?')})"
                                for p in out["papers"][:2]
                            ) or "(empty)"
                        elif "error" in out:
                            preview = f"error: {out['error']}"
                        else:
                            preview = json.dumps(out, ensure_ascii=False)[:150]
                        fallback_lines.append(f"  {tool} [{status}]: {preview}")

            return "\n".join(fallback_lines)

    # ------------------------------------------------------------------
    # M5 §8.2 — Async streaming path
    # ------------------------------------------------------------------
    async def stream_answer(
        self,
        user_query: str,
        run: object,
        sandbox_result: Optional[object] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the conversational explanation token-by-token.

        M5 §8.2 — Returns an AsyncGenerator that yields response tokens as
        they arrive from the Ollama streaming API.  The broadcast API's SSE
        endpoint consumes this generator and forwards each token to the
        client, cutting perceived latency from O(total_tokens) to
        O(first_token).

        Falls back gracefully to a single-chunk yield via _llm.generate()
        when the LLMClient does not yet implement .stream().

        Usage::

            async for token in agent.stream_answer(query, run, sandbox_result):
                yield sse_event("token", token)
        """
        prompt = self._build_prompt(user_query, run, sandbox_result)

        try:
            async for token in self._llm.stream(prompt, system=_SYSTEM_PROMPT):
                yield token
        except (AttributeError, NotImplementedError):
            # LLMClient does not implement .stream() yet — fall back to a
            # single blocking generate() and yield the result as one chunk.
            try:
                result = self._llm.generate(prompt, system=_SYSTEM_PROMPT)
                yield result
            except Exception as exc:
                try:
                    run_dict = run.model_dump()  # type: ignore[attr-defined]
                except AttributeError:
                    run_dict = dict(run) if not isinstance(run, dict) else run  # type: ignore
                yield (
                    f"I processed your query but my language model is currently "
                    f"unavailable ({exc}). Route: "
                    f"{run_dict.get('layer0', {}).get('route', 'n/a')}."
                )
        except Exception as exc:
            try:
                run_dict = run.model_dump()  # type: ignore[attr-defined]
            except AttributeError:
                run_dict = dict(run) if not isinstance(run, dict) else run  # type: ignore
            yield (
                f"I processed your query but streaming failed ({exc}). "
                f"Route: {run_dict.get('layer0', {}).get('route', 'n/a')}."
            )


_agent: Optional[ConversationAgent] = None


def get_conversation_agent() -> ConversationAgent:
    """Return a module-level singleton ConversationAgent."""
    global _agent
    if _agent is None:
        _agent = ConversationAgent()
    return _agent
