"""conversation_agent.py  (Milestone 4)

Conversational agent that explains a Mycelium run in plain language.

Changes (2026-05-14)
--------------------
- Single LLM round-trip per request.  Previously SandboxManager called
  the LLM once for a 3-5 sentence evidence summary, then ConversationAgent
  called it again for the explanation — two cold-starts per query.
  Now SandboxManager returns summary="" and this agent receives the raw
  SandboxResult steps, synthesises the evidence AND produces the
  explanation in one prompt.
- Graceful degradation: if the LLM call fails, the fallback string now
  includes a readable summary of the raw tool results so the frontend
  always shows something useful.
"""
from __future__ import annotations

import json
from typing import Optional

from llm_providers import LLMClient

_SYSTEM_PROMPT = """\
You are Mycelium's reasoning assistant. Your job is to explain, in plain
language, what Mycelium found when it processed the user's query.

You will receive:
  1. A pipeline summary (Layer 0 route, routing, expert decision, validation).
  2. Raw sandbox tool results (if the sandbox ran).

Your response must:
- Be 2-4 paragraphs. Concise but complete.
- First paragraph: explain what Mycelium did — route, domains, experts, validation.
- Second paragraph (if sandbox ran): synthesise the key findings from the tool
  results. Cite the tool that produced each fact (e.g. "DuckDuckGo found...",
  "Wikidata reports...", "Semantic Scholar found...").
  If all tool calls returned empty or failed, state clearly that external
  evidence was unavailable and flag INSUFFICIENT_EVIDENCE.
- If the expert decision was CREATE_NEW_PATCH, explain that no trained domain
  expert exists yet for this query, so Mycelium used its general reasoning
  path and sandbox research.
- If latency data is available, briefly mention the slowest phase.
- Only use facts from the pipeline summary and tool results. Do not invent facts.
- Write in second person: address the user directly.
- Do not mention internal implementation details (class names, JSON keys, etc.).
"""


class ConversationAgent:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self._llm = llm or LLMClient()

    def answer(
        self,
        user_query: str,
        run: object,
        sandbox_result: Optional[object] = None,
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
        """
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

        # --- sandbox tool results (raw, for single-pass synthesis) ---
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
                err_steps = [s for s in steps if s.get("status") != "ok"]
                lines.append(
                    f"{len(steps)} tool call(s) total: "
                    f"{len(ok_steps)} succeeded, {len(err_steps)} failed."
                )
                for i, step in enumerate(steps[:5], 1):
                    tool = step.get("tool", "unknown")
                    inp = (step.get("input") or {}).get("query", "")
                    status = step.get("status", "unknown")
                    out = step.get("output", {})

                    if status != "ok":
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

        prompt = "\n".join(lines)

        try:
            return self._llm.generate(prompt, system=_SYSTEM_PROMPT)
        except Exception as exc:
            # Graceful fallback — build a readable response from raw data
            # so the frontend is never left with an opaque error message.
            fallback_lines = [
                f"Mycelium processed your query but the language model is currently unavailable ({exc}).",
                "",
                "Here is what the pipeline found:",
            ]

            # Pipeline facts
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

            # Sandbox raw step summaries
            if sandbox_result is not None:
                try:
                    sr_dict2 = sandbox_result.model_dump()  # type: ignore[attr-defined]
                except AttributeError:
                    sr_dict2 = dict(sandbox_result) if not isinstance(sandbox_result, dict) else sandbox_result  # type: ignore
                steps2 = sr_dict2.get("steps", []) or []
                if steps2:
                    fallback_lines.append("")
                    fallback_lines.append("Sandbox evidence (raw):")
                    for s in steps2:
                        tool = s.get("tool", "?")
                        status = s.get("status", "?")
                        out = s.get("output", {})
                        if "results" in out:
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


_agent: Optional[ConversationAgent] = None


def get_conversation_agent() -> ConversationAgent:
    """Return a module-level singleton ConversationAgent."""
    global _agent
    if _agent is None:
        _agent = ConversationAgent()
    return _agent
