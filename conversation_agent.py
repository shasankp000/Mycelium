"""conversation_agent.py  (Milestone 4)

Conversational agent that explains a Mycelium run in plain language.
Now enriched with sandbox evidence when available.
"""
from __future__ import annotations

import json
from typing import Optional

from llm_providers import LLMClient

_SYSTEM_PROMPT = """\
You are Mycelium's reasoning assistant. Your job is to explain, in plain
language, what Mycelium found when it processed the user's query.

Rules:
- Be concise but complete. 2-4 paragraphs is ideal.
- Only use facts from the pipeline summary and sandbox evidence provided.
  Do not invent new facts.
- Explain the Layer 0 route, what domain(s) were selected, which expert(s)
  handled the query, and what the final validation result was.
- If sandbox evidence is present, summarise the key findings and cite the
  tool that produced them (e.g. "Semantic Scholar found...",
  "Wikidata reports...").
- If sandbox steps ran but all failed, state that external evidence could
  not be retrieved and flag INSUFFICIENT_EVIDENCE.
- If latency data is available, briefly mention the slowest phase.
- If the pipeline decision was CREATE_NEW_PATCH, explain that no trained
  domain expert exists yet for this query, so Mycelium relied on its
  general reasoning path and sandbox research.
- Write in second person: address the user directly.
- Do not mention internal implementation details (class names, JSON keys,
  etc.).
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
        """Generate a plain-language explanation.

        Parameters
        ----------
        user_query:
            The original user question.
        run:
            A MyceliumRunSummary (or compatible dict/object).
        sandbox_result:
            Optional SandboxResult.  When provided, its steps and summary
            are appended to the LLM context so the agent can reference
            real external evidence.
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

        # --- sandbox evidence ---
        if sandbox_result is not None:
            try:
                sr_dict = sandbox_result.model_dump()  # type: ignore[attr-defined]
            except AttributeError:
                sr_dict = dict(sandbox_result) if not isinstance(sandbox_result, dict) else sandbox_result  # type: ignore

            lines.append("")
            lines.append("=== Sandbox Evidence ===")
            steps = sr_dict.get("steps", []) or []
            if not steps:
                lines.append("No sandbox steps were executed.")
            else:
                ok_steps = [s for s in steps if s.get("status") == "ok"]
                err_steps = [s for s in steps if s.get("status") != "ok"]
                lines.append(
                    f"Tool calls: {len(steps)} total, "
                    f"{len(ok_steps)} succeeded, {len(err_steps)} failed."
                )
                for i, step in enumerate(ok_steps[:4], 1):  # cap at 4 for prompt budget
                    tool = step.get("tool", "unknown")
                    inp = (step.get("input") or {}).get("query", "")
                    out = step.get("output", {})
                    # Produce a compact representation of the output
                    if "results" in out:
                        snippets = [
                            r.get("snippet") or r.get("abstract") or ""
                            for r in (out["results"][:2])
                        ]
                        out_str = " | ".join(s[:200] for s in snippets if s)
                    elif "papers" in out:
                        papers = out["papers"][:2]
                        out_str = "; ".join(
                            f"{p.get('title','')} ({p.get('year','?')})"
                            for p in papers
                        )
                    elif "entities" in out:
                        entities = out["entities"][:2]
                        out_str = "; ".join(
                            f"{e.get('label','')}: {e.get('description','')}"
                            for e in entities
                        )
                    elif "result" in out:
                        out_str = str(out["result"])
                    else:
                        out_str = json.dumps(out, ensure_ascii=False)[:200]
                    lines.append(f"  [{i}] {tool}('{inp}'): {out_str}")

            summary = sr_dict.get("summary", "")
            if summary:
                lines.append(f"\nSandbox summary: {summary}")

        prompt = "\n".join(lines)
        return self._llm.generate(prompt, system=_SYSTEM_PROMPT)


_agent: Optional[ConversationAgent] = None


def get_conversation_agent() -> ConversationAgent:
    """Return a module-level singleton ConversationAgent."""
    global _agent
    if _agent is None:
        _agent = ConversationAgent()
    return _agent
