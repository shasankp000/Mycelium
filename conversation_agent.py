"""Conversational agent that explains a Mycelium run in plain language."""
from __future__ import annotations

from typing import Optional

from llm_providers import LLMClient

_SYSTEM_PROMPT = """\
You are Mycelium's reasoning assistant. Your job is to explain, in plain language,
what Mycelium found when it processed the user's query.

Rules:
- Be concise but complete. 2-4 paragraphs is ideal.
- Only use facts from the pipeline summary provided. Do not invent new facts.
- Explain the Layer 0 route, what domain(s) were selected, which expert(s) handled
  the query, and what the final validation result was.
- If latency data is available, briefly mention the slowest phase.
- Write in second person: address the user directly.
- Do not mention internal implementation details (class names, JSON keys, etc.).
"""


class ConversationAgent:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self._llm = llm or LLMClient()

    def answer(self, user_query: str, run: object) -> str:
        """Generate a plain-language explanation of `run` for `user_query`.

        `run` is expected to be a MyceliumRunSummary or any object/dict with
        the same attributes; we serialise it to a text block and feed it to
        the LLM alongside the system prompt.
        """
        # Build a readable summary block from whatever run exposes
        try:
            # Pydantic model → dict
            run_dict = run.model_dump()  # type: ignore[attr-defined]
        except AttributeError:
            run_dict = dict(run) if not isinstance(run, dict) else run  # type: ignore[call-overload]

        # Flatten the most useful fields into a readable block
        lines = [f"User query: {user_query}", ""]
        lines.append(f"Layer 0 route: {run_dict.get('layer0', {}).get('route', 'n/a')}")
        routing = run_dict.get('routing', {})
        lines.append(f"Routing classification: {routing.get('classification', 'n/a')}")
        lines.append(f"Selected domains: {', '.join(routing.get('selected_domains', []) or ['n/a'])}")
        exp = run_dict.get('expert_decision', {})
        lines.append(f"Expert decision type: {exp.get('decision_type', 'n/a')}")
        lines.append(f"Selected experts: {', '.join(exp.get('selected_experts', []) or ['n/a'])}")
        conf = exp.get('expert_confidence')
        if conf is not None:
            lines.append(f"Expert confidence: {float(conf):.2%}")
        phase3 = run_dict.get('phase3', {})
        vd = phase3.get('validation_decision', {})
        lines.append(f"Validation result: {vd.get('result_class', 'n/a')}")
        latencies = phase3.get('phase_latencies_ms', {})
        if latencies:
            slowest = max(latencies.items(), key=lambda kv: kv[1])
            lines.append(f"Slowest phase: {slowest[0]} ({slowest[1]} ms)")

        prompt = "\n".join(lines)
        return self._llm.generate(prompt, system=_SYSTEM_PROMPT)


_agent: Optional[ConversationAgent] = None


def get_conversation_agent() -> ConversationAgent:
    """Return a module-level singleton ConversationAgent."""
    global _agent
    if _agent is None:
        _agent = ConversationAgent()
    return _agent
