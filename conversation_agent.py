from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from api_models import MyceliumRunSummary
from llm_providers import LLMClient, default_llm_client_from_env


class ConversationTurn(BaseModel):
    role: str
    content: str


class ConversationContext(BaseModel):
    trace: MyceliumRunSummary
    turns: List[ConversationTurn] = Field(default_factory=list)


class ConversationAgent:
    """Conversational layer that explains Mycelium runs.

    Backed by the shared LLMClient (Ollama primary, HF secondary).
    """

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or default_llm_client_from_env()

    def build_system_prompt(self) -> str:
        return (
            "You are Mycelium's explainer. You receive structured pipeline "
            "outputs describing how a query was routed, which experts were "
            "selected, and what validation decided. Your job is to explain "
            "these decisions clearly and conservatively, without inventing "
            "facts that are not implied by the data you are given."
        )

    def build_user_prompt(
        self, user_query: str, context: ConversationContext
    ) -> str:
        run = context.trace
        lines = [
            f"User question: {user_query}",
            "",
            "Mycelium pipeline summary:",
            f"- Layer0 route: {run.layer0.route}",
            f"- Routing classification: {run.routing.classification}",
            f"- Selected domains: {', '.join(run.routing.selected_domains) or 'none'}",
            f"- Expert decision: {run.expert_decision.decision_type}",
            f"- Selected experts: {', '.join(run.expert_decision.selected_experts) or 'none'}",
            f"- Expert confidence: {run.expert_decision.expert_confidence}",
            f"- Validation result: {run.phase3.validation_decision.get('result_class')}",
            f"- Phase latencies (ms): {run.phase3.phase_latencies_ms}",
            "",
            "Explain to the user:",
            "1. How their question was classified and routed.",
            "2. Which experts and domains were involved.",
            "3. What the validation step concluded.",
            "4. Any uncertainties, without over-claiming.",
        ]
        return "\n".join(lines)

    def answer(self, user_query: str, run: MyceliumRunSummary) -> str:
        context = ConversationContext(trace=run)
        system = self.build_system_prompt()
        prompt = self.build_user_prompt(user_query, context)
        return self.llm.generate(prompt, system=system)


# Small helper for API layer

agent_singleton: Optional[ConversationAgent] = None


def get_conversation_agent() -> ConversationAgent:
    global agent_singleton
    if agent_singleton is None:
        agent_singleton = ConversationAgent()
    return agent_singleton
