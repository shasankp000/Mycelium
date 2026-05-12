"""
expert_post_check.trm_adapter
==============================
TRM-slot adapter.

Currently backed by a local Ollama instance running ``qwen3.5:9b``.
When the TRM architecture is retrained on Mycelium's domain corpora,
swap the internals of ``TRMAdapter.reason()`` only -- nothing else
in the post-check system needs to change.

Interface contract
------------------
    adapter = TRMAdapter()
    result  = adapter.reason(query, domain)   # -> ReasoningResult

ReasoningResult fields
----------------------
    answer      : str    -- the reasoned text answer
    confidence  : float  -- self-reported or heuristic confidence [0, 1]
    latency_ms  : float  -- wall-clock inference time
    source      : str    -- identifier for which backend produced the answer
    raw         : dict   -- full raw response payload for tracing
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Hard caps -- prevent Qwen3 thinking-mode from generating unbounded tokens
# and saturating RAM.
#
# OLLAMA_NUM_PREDICT is set to 1024 rather than 512 because Qwen3.5's
# <think>...</think> block alone can consume 400-500 tokens, leaving nothing
# for the actual answer at 512.  Raise further if answers are still truncated.
# Once TRM weights replace the stub this constant becomes irrelevant.
OLLAMA_NUM_PREDICT = 1024   # max output tokens (think block + answer)
OLLAMA_NUM_CTX = 2048       # context window

SYSTEM_PROMPT = (
    "You are a precise domain reasoning engine. "
    "Given a query and its domain context, produce a concise, "
    "factually accurate answer. Do not add disclaimers or padding. "
    "If you are uncertain, say so explicitly with a confidence estimate."
)

# Regex to strip Qwen3 internal <think>...</think> blocks from output
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class ReasoningResult:
    answer: str
    confidence: float
    latency_ms: float
    source: str
    raw: Dict[str, Any] = field(default_factory=dict)


class TRMAdapter:
    """TRM-slot reasoner backed by qwen3.5:9b via Ollama.

    Swap the body of ``reason()`` when the real TRM weights are ready.
    The constructor and public interface are intentionally frozen.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public interface (frozen -- do not rename)
    # ------------------------------------------------------------------

    def reason(self, query: str, domain: str) -> ReasoningResult:
        """Run inference on *query* within *domain* context.

        Args:
            query:  Raw user query string.
            domain: Domain name (e.g. ``"physics"``).

        Returns:
            A ``ReasoningResult`` with the answer and metadata.
        """
        start = time.perf_counter()
        prompt = self._build_prompt(query, domain)

        try:
            raw = self._call_ollama(prompt)
            answer = self._extract_answer(raw)
            confidence = self._heuristic_confidence(answer)
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "TRM-slot(%s) answered in %.1f ms (conf=%.3f)",
                self._model, latency_ms, confidence,
            )
            return ReasoningResult(
                answer=answer,
                confidence=confidence,
                latency_ms=latency_ms,
                source=f"ollama/{self._model}",
                raw=raw,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("TRM-slot call failed: %s", exc)
            return ReasoningResult(
                answer="",
                confidence=0.0,
                latency_ms=latency_ms,
                source=f"ollama/{self._model}",
                raw={"error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_client(self):
        """Return an ollama client if the library is available, else None."""
        try:
            import ollama  # type: ignore
            return ollama
        except ImportError:
            logger.warning(
                "ollama package not installed. "
                "Install with: pip install ollama"
            )
            return None

    def _build_prompt(self, query: str, domain: str) -> str:
        return (
            f"Domain context: {domain}\n"
            f"Query: {query}\n"
            "Answer:"
        )

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call the Ollama REST API and return the raw response dict."""
        if self._client is None:
            raise RuntimeError("ollama client unavailable")

        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "num_predict": OLLAMA_NUM_PREDICT,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        )
        # ollama-python returns an object with a .message attribute
        if hasattr(response, "message"):
            return {
                "content": response.message.content,
                "model": getattr(response, "model", self._model),
                "done": getattr(response, "done", True),
            }
        # dict-style fallback
        if isinstance(response, dict):
            msg = response.get("message", {})
            return {
                "content": msg.get("content", ""),
                "model": response.get("model", self._model),
                "done": response.get("done", True),
            }
        return {"content": str(response)}

    def _extract_answer(self, raw: Dict[str, Any]) -> str:
        """Extract and clean the answer, stripping Qwen3 <think> blocks.

        Logs a DEBUG message showing raw vs stripped lengths so that
        empty-answer cases (think block consumed all num_predict tokens)
        are immediately visible without printing the full model output.
        """
        content = raw.get("content", "").strip()
        raw_len = len(content)
        # Remove internal chain-of-thought blocks emitted by Qwen3 thinking mode
        content = _THINK_RE.sub("", content).strip()
        stripped_len = len(content)
        logger.debug(
            "TRM _extract_answer: raw_len=%d stripped_len=%d empty=%s",
            raw_len, stripped_len, stripped_len == 0,
        )
        if stripped_len == 0 and raw_len > 0:
            logger.warning(
                "TRM answer is empty after stripping <think> block "
                "(raw_len=%d). num_predict=%d may be too low -- "
                "consider raising OLLAMA_NUM_PREDICT.",
                raw_len, OLLAMA_NUM_PREDICT,
            )
        return content

    def _heuristic_confidence(self, answer: str) -> float:
        """Estimate confidence from answer characteristics.

        Rules (additive, capped at 1.0):
          +0.5  base for non-empty answer
          +0.2  length >= 50 chars (substantive response)
          +0.1  no uncertainty markers ("uncertain", "not sure", "I don't know")
          +0.2  explicit confidence stated ("confidence: X" in answer)
        """
        if not answer:
            return 0.0
        score = 0.5
        if len(answer) >= 50:
            score += 0.2
        uncertainty_markers = ("uncertain", "not sure", "i don't know",
                                "i'm not", "unclear", "cannot determine")
        if not any(m in answer.lower() for m in uncertainty_markers):
            score += 0.1
        if "confidence:" in answer.lower():
            score += 0.2
        return min(score, 1.0)
