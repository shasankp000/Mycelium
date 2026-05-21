# Mycelium System Design & Brainstorming Notes

## OOD Detection & Confidence
- OOD (Out-of-Distribution) detection uses SVM decision distance, Nearest Neighbors, and Isolation Forest.
- Ensemble approach: If any method flags OOD, input is considered out-of-distribution.
- OOD penalty reduces confidence in decisions, prioritizing safety.

## Model Calibration
- Calibration adjusts model probabilities to be more reliable.
- Uses `CalibratedClassifierCV` (isotonic regression or Platt scaling).
- Brier Score measures calibration quality (lower is better).
- Confidence is adjusted by calibration score and OOD penalty.

## Confidence Score Philosophy
- 80–85% confidence for in-distribution inputs is healthy.
- 100% confidence is a red flag for overfitting or poor calibration.

## Patch & Domain Training Workflow
- Do not train patches on single examples; batch flagged inputs.
- Store flagged inputs in JSON: `{input, flag, domain}`.
- Train new patches when batch size ≥ 10–20, or after a set time (e.g., 2 weeks).
- For truly new domains, start collecting and train when enough data is available.

## Frontend User Messaging
- If system lacks knowledge: "I'm sorry, I don't have knowledge about this topic right now. I'll learn from your input during my next scheduled training session and remember this information in the future."
- Promotes transparency and trust, counters hallucinations.

## Responsible AI & Hallucination Prevention
- System acknowledges knowledge gaps instead of guessing.
- Aligns with OpenAI research: https://openai.com/index/why-language-models-hallucinate/
- Encourages continual learning and honest feedback.

---

**Return to these notes for design, implementation, and responsible AI strategy.**

> **Archived from root** during cleanup pass (2026-05-21). Original file: `mycelium_brainstorming_notes.md`.
