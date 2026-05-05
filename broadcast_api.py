from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict

from run_workflow import run_mycelium_workflow


app = FastAPI(title="Mycelium Broadcast API", version="0.1.0")

# Allow local Next.js dev server by default; can be restricted in prod
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    text: str


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/api/query")
async def query(req: QueryRequest) -> Dict[str, Any]:
    # Reuse existing orchestrator; single-sentence wrapper for now
    all_data, metrics = run_mycelium_workflow([req.text])
    record = all_data[0]

    return {
        "sentence": record.get("sentence"),
        "layer0": record.get("layer0_routing", {}),
        "routing": record.get("routing_context", {}),
        "phase2": record.get("phase2_result", {}),
        "phase3": record.get("phase3_result", {}),
        "expert_decision": record.get("expert_decision", {}),
        "expert_flag": record.get("expert_flag"),
        "selected_domain": record.get("selected_domain"),
        "decision_confidence": record.get("decision_confidence"),
        "metrics": metrics.to_dict(),
    }
