from __future__ import annotations
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class HeadManifestEntry:
    domain_id: str
    domain_version: int
    head_version: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    model_tag: str = ""
    created_at: str = field(default_factory=_now)
    artifact_path: str = ""
    quantized_path: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadManifest:
    schema_version: str = "1.0"
    entries: List[HeadManifestEntry] = field(default_factory=list)

    def add(self, entry: HeadManifestEntry) -> None:
        # replace existing entry for same domain_id if present
        self.entries = [e for e in self.entries if e.domain_id != entry.domain_id]
        self.entries.append(entry)

    def get(self, domain_id: str) -> Optional[HeadManifestEntry]:
        for e in self.entries:
            if e.domain_id == domain_id:
                return e
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": [asdict(e) for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HeadManifest":
        entries = [HeadManifestEntry(**e) for e in d.get("entries", [])]
        return cls(schema_version=d.get("schema_version", "1.0"), entries=entries)


def head_artifact_name(domain_id: str, domain_version: int, head_version: int) -> str:
    return f"head_v{head_version}_d{domain_version}_{domain_id}.pt"


def save_head(
    head: torch.nn.Module,
    artifact_dir: str,
    domain_id: str,
    domain_version: int,
    head_version: int,
    manifest: Optional[HeadManifest] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir = Path(artifact_dir) / domain_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = head_artifact_name(domain_id, domain_version, head_version)
    out_path = out_dir / fname
    torch.save({"state_dict": head.state_dict(), "meta": extra_meta or {}}, str(out_path))
    logger.info("Saved head → %s", out_path)

    if manifest is not None:
        entry = manifest.get(domain_id)
        if entry:
            entry.artifact_path = str(out_path)
            entry.head_version = head_version
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
    return out_path


def load_head(
    head: torch.nn.Module,
    artifact_path: str,
    *,
    map_location: str = "cpu",
) -> torch.nn.Module:
    state = torch.load(artifact_path, map_location=map_location)
    if "state_dict" in state:
        state = state["state_dict"]
    head.load_state_dict(state)
    logger.info("Loaded head ← %s", artifact_path)
    return head


def load_manifest(artifact_dir: str, domain_id: str) -> Optional[HeadManifest]:
    p = Path(artifact_dir) / domain_id / "manifest.json"
    if not p.exists():
        return None
    return HeadManifest.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_global_manifest(manifests_dir: str, manifest: HeadManifest) -> Path:
    p = Path(manifests_dir) / "global_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return p
