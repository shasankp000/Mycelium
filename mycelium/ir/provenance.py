"""
Phase F — ProvenanceChain Builder
===================================
Populates ProvenanceChain.worker_threads correctly for multi-worker use.

Phase D stub:
    In Phases A-E, ProvenanceChain.worker_threads was always ["main"].
    This was a known Phase D shortcut (§29 note: "worker_threads is
    populated in Phase F").

Phase F fix:
    ProvenanceBuilder.for_worker() reads threading.current_thread().name
    at construction time so each worker’s contribution is correctly
    attributed in the audit trail.

    ProvenanceBuilder.merge() combines two ProvenanceChains into one
    (union of sources, paths, evidence nodes, worker threads) for nodes
    processed by multiple workers in parallel DFS.

Design note:
    ProvenanceBuilder does not mutate existing ProvenanceChain objects.
    It always returns new instances.  This is consistent with the £38
    immutability contract on IRGraph and its sub-objects.
"""

from __future__ import annotations

import datetime
import threading
from typing import List, Optional

try:
    from mycelium.ir.primitives import ProvenanceChain
    _PC_AVAILABLE = True
except ImportError:
    ProvenanceChain = None  # type: ignore[assignment,misc]
    _PC_AVAILABLE = False


class ProvenanceBuilder:
    """Constructs and merges ProvenanceChain objects with worker attribution.

    Usage
    -----
    # In a worker thread:
    builder = ProvenanceBuilder()
    prov = builder.for_worker(
        sources=["https://arxiv.org/abs/..."],
        reasoning_paths=["phase_b_canonicalization", "phase_c_spectral"],
        decomposition_origin="srl_extraction",
    )
    # prov.worker_threads == ["DFSWorker-0"]  (current thread name)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def for_worker(
        self,
        *,
        sources: Optional[List[str]] = None,
        reasoning_paths: Optional[List[str]] = None,
        decomposition_origin: str = "unknown",
        evidence_nodes: Optional[List[str]] = None,
        ontology_resolution_path: Optional[List[str]] = None,
        thread_name: Optional[str] = None,
    ) -> "ProvenanceChain":
        """Build a ProvenanceChain attributed to the current (or named) thread.

        Parameters
        ----------
        sources : list[str], optional
            Source URLs or tool IDs.
        reasoning_paths : list[str], optional
            Ordered list of layer/phase decisions.
        decomposition_origin : str
            Which decomposition template was used.
        evidence_nodes : list[str], optional
            EvidenceItem IDs from Layer 4.
        ontology_resolution_path : list[str], optional
            Sequence of ontology alignments applied.
        thread_name : str, optional
            Override thread name.  If None, reads
            threading.current_thread().name.

        Returns
        -------
        ProvenanceChain
        """
        if not _PC_AVAILABLE:
            raise RuntimeError(
                "ProvenanceBuilder: mycelium.ir.primitives unavailable"
            )

        worker_name = thread_name or threading.current_thread().name

        return ProvenanceChain(
            sources=list(sources or []),
            reasoning_paths=list(reasoning_paths or []),
            decomposition_origin=decomposition_origin,
            evidence_nodes=list(evidence_nodes or []),
            ontology_resolution_path=list(ontology_resolution_path or []),
            worker_threads=[worker_name],   # ← Phase F: real thread name
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

    def from_existing(
        self,
        existing: "ProvenanceChain",
        *,
        thread_name: Optional[str] = None,
    ) -> "ProvenanceChain":
        """Create a new ProvenanceChain from an existing one, updating the
        worker_threads list to include the current thread.

        Used when a worker re-processes a node that already has provenance
        from a previous phase.
        """
        worker_name = thread_name or threading.current_thread().name
        new_threads = list(existing.worker_threads)
        if worker_name not in new_threads:
            new_threads.append(worker_name)

        return ProvenanceChain(
            sources=list(existing.sources),
            reasoning_paths=list(existing.reasoning_paths),
            decomposition_origin=existing.decomposition_origin,
            evidence_nodes=list(existing.evidence_nodes),
            ontology_resolution_path=list(existing.ontology_resolution_path),
            worker_threads=new_threads,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge(
        self,
        prov_a: "ProvenanceChain",
        prov_b: "ProvenanceChain",
    ) -> "ProvenanceChain":
        """Merge two ProvenanceChains into one.

        Used when a node is processed by multiple workers in parallel DFS.
        Union semantics: all unique entries from both chains are preserved.
        The ordering within each list is deterministic (sorted).

        Parameters
        ----------
        prov_a, prov_b : ProvenanceChain
            The two chains to merge.

        Returns
        -------
        ProvenanceChain
            A new chain representing the union of both attribution trails.
        """
        if not _PC_AVAILABLE:
            raise RuntimeError("ProvenanceBuilder: mycelium.ir.primitives unavailable")

        def union_sorted(a: list, b: list) -> list:
            return sorted(set(a) | set(b))

        def union_ordered(a: list, b: list) -> list:
            """Union preserving first-seen order (a first, then new items from b)."""
            seen = set(a)
            result = list(a)
            for item in b:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result

        return ProvenanceChain(
            sources=union_ordered(prov_a.sources, prov_b.sources),
            reasoning_paths=union_ordered(
                prov_a.reasoning_paths, prov_b.reasoning_paths
            ),
            decomposition_origin=(
                prov_a.decomposition_origin
                if prov_a.decomposition_origin == prov_b.decomposition_origin
                else f"{prov_a.decomposition_origin}+{prov_b.decomposition_origin}"
            ),
            evidence_nodes=union_ordered(
                prov_a.evidence_nodes, prov_b.evidence_nodes
            ),
            ontology_resolution_path=union_ordered(
                prov_a.ontology_resolution_path,
                prov_b.ontology_resolution_path,
            ),
            worker_threads=union_sorted(
                prov_a.worker_threads, prov_b.worker_threads
            ),
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )
