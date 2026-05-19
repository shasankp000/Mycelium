"""
Phase F — Multi-Worker DFS Scheduler
======================================
Thread-safe multi-worker upgrade of the Phase D single-worker DFSLookup.

Consolidation Notes §D.2 upgrade note:
    "Phase F will replace the plain Python stack with a multi-worker
    scheduler, but the search logic (scoring, ranking, equivalence
    resolution) is preserved."

Architecture
------------
WorkerPool manages N worker threads (default: min(4, cpu_count)).

Each worker runs _search_worker():
    1. Pops a graph_id from the shared work queue (queue.Queue).
    2. Fetches the latest graph from GraphStore.
    3. Applies the search predicate.
    4. If match: acquires results_lock, appends to results list.
    5. Calls queue.task_done().

DFSLookup.search() / find_equivalent() call _dispatch():
    - If store size <= SINGLE_WORKER_THRESHOLD: use Phase D plain stack
      (zero thread overhead for small stores).
    - Otherwise: fill the work queue with all graph IDs, start workers,
      call queue.join() to wait for all tasks to complete, return results.

Thread safety guarantees:
    - queue.Queue: internally thread-safe (no extra locking needed for put/get)
    - results list: protected by threading.Lock
    - GraphStore: read-only during a search (no writes during lookup)
    - WorkerPool state (running flag): protected by threading.Lock

Backward compatibility:
    DFSLookup public API is unchanged.  Callers see no difference.
    The scheduler is an internal implementation detail.

Worker naming (£29 ProvenanceChain.worker_threads):
    Workers are named "DFSWorker-{i}" so ProvenanceBuilder.for_worker()
    can correctly attribute multi-worker search contributions.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Threshold: stores with <= this many graphs use single-worker DFS.
SINGLE_WORKER_THRESHOLD: int = 50
# Maximum number of worker threads.
MAX_WORKERS: int = 4


class WorkerPool:
    """Thread pool for parallel graph predicate evaluation.

    Parameters
    ----------
    n_workers : int
        Number of worker threads.  Defaults to min(MAX_WORKERS, cpu_count).
    """

    def __init__(self, n_workers: Optional[int] = None) -> None:
        cpu = os.cpu_count() or 1
        self._n = min(n_workers or MAX_WORKERS, MAX_WORKERS, cpu)
        self._work_queue: queue.Queue = queue.Queue()
        self._results: List[Any] = []
        self._results_lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        self._running = False
        self._running_lock = threading.Lock()

    def _search_worker(
        self,
        store: Any,
        predicate: Callable[[Any], bool],
    ) -> None:
        """Worker thread target: pop graph IDs and evaluate predicate."""
        while True:
            try:
                gid = self._work_queue.get(block=True, timeout=0.05)
            except queue.Empty:
                break
            try:
                g = store.get_latest(gid)
                if g is not None:
                    try:
                        if predicate(g):
                            with self._results_lock:
                                self._results.append(g)
                    except Exception as exc:
                        logger.debug("WorkerPool: predicate error on %s: %s", gid, exc)
            finally:
                self._work_queue.task_done()

    def dispatch(
        self,
        store: Any,
        predicate: Callable[[Any], bool],
        graph_ids: List[str],
    ) -> List[Any]:
        """Run predicate over graph_ids using N worker threads.

        Parameters
        ----------
        store : GraphStore
            Store to fetch graphs from.
        predicate : callable
            (IRGraph) -> bool  — True if graph is a match.
        graph_ids : list[str]
            IDs to search over.

        Returns
        -------
        list[IRGraph]
            All graphs for which predicate returned True.
        """
        with self._running_lock:
            self._results = []

        # Fill work queue
        for gid in graph_ids:
            self._work_queue.put(gid)

        # Spawn workers
        threads = []
        for i in range(min(self._n, len(graph_ids))):
            t = threading.Thread(
                target=self._search_worker,
                args=(store, predicate),
                name=f"DFSWorker-{i}",
                daemon=True,
            )
            threads.append(t)
            t.start()

        # Wait for all tasks to complete
        self._work_queue.join()

        # Ensure threads have exited
        for t in threads:
            t.join(timeout=2.0)

        with self._results_lock:
            return list(self._results)

    def shutdown(self) -> None:
        """Signal workers to stop.  Idempotent."""
        with self._running_lock:
            self._running = False


class MultiWorkerDFSLookup:
    """Phase F upgrade of DFSLookup: dispatches via WorkerPool for large stores.

    Drop-in replacement for Phase D DFSLookup.  Identical public API.
    Inherits all search strategies and ranking logic.

    Parameters
    ----------
    store : GraphStore
        The graph store to search.
    n_workers : int, optional
        Number of worker threads for large-store searches.
    """

    def __init__(
        self,
        store: Any,
        n_workers: Optional[int] = None,
    ) -> None:
        self._store = store
        self._pool = WorkerPool(n_workers=n_workers)

        # Re-use Phase D ranking and result-building helpers
        try:
            from mycelium.trm.dfs_lookup import DFSLookup, TRMLookupResult
            from mycelium.trm.promotion import PromotionPolicy
            self._phase_d = DFSLookup(store)
            self._TRMLookupResult = TRMLookupResult
            self._PromotionPolicy = PromotionPolicy
            self._phase_d_available = True
        except ImportError:
            self._phase_d_available = False

    # ------------------------------------------------------------------
    # Public API (identical to Phase D DFSLookup)
    # ------------------------------------------------------------------

    def search(
        self,
        query_node: Any,
        *,
        strategy: str = "BY_HASH",
        min_state_rank: int = 0,
    ) -> Any:
        """Search with automatic single/multi-worker dispatch."""
        n_graphs = len(self._store)
        if n_graphs <= SINGLE_WORKER_THRESHOLD or not self._phase_d_available:
            # Small store: delegate entirely to Phase D plain-stack DFS
            return self._phase_d.search(
                query_node, strategy=strategy, min_state_rank=min_state_rank
            )

        # Large store: multi-worker dispatch for BY_CANONICAL, BY_PREDICATE,
        # BY_STATE_RANK, and BY_EQUIVALENCE.  BY_HASH uses the hash index
        # directly (already O(1), no parallelism needed).
        if strategy == "BY_HASH":
            return self._phase_d.search(
                query_node, strategy=strategy, min_state_rank=min_state_rank
            )

        if strategy == "BY_EQUIVALENCE":
            return self.find_equivalent(query_node)

        # Build predicate from strategy
        predicate = self._make_predicate(
            query_node, strategy, min_state_rank
        )
        if predicate is None:
            return self._phase_d.search(
                query_node, strategy=strategy, min_state_rank=min_state_rank
            )

        all_ids = self._store.all_graph_ids()
        candidates = self._pool.dispatch(self._store, predicate, all_ids)

        if min_state_rank > 0:
            candidates = [
                g for g in candidates
                if self._PromotionPolicy.state_rank(g.state) >= min_state_rank
            ]

        # Re-use Phase D result building
        try:
            sig = query_node.semantic_signature
            query_hash = sig.semantic_hash
        except AttributeError:
            query_hash = ""

        return self._phase_d._build_result(candidates, strategy, query_hash)

    def find_equivalent(self, query_node: Any) -> Any:
        """Equivalence-family search with multi-worker dispatch for large stores."""
        n_graphs = len(self._store)
        if n_graphs <= SINGLE_WORKER_THRESHOLD or not self._phase_d_available:
            return self._phase_d.find_equivalent(query_node)

        try:
            equiv_family = query_node.semantic_signature.equivalence_family or []
            query_hash = query_node.semantic_signature.semantic_hash
        except AttributeError:
            return self._TRMLookupResult(
                found=False,
                explanation="query_node missing semantic_signature",
                strategy_used="BY_EQUIVALENCE",
            )

        if not equiv_family:
            return self._TRMLookupResult(
                found=False,
                explanation="No equivalence_family members to search",
                strategy_used="BY_EQUIVALENCE",
                query_hash=query_hash,
            )

        equiv_set = set(equiv_family)

        def predicate(g: Any) -> bool:
            for node in g.nodes:
                try:
                    if node.semantic_signature.canonical_form in equiv_set:
                        return True
                except AttributeError:
                    pass
            return False

        all_ids = self._store.all_graph_ids()
        candidates = self._pool.dispatch(self._store, predicate, all_ids)
        return self._phase_d._build_result(candidates, "BY_EQUIVALENCE", query_hash)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_predicate(
        self,
        query_node: Any,
        strategy: str,
        min_state_rank: int,
    ) -> Optional[Callable[[Any], bool]]:
        """Return the graph predicate for a given search strategy."""
        try:
            sig = query_node.semantic_signature
        except AttributeError:
            return None

        if strategy == "BY_CANONICAL":
            needle = sig.canonical_form.lower()
            return lambda g: any(
                needle in n.semantic_signature.canonical_form.lower()
                for n in g.nodes
                if hasattr(n, "semantic_signature")
            )

        if strategy == "BY_PREDICATE":
            pf = sig.predicate_family
            return lambda g: any(
                getattr(n.semantic_signature, "predicate_family", None) == pf
                for n in g.nodes
                if hasattr(n, "semantic_signature")
            )

        if strategy == "BY_STATE_RANK":
            return lambda g: (
                self._PromotionPolicy.state_rank(g.state) >= min_state_rank
            )

        return None
