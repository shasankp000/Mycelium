"""
tests/test_replay_hooks.py
--------------------------
Phase D Step 12 — Semantic replay hooks + SSE event hardening.

Spec ref: implementation spec v0.2.1 — Phase D Step 12.

Test classes
------------
TestSequenceMonotonicity      sequence_number is strictly increasing per emitter
TestEventIdUniqueness         every event_id is a distinct UUID
TestReplayJournalOrdering     replay_from returns events in ascending seq order
TestReplayJournalFilter       replay_from respects last_seen cutoff
TestReplayJournalHeartbeat    heartbeat events are excluded from replay_from
TestDuplicateDropping         duplicate sequence_numbers are silently dropped
TestPhaseDPhaseNames          all Phase D phase names resolve to phase_id < 50
TestHeartbeatReplaySafe       heartbeat events carry replay_safe=False;
                               all regular events carry replay_safe=True
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import List

import pytest

from mycelium.pipeline.pipeline_event import (
    EventEmitter,
    PipelineEvent,
    ReplayJournal,
    _phase_id_for,
    PHASE_REGISTRY,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHASE_D_NAMES = [
    "predicate_extraction",
    "evidence_retrieval",
    "evidence_scoring",
    "evidence_dst_fusion",
    "contradiction_integration",
    "promote_shadow_domain",
    "graph_trm_decision",
    "graph:dfs_step",
    "graph:tool_start",
    "graph:tool_done",
    "graph:synthesis_start",
]


def _make_emitter(max_hz: float = 100.0) -> tuple[EventEmitter, List[PipelineEvent]]:
    """Return (emitter, received_events) wired to a collecting callback."""
    received: List[PipelineEvent] = []

    def _on_event(ev: PipelineEvent) -> None:
        received.append(ev)

    emitter = EventEmitter(
        request_id="test-request",
        wall_start=time.monotonic(),
        max_hz=max_hz,
        on_public_event=_on_event,
    )
    return emitter, received


# ---------------------------------------------------------------------------
# TestSequenceMonotonicity
# ---------------------------------------------------------------------------

class TestSequenceMonotonicity:
    def test_sequence_starts_at_one(self):
        emitter, _ = _make_emitter()
        emitter.emit(phase_name="setting_up", message="start")
        events = emitter.journal.all_events()
        assert events[0].sequence_number == 1

    def test_sequence_strictly_increases(self):
        emitter, _ = _make_emitter()
        for phase in ["setting_up", "environment_ready", "graph_expert_init"]:
            emitter.emit(phase_name=phase, message=phase)
        seqs = [e.sequence_number for e in emitter.journal.all_events()]
        assert seqs == sorted(seqs)
        assert len(seqs) == len(set(seqs)), "sequence_numbers must be unique"

    def test_sequence_threadsafe(self):
        """Concurrent emits from multiple threads must still produce unique seqs."""
        emitter, _ = _make_emitter()
        barrier = threading.Barrier(10)

        def _emit_batch():
            barrier.wait()
            for _ in range(10):
                emitter.emit(phase_name="graph_routing", message="concurrent")

        threads = [threading.Thread(target=_emit_batch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        seqs = [e.sequence_number for e in emitter.journal.all_events()]
        assert len(seqs) == len(set(seqs)), "concurrent emits produced duplicate sequence_numbers"


# ---------------------------------------------------------------------------
# TestEventIdUniqueness
# ---------------------------------------------------------------------------

class TestEventIdUniqueness:
    def test_all_event_ids_unique(self):
        emitter, _ = _make_emitter()
        for i in range(20):
            emitter.emit(phase_name="graph_routing", message=f"msg-{i}")
        ids = [e.event_id for e in emitter.journal.all_events()]
        assert len(ids) == len(set(ids)), "event_ids must all be distinct UUIDs"

    def test_event_id_is_valid_uuid(self):
        emitter, _ = _make_emitter()
        emitter.emit(phase_name="setting_up", message="x")
        ev = emitter.journal.all_events()[0]
        parsed = uuid.UUID(ev.event_id)  # raises if not a valid UUID
        assert str(parsed) == ev.event_id


# ---------------------------------------------------------------------------
# TestReplayJournalOrdering
# ---------------------------------------------------------------------------

class TestReplayJournalOrdering:
    def test_replay_from_zero_returns_all_in_order(self):
        emitter, _ = _make_emitter()
        phases = ["setting_up", "environment_ready", "graph_expert_init",
                  "graph_spectral_sync", "graph_router_ready"]
        for p in phases:
            emitter.emit(phase_name=p, message=p)
        replayed = emitter.journal.replay_from(0)
        seqs = [e.sequence_number for e in replayed]
        assert seqs == sorted(seqs), "replay_from must return events in ascending seq order"
        assert len(replayed) == len(phases)

    def test_replay_order_is_deterministic_regardless_of_insertion(self):
        """Manually insert events out of order; replay_from must still sort them."""
        journal = ReplayJournal(maxlen=50)
        for seq in [3, 1, 4, 1, 5, 9]:  # deliberately out of order, with duplicate
            ev = PipelineEvent(
                sequence_number=seq,
                phase_name="graph_routing",
                replay_safe=True,
            )
            journal.record(ev)
        replayed = journal.replay_from(0)
        seqs = [e.sequence_number for e in replayed]
        assert seqs == sorted(seqs), "replay_from output must be sorted ascending"


# ---------------------------------------------------------------------------
# TestReplayJournalFilter
# ---------------------------------------------------------------------------

class TestReplayJournalFilter:
    def test_replay_from_mid_returns_tail(self):
        emitter, _ = _make_emitter()
        for i in range(10):
            emitter.emit(phase_name="graph_routing", message=f"msg-{i}")
        replayed = emitter.journal.replay_from(5)
        seqs = [e.sequence_number for e in replayed]
        assert all(s > 5 for s in seqs), "replay_from(5) must only return seq > 5"

    def test_replay_from_last_returns_empty(self):
        emitter, _ = _make_emitter()
        for i in range(5):
            emitter.emit(phase_name="graph_routing", message=f"msg-{i}")
        last_seq = max(e.sequence_number for e in emitter.journal.all_events())
        replayed = emitter.journal.replay_from(last_seq)
        assert replayed == [], "replay_from(last_seq) must return empty list"


# ---------------------------------------------------------------------------
# TestReplayJournalHeartbeat
# ---------------------------------------------------------------------------

class TestReplayJournalHeartbeat:
    def test_heartbeats_excluded_from_replay(self):
        emitter, _ = _make_emitter()
        emitter.emit(phase_name="setting_up", message="start")
        emitter.emit_heartbeat(idx=0)
        emitter.emit(phase_name="environment_ready", message="ready")
        replayed = emitter.journal.replay_from(0)
        for ev in replayed:
            assert ev.replay_safe, (
                f"replay_from returned a non-replay_safe event: {ev.phase_name}"
            )

    def test_heartbeat_present_in_all_events(self):
        """Heartbeat IS recorded in the journal; just excluded from replay_from."""
        emitter, _ = _make_emitter()
        emitter.emit_heartbeat(idx=1)
        all_ev = emitter.journal.all_events()
        heartbeats = [e for e in all_ev if e.phase_name == "heartbeat"]
        assert len(heartbeats) == 1
        assert heartbeats[0].replay_safe is False


# ---------------------------------------------------------------------------
# TestDuplicateDropping
# ---------------------------------------------------------------------------

class TestDuplicateDropping:
    def test_duplicate_sequence_silently_dropped(self):
        """Force a duplicate sequence_number by manipulating the emitter's
        internal sequence counter directly.

        This test relies on two private attributes (_seq, _seq_lock) that are
        an implementation detail of EventEmitter.  We guard with hasattr so
        the test skips gracefully if the implementation renames them instead
        of raising a confusing AttributeError.
        """
        emitter, _ = _make_emitter()
        emitter.emit(phase_name="setting_up", message="first")
        first_seq = emitter.journal.all_events()[0].sequence_number

        if not (hasattr(emitter, "_seq") and hasattr(emitter, "_seq_lock")):
            pytest.skip(
                "EventEmitter does not expose _seq/_seq_lock; "
                "duplicate-drop invariant cannot be tested via private API."
            )

        # Inject the already-seen sequence back into the seen set and
        # manually force the counter backward to produce a collision.
        with emitter._seq_lock:
            emitter._seq = first_seq  # will produce first_seq+1 on next emit
        # Now decrement so the next increment lands on first_seq exactly
        with emitter._seq_lock:
            emitter._seq = first_seq - 1
        # The next emit will try seq=first_seq, which is already seen → drop
        emitter.emit(phase_name="environment_ready", message="should be dropped")

        journal_seqs = [e.sequence_number for e in emitter.journal.all_events()]
        assert journal_seqs.count(first_seq) == 1, (
            "duplicate sequence_number must not appear twice in the journal"
        )


# ---------------------------------------------------------------------------
# TestPhaseDPhaseNames
# ---------------------------------------------------------------------------

class TestPhaseDPhaseNames:
    @pytest.mark.parametrize("phase_name", PHASE_D_NAMES)
    def test_phase_d_name_resolves_to_known_id(self, phase_name: str):
        """All Phase D phase names must resolve to a phase_id < 50.
        50 is the fallback for unknown phases.
        """
        pid = _phase_id_for(phase_name)
        assert pid < 50, (
            f"Phase D name '{phase_name}' resolved to fallback phase_id=50; "
            f"it must be registered in PHASE_REGISTRY."
        )

    def test_all_phase_d_names_in_registry(self):
        for name in PHASE_D_NAMES:
            assert name in PHASE_REGISTRY, (
                f"'{name}' missing from PHASE_REGISTRY"
            )

    def test_phase_d_evidence_chain_ordering(self):
        """predicate_extraction < evidence_retrieval < evidence_scoring
        < evidence_dst_fusion < contradiction_integration."""
        chain = [
            "predicate_extraction",
            "evidence_retrieval",
            "evidence_scoring",
            "evidence_dst_fusion",
            "contradiction_integration",
        ]
        ids = [_phase_id_for(p) for p in chain]
        assert ids == sorted(ids), (
            "Phase D evidence chain must have strictly increasing phase_ids"
        )


# ---------------------------------------------------------------------------
# TestHeartbeatReplaySafe
# ---------------------------------------------------------------------------

class TestHeartbeatReplaySafe:
    def test_heartbeat_replay_safe_false(self):
        emitter, _ = _make_emitter()
        emitter.emit_heartbeat(idx=0)
        hb = [e for e in emitter.journal.all_events() if e.phase_name == "heartbeat"]
        assert len(hb) == 1
        assert hb[0].replay_safe is False, "heartbeat must carry replay_safe=False"

    def test_regular_event_replay_safe_true(self):
        emitter, _ = _make_emitter()
        emitter.emit(phase_name="setting_up", message="x")
        ev = emitter.journal.all_events()[0]
        assert ev.replay_safe is True, "regular events must carry replay_safe=True"

    def test_all_heartbeats_excluded_mixed_stream(self):
        """Mix regular events and heartbeats; replay must contain only regular ones."""
        emitter, _ = _make_emitter()
        emitter.emit(phase_name="setting_up", message="a")
        emitter.emit_heartbeat(idx=2)
        emitter.emit(phase_name="graph_routing", message="b")
        emitter.emit_heartbeat(idx=3)
        emitter.emit(phase_name="done", message="c")

        replayed = emitter.journal.replay_from(0)
        assert all(e.replay_safe for e in replayed)
        assert len(replayed) == 3  # only the 3 regular events
        assert [e.phase_name for e in replayed] == [
            "setting_up", "graph_routing", "done"
        ]
