// ---------------------------------------------------------------------------
// useGraphStabilization — manages the freeze lifecycle of the force simulation.
// Plan ref: §2.3.6
//
// Freeze triggers:
//   - 2.5 s without a new node arriving
//   - OR receipt of done/error (via markDone())
//
// After freeze the caller must call freeze(positions) on useGraphBuilder
// to fix node positions and enter low-CPU exploration mode.
// ---------------------------------------------------------------------------

import { useState, useEffect, useRef, useCallback } from 'react';
import type { GraphLifecycle } from '../types/graph';

const IDLE_FREEZE_MS    = 2500;   // freeze 2.5 s after last node arrival
const DONE_FREEZE_MS   = 1000;   // freeze 1 s after done/error (shorter — allow final settle)

export interface StabilizationControls {
  /** Notify that a new node arrived — resets the idle timer */
  notifyNodeArrival: () => void;
  /** Notify that done/error was received — triggers shorter timer */
  markDone: () => void;
  /** Whether the simulation should currently be running */
  simulationActive: boolean;
  /** Current stabilization state for UI indicators */
  stabState: 'idle' | 'active' | 'cooling' | 'frozen';
  /** Manually resume simulation (§2.3.6 toolbar action) */
  resumeSimulation: () => void;
  /** Reset all state (called on new query) */
  reset: () => void;
}

export function useGraphStabilization(
  lifecycle: GraphLifecycle,
  /** Called by the hook when freeze should happen.
   *  Caller is responsible for reading node positions from the
   *  canvas ref and passing them to useGraphBuilder.freeze() */
  onFreezeReady: () => void,
): StabilizationControls {
  const [stabState, setStabState] = useState<'idle' | 'active' | 'cooling' | 'frozen'>('idle');
  const [simulationActive, setSimulationActive] = useState(false);

  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const doneTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function clearTimers() {
    if (idleTimerRef.current) { clearTimeout(idleTimerRef.current); idleTimerRef.current = null; }
    if (doneTimerRef.current) { clearTimeout(doneTimerRef.current); doneTimerRef.current = null; }
  }

  // ── notifyNodeArrival ───────────────────────────────────────────────
  const notifyNodeArrival = useCallback(() => {
    if (stabState === 'frozen') return; // don't disturb frozen state
    setStabState('active');
    setSimulationActive(true);
    // Reset idle timer
    if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    idleTimerRef.current = setTimeout(() => {
      setStabState('cooling');
      // Cooling phase — give one more beat then freeze
      idleTimerRef.current = setTimeout(() => {
        setStabState('frozen');
        setSimulationActive(false);
        onFreezeReady();
      }, 600);
    }, IDLE_FREEZE_MS);
  }, [stabState, onFreezeReady]);

  // ── markDone ─────────────────────────────────────────────────────────────
  const markDone = useCallback(() => {
    clearTimers();
    setStabState('cooling');
    doneTimerRef.current = setTimeout(() => {
      setStabState('frozen');
      setSimulationActive(false);
      onFreezeReady();
    }, DONE_FREEZE_MS);
  }, [onFreezeReady]);

  // ── resumeSimulation ──────────────────────────────────────────────────────
  const resumeSimulation = useCallback(() => {
    clearTimers();
    setStabState('active');
    setSimulationActive(true);
    // Auto-refreeze after 8 s of resumed physics
    idleTimerRef.current = setTimeout(() => {
      setStabState('frozen');
      setSimulationActive(false);
      onFreezeReady();
    }, 8000);
  }, [onFreezeReady]);

  // ── reset ───────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    clearTimers();
    setStabState('idle');
    setSimulationActive(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => () => clearTimers(), []);

  // Sync with lifecycle from useGraphBuilder
  useEffect(() => {
    if (lifecycle === 'idle') reset();
  }, [lifecycle, reset]);

  return {
    notifyNodeArrival,
    markDone,
    simulationActive,
    stabState,
    resumeSimulation,
    reset,
  };
}
