// ---------------------------------------------------------------------------
// useElapsedTick — counts elapsed milliseconds since start() was called.
// Ticks every 250 ms. Cleans up on unmount.
// ---------------------------------------------------------------------------

import { useState, useRef, useCallback } from 'react';

export function useElapsedTick() {
  const [elapsedMs, setElapsedMs] = useState(0);
  const tickRef      = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  const start = useCallback(() => {
    startTimeRef.current = Date.now();
    setElapsedMs(0);
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(() => {
      setElapsedMs(Date.now() - startTimeRef.current);
    }, 250);
  }, []);

  const stop = useCallback(() => {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }, []);

  // Expose a dispose function for unmount cleanup
  const dispose = stop;

  return { elapsedMs, start, stop, dispose };
}
