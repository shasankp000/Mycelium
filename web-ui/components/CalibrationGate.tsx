// ---------------------------------------------------------------------------
// CalibrationGate — extracted from pages/index.tsx (Phase 1 refactor)
// Polls /api/calibrate/status until the expert system is ready, then calls
// onReady() to hand control back to the parent page.
// ---------------------------------------------------------------------------

import { useState, useRef, useEffect, useCallback } from 'react';
import axios, { AxiosError } from 'axios';
import styles from '../styles/Home.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';
const CALIBRATION_POLL_INTERVAL_MS = 4000;

type CalibrationStatus = 'checking' | 'pending' | 'running' | 'complete' | 'error';

export function CalibrationGate({ onReady }: { onReady: () => void }) {
  const [status, setStatus]         = useState<CalibrationStatus>('checking');
  const [progress, setProgress]     = useState(0);
  const [statusText, setStatusText] = useState('Checking expert system…');
  const [errorMsg, setErrorMsg]     = useState<string | null>(null);
  const pollRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const jobIdRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  const startPolling = useCallback((jobId: string) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get<{
          job_id: string; status: string; progress: number; error: string | null;
        }>(`${API_BASE}/api/calibrate/status/${jobId}`, { timeout: 8000 });
        const data = res.data;
        setProgress(data.progress ?? 0);
        if (data.status === 'complete') {
          stopPolling();
          setStatus('complete');
          setStatusText('Expert system ready ✓');
          setProgress(100);
          setTimeout(onReady, 600);
        } else if (data.status === 'error') {
          stopPolling();
          setStatus('error');
          setErrorMsg(data.error ?? 'Unknown calibration error');
          setStatusText('Calibration failed');
        } else if (data.status === 'running') {
          setStatus('running');
          setStatusText(`Calibrating expert system… (${data.progress ?? 0}%)`);
        } else {
          setStatusText('Expert calibration queued…');
        }
      } catch {
        setStatusText('Waiting for backend…');
      }
    }, CALIBRATION_POLL_INTERVAL_MS);
  }, [onReady, stopPolling]);

  const kickOffCalibration = useCallback(async () => {
    setStatus('pending');
    setStatusText('Starting expert calibration…');
    setErrorMsg(null);
    try {
      const res = await axios.post<{ job_id: string; status: string }>(
        `${API_BASE}/api/calibrate/start`, {}, { timeout: 10000 },
      );
      const jobId = res.data.job_id;
      jobIdRef.current = jobId;
      setStatusText('Expert calibration started — initialising models…');
      startPolling(jobId);
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      setStatus('error');
      setErrorMsg(axiosErr?.response?.data?.detail ?? axiosErr?.message ?? 'Could not reach backend');
      setStatusText('Failed to start calibration');
    }
  }, [startPolling]);

  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        const res = await axios.get<{ ready: boolean }>(
          `${API_BASE}/api/calibrate/ready`, { timeout: 5000 },
        );
        if (cancelled) return;
        if (res.data.ready) { onReady(); return; }
      } catch { /* proceed to full calibration */ }
      if (!cancelled) kickOffCalibration();
    }
    init();
    return () => { cancelled = true; stopPolling(); };
  }, [kickOffCalibration, onReady, stopPolling]);

  const isComplete = status === 'complete';
  const isError    = status === 'error';

  return (
    <div className={styles.calibrationGate} role="status" aria-live="polite">
      <div className={styles.calibrationCard}>
        <div className={styles.calibrationLogo} aria-hidden="true">
          <svg width="48" height="48" viewBox="0 0 28 28" fill="none">
            <circle cx="14" cy="14" r="3.5" fill="currentColor" opacity="0.9" />
            <line x1="14" y1="14" x2="4"  y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <line x1="14" y1="14" x2="24" y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <line x1="14" y1="14" x2="4"  y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <line x1="14" y1="14" x2="24" y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <line x1="14" y1="14" x2="14" y2="2"  stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <line x1="14" y1="14" x2="14" y2="26" stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <circle cx="4"  cy="6"  r="2" fill="currentColor" opacity="0.3" />
            <circle cx="24" cy="6"  r="2" fill="currentColor" opacity="0.3" />
            <circle cx="4"  cy="22" r="2" fill="currentColor" opacity="0.3" />
            <circle cx="24" cy="22" r="2" fill="currentColor" opacity="0.3" />
            <circle cx="14" cy="2"  r="2" fill="currentColor" opacity="0.3" />
            <circle cx="14" cy="26" r="2" fill="currentColor" opacity="0.3" />
          </svg>
        </div>
        <h1 className={styles.calibrationTitle}>Mycelium</h1>
        <p className={styles.calibrationSubtitle}>Setting up expert system</p>
        <div
          className={styles.calibrationBarTrack}
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Calibration progress"
        >
          <div
            className={`${styles.calibrationBarFill} ${
              isComplete ? styles.calibrationBarComplete :
              isError    ? styles.calibrationBarError    : ''
            }`}
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className={`${styles.calibrationStatus} ${
          isComplete ? styles.calibrationStatusOk  :
          isError    ? styles.calibrationStatusErr : ''
        }`}>
          {statusText}
        </p>
        {isError && errorMsg && (
          <div className={styles.calibrationError}>
            <p className={styles.calibrationErrorDetail}>{errorMsg}</p>
            <button
              className={styles.calibrationRetry}
              onClick={kickOffCalibration}
              aria-label="Retry calibration"
            >
              Retry
            </button>
          </div>
        )}
        {!isComplete && !isError && status !== 'checking' && (
          <p className={styles.calibrationHint}>
            This only runs once on startup — K-Medoids clustering, OOD detection,
            and probability calibration across all expert domains.
          </p>
        )}
      </div>
    </div>
  );
}
