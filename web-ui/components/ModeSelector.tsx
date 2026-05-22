// ---------------------------------------------------------------------------
// ModeSelector — ambient pill toggle in the input row.
// Persists the selected ReasoningMode in localStorage.
// Plan ref: Section 2.1
// ---------------------------------------------------------------------------

import { useEffect } from 'react';
import type { ReasoningMode } from '../types/pipeline';
import styles from '../styles/ModeSelector.module.css';

const STORAGE_KEY = 'mycelium_reasoning_mode';

const MODES: {
  value: ReasoningMode;
  icon: string;
  label: string;
  tooltip: string;
  className: string;
}[] = [
  {
    value:     'fast',
    icon:      '⚡',
    label:     'Fast',
    tooltip:   'Layer 0 + Layer 1 only · low latency',
    className: styles.modeBtnFast,
  },
  {
    value:     'smart',
    icon:      '🧠',
    label:     'Smart',
    tooltip:   'Full pipeline + summarized reasoning (default)',
    className: styles.modeBtnSmart,
  },
  {
    value:     'researcher',
    icon:      '🔬',
    label:     'Researcher',
    tooltip:   'All layers + full DFS + evidence DAG',
    className: styles.modeBtnResearcher,
  },
];

export function loadSavedMode(): ReasoningMode {
  if (typeof window === 'undefined') return 'smart';
  const saved = window.localStorage.getItem(STORAGE_KEY);
  if (saved === 'fast' || saved === 'smart' || saved === 'researcher') return saved;
  return 'smart';
}

interface ModeSelectorProps {
  mode: ReasoningMode;
  onChange: (mode: ReasoningMode) => void;
  disabled?: boolean;
}

export function ModeSelector({ mode, onChange, disabled = false }: ModeSelectorProps) {
  // Persist to localStorage on every change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, mode);
    }
  }, [mode]);

  return (
    <div
      className={styles.modeSelector}
      role="group"
      aria-label="Reasoning mode"
    >
      {MODES.map(({ value, icon, label, tooltip, className }) => {
        const isActive = mode === value;
        return (
          <button
            key={value}
            className={[
              styles.modeBtn,
              className,
              isActive ? styles.modeBtnActive : '',
            ].join(' ').trim()}
            onClick={() => !disabled && onChange(value)}
            disabled={disabled}
            aria-pressed={isActive}
            aria-label={`${label} mode — ${tooltip}`}
            title={tooltip}
          >
            <span className={styles.modeIcon} aria-hidden="true">{icon}</span>
            <span className={styles.modeLabel}>{label}</span>
          </button>
        );
      })}
    </div>
  );
}
