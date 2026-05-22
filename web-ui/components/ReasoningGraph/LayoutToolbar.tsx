// ---------------------------------------------------------------------------
// LayoutToolbar — Reset layout | Hierarchy | Radial + Resume simulation
// Plan ref: §2.3.4, §2.3.6
// ---------------------------------------------------------------------------

import type { LayoutMode } from '../../types/graph';
import styles from '../../styles/ReasoningGraph.module.css';

const LAYOUTS: { value: LayoutMode; icon: string; label: string }[] = [
  { value: 'force',     icon: '⟳', label: 'Reset layout'  },
  { value: 'hierarchy', icon: '→', label: 'Hierarchy'     },
  { value: 'radial',    icon: '○', label: 'Radial'        },
];

export function LayoutToolbar({
  current,
  stabState,
  onChange,
  onResume,
}: {
  current:   LayoutMode;
  stabState: 'idle' | 'active' | 'cooling' | 'frozen';
  onChange:  (lm: LayoutMode) => void;
  onResume:  () => void;
}) {
  return (
    <div className={styles.layoutToolbar} role="group" aria-label="Layout controls">
      {LAYOUTS.map(({ value, icon, label }) => (
        <button
          key={value}
          className={`${styles.layoutBtn} ${current === value ? styles.layoutBtnActive : ''}`}
          onClick={() => onChange(value)}
          aria-pressed={current === value}
          aria-label={label}
          title={label}
        >
          {icon}
        </button>
      ))}
      {stabState === 'frozen' && (
        <button
          className={styles.resumeBtn}
          onClick={onResume}
          aria-label="Resume force simulation"
          title="Resume simulation"
        >
          ⟳ Resume
        </button>
      )}
    </div>
  );
}
