// ---------------------------------------------------------------------------
// SubgraphControls — three toggle buttons: Pipeline | Reasoning | Evidence
// Plan ref: §2.3.4
// ---------------------------------------------------------------------------

import type { SubgraphZone } from '../../types/graph';
import styles from '../../styles/ReasoningGraph.module.css';

const ZONES: { value: SubgraphZone; label: string }[] = [
  { value: 'pipeline',  label: 'Pipeline'  },
  { value: 'reasoning', label: 'Reasoning' },
  { value: 'evidence',  label: 'Evidence'  },
];

export function SubgraphControls({
  activeZones,
  onToggleZone,
}: {
  activeZones: Set<SubgraphZone>;
  onToggleZone: (zone: SubgraphZone) => void;
}) {
  return (
    <div className={styles.subgraphControls} role="group" aria-label="Subgraph visibility">
      {ZONES.map(({ value, label }) => (
        <button
          key={value}
          className={`${styles.subgraphBtn} ${activeZones.has(value) ? styles.subgraphBtnActive : ''}`}
          onClick={() => onToggleZone(value)}
          aria-pressed={activeZones.has(value)}
          aria-label={`Toggle ${label} subgraph`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
