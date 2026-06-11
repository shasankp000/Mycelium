// ---------------------------------------------------------------------------
// HomeScreen — extracted from pages/index.tsx (Phase 1 refactor)
// Phase 5: mode + onModeChange props added so ModeSelector is available
//           before the first message is sent (Gap 1 fix).
// ---------------------------------------------------------------------------

import styles from '../styles/Home.module.css';
import { ModeSelector } from './ModeSelector';
import type { ReasoningMode } from '../types/pipeline';

const MAX_INPUT_CHARS = 2000;

const TYPEWRITER_TEXTS = [
  "Truth isn't assumed. It's earned.",
  "An AI that admits when it doesn't know is more powerful than one that pretends it does.",
  'Facts and values are different things. We treat them that way.',
  'Not built to impress. Built to be honest.',
  "Bias enters when we pretend values are facts. We don't pretend.",
  'Modular by design. Honest by principle.',
  "The no-bullshit promise: find truth where it exists, admit when it doesn't.",
  'Intelligence distributed like mycelium — resilient, adaptive, no single point of failure.',
];

export const PROMPT_SUGGESTIONS: readonly string[] = [
  'Does coffee cause cancer?',
  'Is string theory scientifically proven?',
  'What are the effects of universal basic income?',
  'How does CRISPR gene editing work?',
];

// ---------------------------------------------------------------------------
// useTypewriter hook (lives here; only HomeScreen uses it)
// ---------------------------------------------------------------------------

import { useState, useEffect } from 'react';

function useTypewriter(
  texts: string[],
  typingSpeed = 68,
  deletingSpeed = 32,
  pauseMs = 2400,
) {
  const [displayed, setDisplayed] = useState('');
  const [textIdx, setTextIdx]     = useState(0);
  const [charIdx, setCharIdx]     = useState(0);
  const [deleting, setDeleting]   = useState(false);
  const [paused, setPaused]       = useState(false);

  useEffect(() => {
    if (paused) {
      const t = setTimeout(() => { setPaused(false); setDeleting(true); }, pauseMs);
      return () => clearTimeout(t);
    }
    const current = texts[textIdx];
    if (!deleting) {
      if (charIdx < current.length) {
        const t = setTimeout(() => {
          setDisplayed(current.slice(0, charIdx + 1));
          setCharIdx((c) => c + 1);
        }, typingSpeed);
        return () => clearTimeout(t);
      } else {
        setPaused(true);
      }
    } else {
      if (charIdx > 0) {
        const t = setTimeout(() => {
          setDisplayed(current.slice(0, charIdx - 1));
          setCharIdx((c) => c - 1);
        }, deletingSpeed);
        return () => clearTimeout(t);
      } else {
        setDeleting(false);
        setTextIdx((i) => (i + 1) % texts.length);
      }
    }
  }, [charIdx, deleting, paused, textIdx, texts, typingSpeed, deletingSpeed, pauseMs]);

  return displayed;
}

// ---------------------------------------------------------------------------
// HomeScreen component
// ---------------------------------------------------------------------------

export function HomeScreen({
  input,
  onInputChange,
  onSend,
  onKeyDown,
  loading,
  mode,
  onModeChange,
}: {
  input: string;
  onInputChange: (v: string) => void;
  onSend: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  loading: boolean;
  mode: ReasoningMode;
  onModeChange: (m: ReasoningMode) => void;
}) {
  const typed   = useTypewriter(TYPEWRITER_TEXTS);
  const atLimit = input.length >= MAX_INPUT_CHARS;

  return (
    <div className={styles.homeScreen}>
      <div className={styles.homeContent}>
        <div className={styles.homeLogo} aria-hidden="true">
          <svg width="52" height="52" viewBox="0 0 28 28" fill="none">
            <circle cx="14" cy="14" r="3.5" fill="currentColor" opacity="0.9" />
            <line x1="14" y1="14" x2="4"  y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <line x1="14" y1="14" x2="24" y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <line x1="14" y1="14" x2="4"  y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <line x1="14" y1="14" x2="24" y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <line x1="14" y1="14" x2="14" y2="2"  stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <line x1="14" y1="14" x2="14" y2="26" stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <circle cx="4"  cy="6"  r="2" fill="currentColor" opacity="0.4" />
            <circle cx="24" cy="6"  r="2" fill="currentColor" opacity="0.4" />
            <circle cx="4"  cy="22" r="2" fill="currentColor" opacity="0.4" />
            <circle cx="24" cy="22" r="2" fill="currentColor" opacity="0.4" />
            <circle cx="14" cy="2"  r="2" fill="currentColor" opacity="0.4" />
            <circle cx="14" cy="26" r="2" fill="currentColor" opacity="0.4" />
          </svg>
        </div>
        <h1 className={styles.homeTitle}>Mycelium</h1>
        <div className={styles.homeSubtitle}>
          <span className={styles.homeTyped}>{typed}</span>
          <span className={styles.homeCursor} aria-hidden="true" />
        </div>

        {/* Mode selector — above the input so the user sets intent before typing */}
        <ModeSelector mode={mode} onChange={onModeChange} disabled={loading} />

        <div className={styles.homeInputWrap}>
          <input
            className={styles.homeInput}
            placeholder="Ask anything…"
            value={input}
            onChange={(e) => onInputChange(e.target.value.slice(0, MAX_INPUT_CHARS))}
            onKeyDown={onKeyDown}
            disabled={loading}
            autoFocus
            maxLength={MAX_INPUT_CHARS}
            aria-label="Ask Mycelium a question"
          />
          <button
            className={styles.homeSendButton}
            onClick={onSend}
            disabled={loading || !input.trim()}
            aria-label="Send message"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              <path d="M2 9h14M10 3l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
        {input.length > MAX_INPUT_CHARS * 0.8 && (
          <p
            className={`${styles.charCounter} ${atLimit ? styles.charCounterLimit : ''}`}
            aria-live="polite"
          >
            {input.length} / {MAX_INPUT_CHARS}
          </p>
        )}
      </div>
    </div>
  );
}
