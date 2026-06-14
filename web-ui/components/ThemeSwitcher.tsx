'use client';
import { useState, useEffect, useRef } from 'react';

const THEMES = [
  { id: 'amoled',    label: 'AMOLED',     bg: '#000000', ring: '#4f98a3' },
  { id: 'anthropic', label: 'Anthropic',  bg: '#1f1e1a', ring: '#d97757' },
  { id: 'apple',     label: 'Apple',      bg: '#1c1c1e', ring: '#0a84ff' },
  { id: 'arc',       label: 'Arc',        bg: '#13101f', ring: '#9b7ff4' },
] as const;

type ThemeId = typeof THEMES[number]['id'];

export default function ThemeSwitcher() {
  const [active, setActive] = useState<ThemeId>('amoled');
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stored = (localStorage.getItem('mycelium-theme') || 'amoled') as ThemeId;
    setActive(stored);
  }, []);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  function pick(id: ThemeId) {
    document.documentElement.setAttribute('data-theme', id);
    localStorage.setItem('mycelium-theme', id);
    setActive(id);
    setOpen(false);
  }

  return (
    <div ref={ref} style={{ position: 'relative', display: 'inline-flex' }}>
      <button
        onClick={() => setOpen(o => !o)}
        aria-label="Switch theme"
        title="Switch theme"
        style={{
          background: 'none',
          border: '1px solid var(--c-border-default)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--c-text-faint)',
          cursor: 'pointer',
          width: 30,
          height: 30,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'color var(--dur-fast) var(--ease-spring), border-color var(--dur-fast) var(--ease-spring)',
        }}
      >
        {/* Palette icon (inline SVG — no lib needed) */}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="13.5" cy="6.5" r=".5" fill="currentColor"/>
          <circle cx="17.5" cy="10.5" r=".5" fill="currentColor"/>
          <circle cx="8.5"  cy="7.5"  r=".5" fill="currentColor"/>
          <circle cx="6.5"  cy="12.5" r=".5" fill="currentColor"/>
          <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/>
        </svg>
      </button>

      {open && (
        <div style={{
          position: 'absolute',
          top: 'calc(100% + 6px)',
          right: 0,
          background: 'var(--c-bg-1)',
          border: '1px solid var(--c-border-default)',
          borderRadius: 'var(--radius-lg)',
          padding: '8px',
          display: 'flex',
          gap: '8px',
          zIndex: 100,
          boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
          animation: 'themeSwitcherFadeIn 120ms ease',
        }}>
          {THEMES.map(t => (
            <button
              key={t.id}
              title={t.label}
              onClick={() => pick(t.id)}
              style={{
                width: 24,
                height: 24,
                borderRadius: '50%',
                background: t.bg,
                border: 'none',
                cursor: 'pointer',
                outline: active === t.id
                  ? `2px solid var(--c-accent-base)`
                  : `1px solid ${t.ring}`,
                outlineOffset: active === t.id ? '2px' : '0px',
                transition: 'outline var(--dur-fast) var(--ease-spring)',
                boxShadow: `inset 0 0 0 2px ${t.ring}33`,
              }}
              aria-label={`Switch to ${t.label} theme`}
              aria-pressed={active === t.id}
            />
          ))}
        </div>
      )}

      <style>{`
        @keyframes themeSwitcherFadeIn {
          from { opacity: 0; transform: translateY(-4px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
