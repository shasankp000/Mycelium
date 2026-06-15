// ---------------------------------------------------------------------------
// Navbar — shared navigation bar used on every page.
//
// Slot API
// ─────────
//   showNavLinks      show the centred text links (How it works / Philosophy)
//   showSidebarToggle show the hamburger button (app shell)
//   showGraphButton   show the Graph pill button (app shell)
//   graphHasNodes     true once the first graph node arrives (controls visibility)
//   graphOpen         true while the graph overlay is open
//   loading           true while the pipeline is running (shows live dot)
//   onSidebarToggle   called when hamburger is clicked
//   onGraphToggle     called when Graph pill is clicked
// ---------------------------------------------------------------------------

import ThemeSwitcher from './ThemeSwitcher';
import styles from '../styles/Navbar.module.css';

const GITHUB_URL = 'https://github.com/shasankp000/Mycelium';

interface NavbarProps {
  /** Show centred nav links (landing page, architecture, philosophy pages) */
  showNavLinks?: boolean;
  /** Show hamburger sidebar-toggle (app shell) */
  showSidebarToggle?: boolean;
  /** Show the Graph pill button (app shell) */
  showGraphButton?: boolean;
  /** Whether the reasoning graph has at least one node yet */
  graphHasNodes?: boolean;
  /** Whether the graph overlay is currently open */
  graphOpen?: boolean;
  /** Whether the pipeline is actively running */
  loading?: boolean;
  /** Callback for sidebar toggle click */
  onSidebarToggle?: () => void;
  /** Callback for graph toggle click */
  onGraphToggle?: () => void;
  /** aria-expanded state for sidebar toggle */
  sidebarExpanded?: boolean;
}

export default function Navbar({
  showNavLinks = false,
  showSidebarToggle = false,
  showGraphButton = false,
  graphHasNodes = false,
  graphOpen = false,
  loading = false,
  onSidebarToggle,
  onGraphToggle,
  sidebarExpanded = false,
}: NavbarProps) {
  return (
    <nav className={styles.nav} aria-label="Main navigation">

      {/* ── Left slot ── */}
      <div className={styles.navLeft}>
        {showSidebarToggle && (
          <button
            className={styles.sidebarToggle}
            onClick={onSidebarToggle}
            aria-label="Toggle history panel"
            aria-expanded={sidebarExpanded}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <rect x="1" y="3"    width="14" height="1.5" rx="0.75" fill="currentColor" />
              <rect x="1" y="7.25" width="10" height="1.5" rx="0.75" fill="currentColor" />
              <rect x="1" y="11.5" width="14" height="1.5" rx="0.75" fill="currentColor" />
            </svg>
          </button>
        )}

        <a href="/" className={styles.wordmark} aria-label="Mycelium home">
          <svg
            className={styles.wordmarkIcon}
            width="18" height="18" viewBox="0 0 28 28" fill="none"
            aria-hidden="true"
          >
            <circle cx="14" cy="14" r="3.5" fill="currentColor" opacity="0.9" />
            <line x1="14" y1="14" x2="4"  y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
            <line x1="14" y1="14" x2="24" y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
            <line x1="14" y1="14" x2="4"  y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
            <line x1="14" y1="14" x2="24" y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
            <line x1="14" y1="14" x2="14" y2="2"  stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
            <line x1="14" y1="14" x2="14" y2="26" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
            <circle cx="4"  cy="6"  r="2" fill="currentColor" opacity="0.35" />
            <circle cx="24" cy="6"  r="2" fill="currentColor" opacity="0.35" />
            <circle cx="4"  cy="22" r="2" fill="currentColor" opacity="0.35" />
            <circle cx="24" cy="22" r="2" fill="currentColor" opacity="0.35" />
            <circle cx="14" cy="2"  r="2" fill="currentColor" opacity="0.35" />
            <circle cx="14" cy="26" r="2" fill="currentColor" opacity="0.35" />
          </svg>
          <span className={styles.wordmarkText}>Mycelium</span>
        </a>
      </div>

      {/* ── Centre slot — nav links (landing / info pages only) ── */}
      {showNavLinks && (
        <div className={styles.navCenter}>
          <a href="/architecture" className={styles.navLink}>How it works</a>
          <a href="/philosophy"   className={styles.navLink}>Philosophy</a>
        </div>
      )}

      {/* ── Right slot ── */}
      <div className={styles.navRight}>

        {/* Graph toggle pill — only visible in app shell after first node */}
        {showGraphButton && graphHasNodes && (
          <button
            className={`${styles.graphBtn} ${graphOpen ? styles.graphBtnActive : ''}`}
            onClick={onGraphToggle}
            aria-label={graphOpen ? 'Close reasoning graph' : 'Open reasoning graph'}
            aria-pressed={graphOpen}
          >
            {loading && <span className={styles.graphBtnDot} aria-hidden="true" />}
            <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <circle cx="7" cy="7" r="2" fill="currentColor" opacity="0.8" />
              <line x1="7" y1="7" x2="2" y2="3" stroke="currentColor" strokeWidth="1.1" opacity="0.5" />
              <line x1="7" y1="7" x2="12" y2="3" stroke="currentColor" strokeWidth="1.1" opacity="0.5" />
              <line x1="7" y1="7" x2="2" y2="11" stroke="currentColor" strokeWidth="1.1" opacity="0.5" />
              <line x1="7" y1="7" x2="12" y2="11" stroke="currentColor" strokeWidth="1.1" opacity="0.5" />
              <circle cx="2"  cy="3"  r="1.4" fill="currentColor" opacity="0.4" />
              <circle cx="12" cy="3"  r="1.4" fill="currentColor" opacity="0.4" />
              <circle cx="2"  cy="11" r="1.4" fill="currentColor" opacity="0.4" />
              <circle cx="12" cy="11" r="1.4" fill="currentColor" opacity="0.4" />
            </svg>
            <span className={styles.graphBtnLabel}>Graph</span>
          </button>
        )}

        <ThemeSwitcher />

        <a
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GitHub repository"
          className={styles.iconLink}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z" />
          </svg>
        </a>
      </div>
    </nav>
  );
}
