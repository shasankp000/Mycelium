import { useRouter } from 'next/router';
import ThemeSwitcher from './ThemeSwitcher';
import styles from '../styles/LandingPage.module.css';

const DIFFERENTIATORS = [
  {
    eyebrow: 'The Gatekeeper',
    heading: 'Before answering, Mycelium asks: can this question even have an objective answer?',
    body: 'Objective questions enter a 6-layer evidence-grounded reasoning pipeline and receive a definitive answer. Value-laden questions receive multi-perspective evidence — no hidden opinion embedded as fact. Manipulation attempts are refused and explained.',
    note: 'Most AI systems skip this step entirely. Mycelium makes it the foundation.',
  },
  {
    eyebrow: 'Refutation-First Evidence',
    heading: 'Evidence retrieval is designed to disprove the claim first, then confirm it.',
    body: 'The system actively searches for counterevidence before supporting evidence. For universal claims, a single strong counterexample halts retrieval immediately. This structurally prevents confirmation bias at the retrieval layer — confidence scores reflect what evidence actually shows.',
    note: 'Confidence scores mean something here.',
  },
  {
    eyebrow: 'Semantic Reasoning',
    heading: 'Mycelium reasons over structured semantic claims, not raw text.',
    body: 'Every claim is decomposed into a PredicateFrame: subject, relation, object, negation, scope, falsifiability, and provenance. Reasoning propagates through a directed acyclic graph of structured predicates. The full reasoning chain is visible — what was concluded, what evidence was used, what was considered and rejected.',
    note: 'Not text autocomplete. Actual reasoning.',
  },
];

export default function LandingPage() {
  const router = useRouter();
  const version = process.env.NEXT_PUBLIC_APP_VERSION || 'dev';

  function enter() {
    localStorage.setItem('mycelium-visited', '1');
    router.push('/app');
  }

  return (
    <div className={styles.page}>
      {/* Navbar */}
      <nav className={styles.nav}>
        <span className={styles.navWordmark}>Mycelium</span>
        <div className={styles.navActions}>
          <ThemeSwitcher />
          <a
            href="https://github.com/shasankp000/Mycelium"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub repository"
            style={{ color: 'var(--c-text-faint)', display: 'flex', alignItems: 'center' }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z"/>
            </svg>
          </a>
        </div>
      </nav>

      {/* Hero */}
      <main className={styles.hero}>
        <div className={`${styles.heroBg} landingHeroBg`} />

        <div className={styles.statusPill}>
          <span className={styles.statusDot} />
          Research Preview
        </div>

        <h1 className={`${styles.headline} landingHeadline`}>Mycelium</h1>

        <p className={styles.tagline}>
          Multi-path semantic reasoning. Built to find truth where it exists, and admit when it doesn&apos;t.
        </p>

        <button className={`${styles.cta} landingCta`} onClick={enter}>
          Start reasoning →
        </button>

        <div className={styles.modePills}>
          {['Quick', 'Smart', 'Researcher'].map(m => (
            <span key={m} className={`${styles.modePill} landingModePill`}>
              ○ {m}
            </span>
          ))}
        </div>
      </main>

      {/* Differentiators */}
      <section className={styles.differentiators} aria-label="What makes Mycelium different">
        <p className={styles.sectionLabel}>What makes this different</p>
        <div className={styles.differentiatorGrid}>
          {DIFFERENTIATORS.map((d) => (
            <div key={d.eyebrow} className={styles.differentiatorItem}>
              <span className={styles.differentiatorEyebrow}>{d.eyebrow}</span>
              <h2 className={styles.differentiatorHeading}>{d.heading}</h2>
              <p className={styles.differentiatorBody}>{d.body}</p>
              <p className={styles.differentiatorNote}>{d.note}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Philosophy */}
      <section id="philosophy" className={styles.philosophy} aria-label="Philosophy">
        <div className={styles.philosophyInner}>
          <span className={styles.philosophyLabel}>On epistemic honesty</span>
          <p className={styles.philosophyText}>
            David Hume observed in 1739 that you cannot derive <em>ought</em> from <em>is</em> — facts and values are fundamentally different categories.
            Most AI systems blur this line constantly, presenting value judgments as factual outputs.
          </p>
          <p className={styles.philosophyText}>
            Mycelium treats this boundary as an architectural constraint. Value-laden questions route to a multi-perspective evidence engine that presents the full option space. The system does not have opinions on questions that require value judgments. It has evidence, and it shows it to you.
          </p>
          <blockquote className={styles.philosophyQuote}>
            &ldquo;I will find truth where it exists. I will admit when it doesn&apos;t. I will not pretend my preferences are your facts.&rdquo;
          </blockquote>
        </div>
      </section>

      {/* Footer */}
      <footer className={styles.footer}>
        <span>v{version}</span>
        <span className={styles.footerDot}>·</span>
        <span>web-ui-prototype</span>
        <span className={styles.footerDot}>·</span>
        <span>MIT</span>
        <span className={styles.footerDot}>·</span>
        <a href="#philosophy" className={styles.footerLink}>Philosophy</a>
        <span className={styles.footerDot}>·</span>
        <a
          href="https://github.com/shasankp000/Mycelium"
          target="_blank"
          rel="noopener noreferrer"
          className={styles.footerLink}
        >
          GitHub
        </a>
      </footer>
    </div>
  );
}
