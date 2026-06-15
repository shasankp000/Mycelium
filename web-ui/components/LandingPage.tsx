import { useEffect } from 'react';
import { useRouter } from 'next/router';
import Navbar from './Navbar';
import styles from '../styles/LandingPage.module.css';

/* ── Scroll-reveal hook (bidirectional) ───────────────────── */
function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll<HTMLElement>('.' + styles.reveal);
    if (!els.length) return;

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            (e.target as HTMLElement).classList.add(styles.visible);
          } else {
            (e.target as HTMLElement).classList.remove(styles.visible);
          }
        });
      },
      { threshold: 0.12 }
    );

    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);
}

/* ── Data ──────────────────────────────────────────────── */
const HOW_IT_WORKS = [
  {
    step: '01',
    heading: 'You ask a question',
    body: 'Any question — factual, complex, contested. Mycelium does not pre-judge what you are allowed to ask.',
  },
  {
    step: '02',
    heading: 'Mycelium classifies it',
    body: 'Before reasoning begins, the system determines whether your question has an objective answer, requires weighing values, or is attempting to manipulate the output. Each type gets a different, appropriate treatment.',
  },
  {
    step: '03',
    heading: 'Evidence is gathered — against the claim first',
    body: 'Unlike most AI, Mycelium actively searches for counterevidence before looking for supporting evidence. This is not a setting. It is how the system is built.',
  },
  {
    step: '04',
    heading: 'You get an honest answer',
    body: 'Objective questions get a direct answer with traceable evidence. Value-laden questions get the full picture — multiple perspectives, each with their evidence — and no hidden opinion dressed as fact.',
  },
];

const DIFFERENTIATORS = [
  {
    eyebrow: 'The Gatekeeper',
    heading: 'Every question is screened before reasoning starts',
    body: 'Mycelium checks three things before it thinks: Can this question have a factual answer? Does it depend on values and priorities? Is it trying to manipulate the output? Most AI tools never ask these questions — they just respond.',
    note: 'This alone prevents most of the subtle misinformation AI systems produce.',
  },
  {
    eyebrow: 'Evidence that can say no',
    heading: 'Counterevidence is retrieved before supporting evidence',
    body: 'When you ask something, the system searches for reasons it might be wrong before reasons it might be right. Confidence scores reflect the actual balance of evidence — not how fluently the model can produce text about the topic.',
    note: 'If the evidence is weak, Mycelium says so.',
  },
  {
    eyebrow: 'Transparent reasoning',
    heading: 'You can see how every conclusion was reached',
    body: 'Each answer comes with a full reasoning trace — what claims were evaluated, what evidence was used, what was considered and rejected. You are never just trusting an output. You can follow the logic yourself.',
    note: 'Every step is visible. Nothing is hidden.',
  },
];

const MODES = [
  {
    name: 'Quick',
    desc: 'Fast answers for straightforward factual questions. Lightweight reasoning with honest uncertainty flagging.',
  },
  {
    name: 'Smart',
    desc: 'Full multi-path reasoning for complex questions. Balances depth with speed. The default for most questions.',
  },
  {
    name: 'Researcher',
    desc: 'Maximum depth. Expanded evidence search, multiple source types, and the most detailed reasoning trace. For questions that really matter.',
  },
];

/* ── Component ─────────────────────────────────────────────── */
export default function LandingPage() {
  const router = useRouter();
  const version = process.env.NEXT_PUBLIC_APP_VERSION || 'dev';
  useReveal();

  function enter() {
    router.push('/app');
  }

  const R = styles.reveal;

  return (
    <div className={styles.page}>

      {/* ── Shared navbar (showNavLinks = true for landing page) ── */}
      <Navbar showNavLinks />

      {/* ── Hero ── */}
      <main className={styles.hero}>
        <div className={`${styles.heroBg} landingHeroBg`} />

        <div className={`${styles.statusPill} ${styles.fadeUp}`} style={{ animationDelay: '0ms' }}>
          <span className={styles.statusDot} />
          Research Preview
        </div>

        <h1
          className={`${styles.headline} ${styles.fadeUp} landingHeadline`}
          style={{ animationDelay: '80ms' }}
        >
          Mycelium
        </h1>

        <p className={`${styles.tagline} ${styles.fadeUp}`} style={{ animationDelay: '160ms' }}>
          An AI reasoning system that finds truth where it exists — and admits when it doesn&apos;t.
        </p>

        <div className={`${styles.heroActions} ${styles.fadeUp}`} style={{ animationDelay: '240ms' }}>
          <button className={`${styles.cta} landingCta`} onClick={enter}>
            Start reasoning →
          </button>
          <a href="/architecture" className={styles.ctaSecondary}>
            See how it works ↓
          </a>
        </div>

        <div className={`${styles.modePills} ${styles.fadeUp}`} style={{ animationDelay: '320ms' }}>
          {['Quick', 'Smart', 'Researcher'].map(m => (
            <span key={m} className={`${styles.modePill} landingModePill`}>
              ○ {m}
            </span>
          ))}
        </div>
      </main>

      {/* ── How it works ── */}
      <section id="how-it-works" className={styles.section} aria-label="How it works">
        <div className={styles.sectionInner}>
          <p className={`${styles.sectionEyebrow} ${R}`}>How it works</p>
          <h2 className={`${styles.sectionHeading} ${R}`}>
            Reasoning you can actually follow
          </h2>
          <p className={`${styles.sectionSubheading} ${R}`}>
            Most AI tools give you an answer. Mycelium shows you the work.
          </p>

          <div className={styles.stepsGrid}>
            {HOW_IT_WORKS.map((s, i) => (
              <div
                key={s.step}
                className={`${styles.stepItem} ${R}`}
                style={{ transitionDelay: `${i * 80}ms` }}
              >
                <span className={styles.stepNumber}>{s.step}</span>
                <h3 className={styles.stepHeading}>{s.heading}</h3>
                <p className={styles.stepBody}>{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Differentiators ── */}
      <section className={styles.section} aria-label="What makes Mycelium different">
        <div className={styles.sectionInner}>
          <p className={`${styles.sectionEyebrow} ${R}`}>What makes this different</p>
          <h2 className={`${styles.sectionHeading} ${R}`}>
            Built around honesty, not confidence
          </h2>
          <p className={`${styles.sectionSubheading} ${R}`}>
            Sounding certain is easy. Being correct requires a different kind of design.
          </p>

          <div className={styles.differentiatorGrid}>
            {DIFFERENTIATORS.map((d, i) => (
              <div
                key={d.eyebrow}
                className={`${styles.differentiatorItem} ${R}`}
                style={{ transitionDelay: `${i * 100}ms` }}
              >
                <span className={styles.differentiatorEyebrow}>{d.eyebrow}</span>
                <h3 className={styles.differentiatorHeading}>{d.heading}</h3>
                <p className={styles.differentiatorBody}>{d.body}</p>
                <p className={styles.differentiatorNote}>{d.note}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Modes ── */}
      <section className={styles.section} aria-label="Reasoning modes">
        <div className={styles.sectionInner}>
          <p className={`${styles.sectionEyebrow} ${R}`}>Three modes</p>
          <h2 className={`${styles.sectionHeading} ${R}`}>
            Match the depth to the question
          </h2>

          <div className={styles.modesGrid}>
            {MODES.map((m, i) => (
              <div
                key={m.name}
                className={`${styles.modeCard} ${R}`}
                style={{ transitionDelay: `${i * 80}ms` }}
              >
                <span className={styles.modeName}>○ {m.name}</span>
                <p className={styles.modeDesc}>{m.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Philosophy ── */}
      <section id="philosophy" className={styles.section} aria-label="Philosophy">
        <div className={styles.sectionInner}>
          <div className={`${styles.philosophyInner} ${R}`}>
            <span className={styles.philosophyLabel}>On epistemic honesty</span>
            <p className={styles.philosophyText}>
              The philosopher David Hume pointed out in 1739 that facts and values are fundamentally different things. You cannot derive what <em>ought</em> to be from what <em>is</em>. Most AI systems constantly blur this line — presenting opinions, predictions, and value judgments with the same confident tone as verified facts.
            </p>
            <p className={styles.philosophyText}>
              Mycelium treats this boundary as a hard architectural rule. Questions that depend on values — political, ethical, personal — are handled differently from questions that have factual answers. For value-laden questions, the system presents the full landscape of perspectives and evidence, and lets you decide. It does not have preferences it will sneak into your answer.
            </p>
            <blockquote className={styles.philosophyQuote}>
              &ldquo;I will find truth where it exists. I will admit when it doesn&apos;t. I will not pretend my preferences are your facts.&rdquo;
            </blockquote>
            <a href="/philosophy" className={styles.philosophyReadMore}>Read more about our philosophy →</a>
          </div>
        </div>
      </section>

      {/* ── Final CTA ── */}
      <section className={styles.ctaSection} aria-label="Get started">
        <div className={`${styles.ctaSectionInner} ${R}`}>
          <h2 className={styles.ctaSectionHeading}>Ready to ask something that matters?</h2>
          <p className={styles.ctaSectionSub}>
            Mycelium is in active development. The reasoning engine is real. Expect rough edges.
          </p>
          <button className={`${styles.cta} ${styles.ctaLarge} landingCta`} onClick={enter}>
            Start reasoning →
          </button>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className={styles.footer}>
        <span>v{version}</span>
        <span className={styles.footerDot}>·</span>
        <span>web-ui-prototype</span>
        <span className={styles.footerDot}>·</span>
        <span>MIT</span>
        <span className={styles.footerDot}>·</span>
        <a href="/philosophy" className={styles.footerLink}>Philosophy</a>
        <span className={styles.footerDot}>·</span>
        <a href="/architecture" className={styles.footerLink}>How it works</a>
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
