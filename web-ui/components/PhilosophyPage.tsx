import React, { useEffect } from 'react';
import Link from 'next/link';
import Navbar from './Navbar';
import styles from '../styles/PhilosophyPage.module.css';

export default function PhilosophyPage() {
  return (
    <div className={styles.page}>

      {/* ── Shared navbar ── */}
      <Navbar showNavLinks />

      {/* ── Hero ── */}
      <header className={styles.hero}>
        <p className={styles.eyebrow}>Philosophy</p>
        <h1 className={styles.title}>On epistemic honesty</h1>
        <p className={styles.subtitle}>
          Why the boundary between facts and values is not a feature — it is the foundation.
        </p>
      </header>

      {/* ── Body ── */}
      <main className={styles.main}>

        <section className={styles.section}>
          <h2 className={styles.heading}>The Is-Ought problem</h2>
          <p className={styles.body}>
            In 1739, David Hume observed something that most systems built since have ignored:
            you cannot derive what <em>ought</em> to be from what <em>is</em>. Facts and values
            are fundamentally different categories. No chain of purely factual premises can
            logically produce a value judgment as its conclusion.
          </p>
          <p className={styles.body}>
            This is not a philosophical technicality. It is the source of most bias in AI
            systems. When a model presents a value judgment — a preference, a political
            leaning, a moral conclusion — as if it were a factual output, it has committed
            Hume&apos;s error at scale. The user receives an opinion dressed as a finding.
          </p>
        </section>

        <section className={styles.section}>
          <h2 className={styles.heading}>How Mycelium treats this boundary</h2>
          <p className={styles.body}>
            Mycelium treats the is-ought boundary as an architectural constraint, not a
            guideline. Every query is classified before any reasoning begins. Objective
            questions — those with a factual answer that evidence can settle — enter the
            full reasoning pipeline and receive a definitive, evidence-grounded response.
          </p>
          <p className={styles.body}>
            Value-laden questions are routed differently. The system identifies the
            underlying value assumptions, surfaces the full option space, and presents
            evidence for each position. It does not pick a side. A question like
            &ldquo;Is nuclear energy safe?&rdquo; contains both an objective component
            (accident statistics, mortality data) and a value component (what counts as
            &ldquo;safe enough&rdquo; and for whom). Mycelium separates these structurally
            rather than blending them silently.
          </p>
        </section>

        <section className={styles.section}>
          <h2 className={styles.heading}>On manipulation</h2>
          <p className={styles.body}>
            A third category exists beyond objective and value-laden questions: manipulation
            attempts. These are queries designed not to seek an answer but to force a
            predetermined conclusion — instruction overrides, cherry-pick requests,
            conclusion-forcing prompts. Most systems comply with these because compliance
            is the path of least resistance for a text predictor.
          </p>
          <p className={styles.body}>
            Mycelium refuses them. Not because of a content policy, but because the
            architecture detects them before reasoning begins and treats them as a separate
            class. The refusal includes an explanation and, where possible, a reframe of
            the honest version of the question.
          </p>
        </section>

        <section className={styles.section}>
          <h2 className={styles.heading}>The no-bullshit commitment</h2>
          <p className={styles.body}>
            This is the plainest statement of what Mycelium is for:
          </p>
          <blockquote className={styles.blockquote}>
            I will find truth where it exists.<br />
            I will admit when it doesn&rsquo;t.<br />
            I will not pretend my preferences are your facts.<br />
            I will not be manipulated into bias.
          </blockquote>
          <p className={styles.body}>
            These are not marketing claims. Each one corresponds to a concrete architectural
            decision: refutation-first evidence retrieval, explicit uncertainty quantification,
            structural separation of FACTUAL and NORMATIVE predicates, and L0 manipulation
            detection. The philosophy is implemented, not declared.
          </p>
        </section>

        <section className={styles.section}>
          <h2 className={styles.heading}>Why this matters</h2>
          <p className={styles.body}>
            The question Mycelium is designed to answer is not &ldquo;what is the most
            plausible continuation of this text?&rdquo; It is: &ldquo;what does the evidence
            actually show, and where does the evidence run out?&rdquo; These are different
            questions. The first produces fluent, confident-sounding output regardless of
            truth. The second produces honest output that is sometimes less satisfying and
            always more useful.
          </p>
          <p className={styles.body}>
            Epistemic honesty is not a constraint on usefulness. It is the precondition for it.
          </p>
        </section>

        {/* ── CTA ── */}
        <div className={styles.cta}>
          <Link href="/architecture" className={styles.ctaPrimary}>See how it works →</Link>
          <Link href="/" className={styles.ctaBack}>← Back to home</Link>
        </div>

      </main>

      {/* ── Footer ── */}
      <footer className={styles.footer}>
        <span>vdev</span>
        <span>·</span>
        <span>web-ui-prototype</span>
        <span>·</span>
        <span>MIT</span>
      </footer>
    </div>
  );
}
