import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import styles from '../styles/NotFoundPage.module.css';

const THEME_KEY = 'mycelium-theme';

export default function NotFoundPage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const saved = sessionStorage.getItem(THEME_KEY);
    if (saved) document.documentElement.setAttribute('data-theme', saved);
    setMounted(true);
  }, []);

  return (
    <div className={styles.page}>
      <nav className={styles.nav}>
        <Link href="/" className={styles.navWordmark}>Mycelium</Link>
      </nav>

      <main className={`${styles.main} ${mounted ? styles.visible : ''}`}>
        <p className={styles.code}>404</p>
        <h1 className={styles.title}>Path not found</h1>
        <p className={styles.body}>
          This route doesn&rsquo;t exist — or not yet. Unlike the reasoning pipeline,
          there&rsquo;s no evidence to retrieve here.
        </p>
        <div className={styles.actions}>
          <Link href="/" className={styles.btnPrimary}>Back to home</Link>
          <Link href="/architecture" className={styles.btnSecondary}>How it works</Link>
        </div>
      </main>

      <footer className={styles.footer}>
        <span>vdev · web-ui-prototype · MIT</span>
      </footer>
    </div>
  );
}
