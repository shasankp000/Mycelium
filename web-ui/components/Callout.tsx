import styles from '../styles/Callout.module.css';

type CalloutVariant = 'insight' | 'caution' | 'note';

interface CalloutProps {
  variant?: CalloutVariant;
  heading?: string;
  children: React.ReactNode;
}

export default function Callout({ variant = 'insight', heading, children }: CalloutProps) {
  return (
    <aside className={`${styles.callout} ${styles[variant]}`}>
      {heading && <strong className={styles.heading}>{heading}</strong>}
      <div className={styles.body}>{children}</div>
    </aside>
  );
}
