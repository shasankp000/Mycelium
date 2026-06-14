import type { AppProps } from 'next/app';
import { useRouter } from 'next/router';
import { useEffect, useRef, useState } from 'react';
import '../styles/globals.css';

export default function MyceliumApp({ Component, pageProps }: AppProps) {
  const router = useRouter();
  const [fading, setFading] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleStart = () => setFading(true);
    const handleDone  = () => setFading(false);
    router.events.on('routeChangeStart',    handleStart);
    router.events.on('routeChangeComplete', handleDone);
    router.events.on('routeChangeError',    handleDone);
    return () => {
      router.events.off('routeChangeStart',    handleStart);
      router.events.off('routeChangeComplete', handleDone);
      router.events.off('routeChangeError',    handleDone);
    };
  }, [router.events]);

  return (
    <div
      ref={wrapRef}
      style={{
        opacity: fading ? 0 : 1,
        transition: 'opacity 180ms ease',
      }}
    >
      <Component {...pageProps} />
    </div>
  );
}
