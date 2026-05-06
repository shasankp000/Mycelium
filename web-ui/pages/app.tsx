import type { AppProps } from 'next/app';
import '../styles/globals.css';

export default function MyceliumApp({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />;
}
