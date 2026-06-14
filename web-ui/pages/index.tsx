import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import LandingPage from '../components/LandingPage';

export default function Page() {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const hasVisited = localStorage.getItem('mycelium-visited');
    setReady(true);
    if (hasVisited) router.replace('/app');
  }, []);

  if (!ready) return null;
  return <LandingPage />;
}
