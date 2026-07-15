import Head from 'next/head';
import ChatPage from '../components/ChatPage';

export default function Chat() {
  return (
    <>
      <Head>
        <title>Chat · Mycelium</title>
        <meta name="description" content="Chat with Mycelium" />
      </Head>
      <ChatPage />
    </>
  );
}