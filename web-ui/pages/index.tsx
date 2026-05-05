import Head from 'next/head';
import { useState } from 'react';
import axios from 'axios';
import styles from '@/styles/Home.module.css';

interface Message {
  role: 'user' | 'system';
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSend() {
    if (!input.trim()) return;
    const userMsg: Message = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post('http://localhost:8000/api/query', {
        text: userMsg.content,
      });

      const data = res.data;
      const summaryLines: string[] = [];

      if (data.layer0?.route) {
        summaryLines.push(`Layer0 route: ${data.layer0.route}`);
      }
      if (data.routing?.classification) {
        summaryLines.push(`Routing: ${data.routing.classification}`);
      }
      if (data.expert_decision?.decision_type) {
        summaryLines.push(
          `Expert: ${data.expert_decision.decision_type} → ${
            (data.expert_decision.selected_experts || []).join(', ') || 'none'
          }`,
        );
      }
      if (data.phase3?.validation_decision?.result_class) {
        summaryLines.push(
          `Validation: ${data.phase3.validation_decision.result_class}`,
        );
      }

      const systemMsg: Message = {
        role: 'system',
        content:
          summaryLines.join('\n') ||
          'Received response from Mycelium (see browser console for full JSON).',
      };
      setMessages((prev) => [...prev, systemMsg]);
      console.debug('Full Mycelium response', data);
    } catch (err) {
      console.error(err);
      const systemMsg: Message = {
        role: 'system',
        content: 'Error contacting backend. Is the FastAPI server running on :8000?',
      };
      setMessages((prev) => [...prev, systemMsg]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className={styles.container}>
      <Head>
        <title>Mycelium Console</title>
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>Mycelium Interactive Console</h1>
        <p className={styles.subtitle}>
          Type a query to send it through Layer 0 → Routing → Experts → Validation.
        </p>

        <div className={styles.chatWindow}>
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={
                m.role === 'user' ? styles.userMessage : styles.systemMessage
              }
            >
              <span className={styles.messageRole}>
                {m.role === 'user' ? 'You' : 'Mycelium'}
              </span>
              <pre className={styles.messageContent}>{m.content}</pre>
            </div>
          ))}
          {loading && <div className={styles.loading}>Thinking…</div>}
        </div>

        <div className={styles.inputRow}>
          <input
            className={styles.input}
            placeholder="Ask Mycelium something…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className={styles.sendButton}
            onClick={handleSend}
            disabled={loading}
          >
            Send
          </button>
        </div>
      </main>
    </div>
  );
}
