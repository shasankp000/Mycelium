import { useEffect, useRef, useState, KeyboardEvent } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import clsx from 'clsx';
import styles from '../styles/ChatPage.module.css';
import Navbar from './Navbar';
// -----------------------------------------------------------------------
// PLACEHOLDER API ENDPOINT
// Swap this out for the real backend once it exists. The page sends the
// raw user message as plain text and expects a plain-text reply back.
// -----------------------------------------------------------------------
const CHAT_API_ENDPOINT = 'https://example.com/api/chat'; // TODO: replace with real endpoint

type Role = 'user' | 'assistant';

interface Message {
  id: string;
  role: Role;
  content: string;
  isError?: boolean;
}

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function getGreeting() {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good morning.';
  if (hour < 18) return 'Good afternoon.';
  return 'Good evening.';
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<'light' | 'dark'>('light');

  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Initialise mode: stored preference, falling back to system preference.
  useEffect(() => {
    const stored = localStorage.getItem('chat-ui-mode');
    if (stored === 'light' || stored === 'dark') {
      setMode(stored);
    } else if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
      setMode('dark');
    }
  }, []);

  function toggleMode() {
    setMode((prev) => {
      const next = prev === 'light' ? 'dark' : 'light';
      localStorage.setItem('chat-ui-mode', next);
      return next;
    });
  }

  // Auto-scroll to the latest message.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Auto-resize the textarea as the user types.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  async function handleSend() {
    const text = input.trim();
    if (!text || isLoading) return;

    const userMessage: Message = { id: makeId(), role: 'user', content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Sent as plain text, per the placeholder contract. Swap
      // CHAT_API_ENDPOINT (and this request shape) for the real API later.
      const response = await axios.post(CHAT_API_ENDPOINT, text, {
        headers: { 'Content-Type': 'text/plain' },
        transformResponse: [(data) => data], // keep raw text, skip JSON parsing
      });

      const reply = typeof response.data === 'string'
        ? response.data
        : String(response.data ?? '');

      setMessages((prev) => [
        ...prev,
        { id: makeId(), role: 'assistant', content: reply || '(empty response)' },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role: 'assistant',
          isError: true,
          content:
            "Couldn't reach the assistant. This page currently points at a placeholder endpoint (" +
            CHAT_API_ENDPOINT +
            ') — connect it to a real backend to get live responses.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className={styles.root} data-mode={mode}>
     <Navbar showSidebarToggle={false} showGraphButton={false} onGraphToggle={() => {}} />

      <div className={styles.scrollArea} ref={scrollRef}>
        <div className={styles.column}>
          {messages.length === 0 && !isLoading && (
            <div className={styles.emptyState}>
              <div className={styles.emptyTitle}>{getGreeting()}</div>
              <div className={styles.emptySubtitle}>What would you like to talk about today?</div>
            </div>
          )}

          <AnimatePresence initial={false}>
            {messages.map((m) => (
              <motion.div
                key={m.id}
                className={clsx(styles.messageRow, m.role === 'user' ? styles.rowUser : styles.rowAssistant)}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
              >
                <div
                  className={clsx(
                    styles.bubble,
                    m.role === 'user' ? styles.bubbleUser : styles.bubbleAssistant,
                    m.isError && styles.bubbleError
                  )}
                >
                  {m.content}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {isLoading && (
            <motion.div
              className={clsx(styles.messageRow, styles.rowAssistant)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className={clsx(styles.bubble, styles.bubbleAssistant, styles.typingBubble)}>
                <span className={styles.dot} />
                <span className={styles.dot} />
                <span className={styles.dot} />
              </div>
            </motion.div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      <div className={styles.inputBar}>
        <div className={styles.inputColumn}>
          <textarea
            ref={textareaRef}
            className={styles.textarea}
            placeholder="Message..."
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className={styles.sendButton}
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            aria-label="Send message"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 19V5M5 12l7-7 7 7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}