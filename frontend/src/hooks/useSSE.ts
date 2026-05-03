import { useCallback, useRef, useState } from 'react';
import { streamQuery } from '../api/query';

export type SSEState = 'idle' | 'connecting' | 'streaming' | 'done' | 'error';

export interface UseSSEResult {
  state: SSEState;
  tokens: string;
  error: string | null;
  start: (question: string) => void;
  reset: () => void;
}

export function useSSE(): UseSSEResult {
  const [state, setState] = useState<SSEState>('idle');
  const [tokens, setTokens] = useState('');
  const [error, setError] = useState<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  const reset = useCallback(() => {
    cleanupRef.current?.();
    cleanupRef.current = null;
    setState('idle');
    setTokens('');
    setError(null);
  }, []);

  const start = useCallback((question: string) => {
    cleanupRef.current?.();
    setState('connecting');
    setTokens('');
    setError(null);
    cleanupRef.current = streamQuery(
      question,
      (token) => {
        setState('streaming');
        setTokens((current) => `${current}${token}`);
      },
      () => {
        setState('done');
        cleanupRef.current = null;
      },
      (streamError) => {
        setState('error');
        setError(streamError.message);
        cleanupRef.current = null;
      },
    );
  }, []);

  return { state, tokens, error, start, reset };
}
