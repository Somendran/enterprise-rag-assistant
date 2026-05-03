import { API_BASE, API_KEY, AUTH_TOKEN_STORAGE_KEY, DEMO_SESSION_STORAGE_KEY } from '../types';
import type { StreamDonePayload, StreamEventPayload } from '../types';

export async function ask(question: string): Promise<StreamDonePayload> {
  const response = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: buildStreamHeaders(),
    body: JSON.stringify({ question }),
  });
  if (!response.ok) throw new Error(`Query failed with status ${response.status}`);
  return await response.json() as StreamDonePayload;
}

export function streamQuery(
  question: string,
  onToken: (t: string) => void,
  onDone: () => void,
  onError: (e: Error) => void,
): () => void {
  const controller = new AbortController();
  void readStream(question, controller, onToken, onDone, onError);
  return () => controller.abort();
}

function buildStreamHeaders(): Record<string, string> {
  const authToken = window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) || '';
  const demoToken = window.localStorage.getItem(DEMO_SESSION_STORAGE_KEY) || '';
  return {
    'Content-Type': 'application/json',
    ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
    ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
    ...(demoToken ? { 'X-Demo-Session-Id': demoToken } : {}),
  };
}

async function readStream(
  question: string,
  controller: AbortController,
  onToken: (t: string) => void,
  onDone: () => void,
  onError: (e: Error) => void,
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers: buildStreamHeaders(),
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });

    if (response.status === 401) {
      throw new Error('Login expired or API key missing. Sign in again.');
    }
    if (!response.ok || !response.body) {
      throw new Error(`Streaming request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() || '';

      for (const eventBlock of events) {
        const lines = eventBlock.split('\n');
        const eventName = lines.find((line) => line.startsWith('event:'))?.replace('event:', '').trim();
        const dataLine = lines.find((line) => line.startsWith('data:'))?.replace('data:', '').trim();
        if (!eventName || !dataLine) continue;
        const payload = JSON.parse(dataLine) as StreamEventPayload;
        if (eventName === 'chunk') onToken(String(payload.text || ''));
        if (eventName === 'done') onDone();
        if (eventName === 'error') throw new Error(String(payload.detail || 'Streaming query failed'));
      }
    }
  } catch (error) {
    if (controller.signal.aborted) return;
    onError(error instanceof Error ? error : new Error('Streaming query failed'));
  }
}
