import { useCallback, useMemo, useState } from 'react';
import { useSSE } from './useSSE';
import type { Message } from '../types';

export interface UseChatResult {
  messages: Message[];
  sessionId: string | null;
  isLoading: boolean;
  sendMessage: (content: string) => void;
  clearChat: () => void;
}

export function useChat(): UseChatResult {
  const stream = useSSE();
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId] = useState<string | null>(null);
  const [assistantId, setAssistantId] = useState<string | null>(null);

  const sendMessage = useCallback((content: string) => {
    const trimmed = content.trim();
    if (!trimmed) return;
    const userId = Date.now().toString();
    const nextAssistantId = (Date.now() + 1).toString();
    setAssistantId(nextAssistantId);
    setMessages((current) => [
      ...current,
      { id: userId, role: 'user', content: trimmed, timestamp: Date.now() },
      { id: nextAssistantId, role: 'assistant', content: '', timestamp: Date.now() },
    ]);
    stream.start(trimmed);
  }, [stream]);

  const clearChat = useCallback(() => {
    stream.reset();
    setMessages([]);
    setAssistantId(null);
  }, [stream]);

  const renderedMessages = useMemo(() => (
    assistantId
      ? messages.map((message) => (
        message.id === assistantId ? { ...message, content: stream.tokens } : message
      ))
      : messages
  ), [assistantId, messages, stream.tokens]);

  return {
    messages: renderedMessages,
    sessionId,
    isLoading: stream.state === 'connecting' || stream.state === 'streaming',
    sendMessage,
    clearChat,
  };
}
