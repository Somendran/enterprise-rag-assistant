import ReactMarkdown from 'react-markdown';
import { AlertTriangle, Heart, ThumbsDown } from 'lucide-react';
import type { Message } from '../../types';

interface MessageBubbleProps {
  message: Message;
  isDemoMode: boolean;
  onFeedback: (message: Message, rating: string, reason?: string) => void;
}

export function MessageBubble({ message, isDemoMode, onFeedback }: MessageBubbleProps) {
  return (
    <div className={`chat-row ${message.role}`}>
      <div className={`chat-bubble ${message.role}`}>
        {message.role === 'assistant' ? <ReactMarkdown>{message.content}</ReactMarkdown> : <p>{message.content}</p>}
        {message.role === 'assistant' && message.content && !isDemoMode && (
          <div className="feedback-row">
            <button type="button" onClick={() => onFeedback(message, 'helpful')}>
              <Heart size={13} />
              Helpful
            </button>
            <button type="button" onClick={() => onFeedback(message, 'not_helpful')}>
              <ThumbsDown size={13} />
              Not helpful
            </button>
            <button type="button" onClick={() => onFeedback(message, 'wrong_source', 'wrong_source')}>
              <AlertTriangle size={13} />
              Wrong source
            </button>
            <button type="button" onClick={() => onFeedback(message, 'missing_info', 'missing_info')}>
              Missing info
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
