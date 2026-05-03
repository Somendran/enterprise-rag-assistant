import type { FormEvent, MouseEventHandler, RefObject, TouchEventHandler, WheelEventHandler } from 'react';
import { Loader2, Paperclip, Send } from 'lucide-react';
import { STARTER_PROMPTS } from '../../types';
import type { Message } from '../../types';
import { MessageBubble } from './MessageBubble';

interface ChatWindowProps {
  messages: Message[];
  inputTitle: string;
  isQuerying: boolean;
  isDemoMode: boolean;
  queryInputRef: RefObject<HTMLTextAreaElement | null>;
  chatContainerRef: RefObject<HTMLDivElement | null>;
  chatEndRef: RefObject<HTMLDivElement | null>;
  onInputChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  onStarterPrompt: (prompt: string) => void;
  onFeedback: (message: Message, rating: string, reason?: string) => void;
  onAttach: () => void;
  onScroll: () => void;
  onWheel: WheelEventHandler<HTMLDivElement>;
  onTouchStart: TouchEventHandler<HTMLDivElement>;
  onMouseDown: MouseEventHandler<HTMLDivElement>;
}

export function ChatWindow({
  messages,
  inputTitle,
  isQuerying,
  isDemoMode,
  queryInputRef,
  chatContainerRef,
  chatEndRef,
  onInputChange,
  onSubmit,
  onStarterPrompt,
  onFeedback,
  onAttach,
  onScroll,
  onWheel,
  onTouchStart,
  onMouseDown,
}: ChatWindowProps) {
  return (
    <main className="workspace-pane">
      <header className="workspace-header">
        <div>
          <h1>Assistant Workspace</h1>
          <p>Ask questions and inspect confidence before acting.</p>
        </div>
      </header>

      <div
        className="chat-scroll"
        ref={chatContainerRef}
        onScroll={onScroll}
        onWheel={onWheel}
        onTouchStart={onTouchStart}
        onMouseDown={onMouseDown}
      >
        {messages.length === 0 ? (
          <div className="empty-workspace">
            <h2>Ask anything about your indexed workspace.</h2>
            <div className="starter-prompts">
              {STARTER_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="prompt-chip"
                  onClick={() => onStarterPrompt(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-list">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                isDemoMode={isDemoMode}
                onFeedback={onFeedback}
              />
            ))}
            {isQuerying && (
              <div className="chat-row assistant">
                <div className="chat-bubble assistant loading-bubble">
                  <Loader2 className="animate-spin" size={16} /> Thinking...
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        )}
      </div>

      <div className="composer-wrap">
        <form onSubmit={onSubmit} className="composer">
          <textarea
            ref={queryInputRef}
            placeholder="Ask anything about your indexed documents..."
            value={inputTitle}
            onChange={(event) => onInputChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            disabled={isQuerying}
            rows={1}
          />
          <button type="submit" className="send-btn" disabled={isQuerying || !inputTitle.trim()}>
            <Send size={15} />
            Send
          </button>
          <button type="button" className="attach-btn" onClick={onAttach}>
            <Paperclip size={15} />
          </button>
        </form>
      </div>
    </main>
  );
}
