import { Eye } from 'lucide-react';
import { ConfidenceBar } from './ConfidenceBar';
import type { DocumentChunksResponse, Message, RetrievalDiagnostics, Source } from '../../types';

interface SourcePanelProps {
  latestAssistant?: Message;
  latestDiagnostics: RetrievalDiagnostics | null;
  contextCards: Source[];
  selectedChunks: DocumentChunksResponse | null;
  isLoadingChunks: boolean;
  feedbackStatus: string | null;
  formatPercent: (value?: number | null) => string;
  onOpenSourceChunks: (source: Source) => void;
  onCloseChunks: () => void;
}

export function SourcePanel({
  latestAssistant,
  latestDiagnostics,
  contextCards,
  selectedChunks,
  isLoadingChunks,
  feedbackStatus,
  formatPercent,
  onOpenSourceChunks,
  onCloseChunks,
}: SourcePanelProps) {
  return (
    <aside className="context-pane">
      <h3>Retrieved Context</h3>
      {latestDiagnostics && (
        <div className="diagnostics-panel">
          <div>
            <span>Confidence</span>
            <ConfidenceBar score={latestAssistant?.confidence_score} formatPercent={formatPercent} />
          </div>
          <div>
            <span>Reranker</span>
            <strong>{latestDiagnostics.reranker_applied ? 'Applied' : latestDiagnostics.reranker_skipped_reason || 'Skipped'}</strong>
          </div>
          <div>
            <span>Candidates</span>
            <strong>{latestDiagnostics.candidates_considered}</strong>
          </div>
          <div>
            <span>Retrieval</span>
            <strong>{Math.round(latestDiagnostics.retrieval_ms || 0)} ms</strong>
          </div>
        </div>
      )}
      {contextCards.length === 0 ? (
        <p className="empty-context">Context cards appear here after a response with sources.</p>
      ) : (
        <div className="context-list">
          {contextCards.map((src, index) => (
            <article key={`${src.document}-${src.page}-${index}`} className="context-card">
              <h4>{src.document.replace('.pdf', '')}, Page {src.page}</h4>
              {src.section && <p className="context-section">{src.section}</p>}
              <div className="score-grid">
                <span>Final {formatPercent(src.final_score ?? src.relevance_score)}</span>
                <span>Vector {formatPercent(src.vector_score)}</span>
                <span>Lexical {formatPercent(src.lexical_score)}</span>
                <span>BM25 {formatPercent(src.bm25_score)}</span>
              </div>
              <p className="context-snippet">
                {src.snippet?.trim() || 'Snippet unavailable.'}
              </p>
              <button
                type="button"
                className="source-view-btn"
                onClick={() => onOpenSourceChunks(src)}
                disabled={isLoadingChunks}
              >
                <Eye size={13} />
                Open source chunks
              </button>
            </article>
          ))}
        </div>
      )}
      {feedbackStatus && <p className="feedback-status">{feedbackStatus}</p>}
      {selectedChunks && (
        <div className="chunk-viewer">
          <div className="chunk-viewer-header">
            <h4>{selectedChunks.filename}</h4>
            <button type="button" onClick={onCloseChunks}>Close</button>
          </div>
          {selectedChunks.chunks.length === 0 ? (
            <p className="empty-context">No chunks found for this document.</p>
          ) : (
            selectedChunks.chunks.map((chunk) => (
              <article
                key={chunk.id}
                className={`chunk-item ${selectedChunks.focus_chunk_index === chunk.chunk_index ? 'focused' : ''}`}
              >
                <strong>Page {chunk.page || 'n/a'} · Chunk {chunk.chunk_index}</strong>
                {chunk.section && <span>{chunk.section}</span>}
                <p>{chunk.content}</p>
              </article>
            ))
          )}
        </div>
      )}
    </aside>
  );
}
