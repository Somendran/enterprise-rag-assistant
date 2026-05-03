import type { ChangeEvent, DragEvent, RefObject } from 'react';
import { Loader2, RotateCcw, UploadCloud } from 'lucide-react';

interface UploadZoneProps {
  fileInputRef: RefObject<HTMLInputElement | null>;
  isDragActive: boolean;
  isUploading: boolean;
  isResetting: boolean;
  uploadStatus: string | null;
  uploadSteps: string[];
  isDemoMode: boolean;
  onDragOver: (event: DragEvent<HTMLDivElement>) => void;
  onDragLeave: (event: DragEvent<HTMLDivElement>) => void;
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
  onFileUpload: (event: ChangeEvent<HTMLInputElement>) => void;
  onReset: () => void;
}

export function UploadZone({
  fileInputRef,
  isDragActive,
  isUploading,
  isResetting,
  uploadStatus,
  uploadSteps,
  isDemoMode,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileUpload,
  onReset,
}: UploadZoneProps) {
  return (
    <div
      className={`upload-dropzone ${isDragActive ? 'drag-active' : ''}`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <h2>Knowledge Base</h2>
      <div className="drop-inner">
        <UploadCloud size={24} />
        <p>
          Drag and drop PDFs here,
          <br />
          or <button type="button" className="inline-link" onClick={() => fileInputRef.current?.click()}>browse</button>.
        </p>
      </div>

      <input
        type="file"
        accept=".pdf"
        multiple
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={onFileUpload}
      />

      {uploadStatus && (
        <div className={`status-note ${uploadStatus.includes('Failed') || uploadStatus.includes('API key') ? 'error' : 'success'}`}>
          {uploadStatus}
        </div>
      )}

      {uploadSteps.length > 0 && (
        <ol className="upload-steps">
          {uploadSteps.map((step, index) => (
            <li key={`${step}-${index}`}>{step}</li>
          ))}
        </ol>
      )}

      {!isDemoMode && (
        <button
          type="button"
          className="reset-kb-btn"
          onClick={onReset}
          disabled={isResetting || isUploading}
        >
          {isResetting ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
          {isResetting ? 'Resetting...' : 'Reset Knowledge Base'}
        </button>
      )}
    </div>
  );
}
