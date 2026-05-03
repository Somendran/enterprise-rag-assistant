import { useCallback, useState } from 'react';
import {
  deleteDocument,
  listDocuments,
  reindexDocument,
  upload as uploadDocuments,
  uploadJobStatus,
} from '../api/documents';
import type { KnowledgeFile } from '../types';

export interface UseDocumentsResult {
  documents: KnowledgeFile[];
  isUploading: boolean;
  uploadProgress: number;
  upload: (files: File[]) => Promise<void>;
  deleteDoc: (fileHash: string) => Promise<void>;
  reindex: (fileHash: string) => Promise<void>;
  fetchDocuments: () => Promise<void>;
}

export function useDocuments(): UseDocumentsResult {
  const [documents, setDocuments] = useState<KnowledgeFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const fetchDocuments = useCallback(async () => {
    const response = await listDocuments();
    setDocuments((response.files || []).map((item) => ({
      fileHash: item.file_hash,
      documentId: item.document_id,
      filename: item.filename,
      chunks: Number(item.chunk_count || 0),
      indexedAt: Number(item.indexed_at || 0),
      parsingMethod: item.parsing_method || 'unknown',
      uploadStatus: item.upload_status || 'indexed',
      visionCallsUsed: Number(item.vision_calls_used || 0),
      embeddingModel: item.embedding_model || '',
      ownerUserId: item.owner_user_id || '',
      visibility: item.visibility || 'shared',
      allowedRoles: item.allowed_roles || [],
    })));
  }, []);

  const upload = useCallback(async (files: File[]) => {
    setIsUploading(true);
    setUploadProgress(0);
    try {
      const job = await uploadDocuments(files, (event) => {
        if (!event.total) return;
        setUploadProgress(Math.round((event.loaded / event.total) * 100));
      });
      await uploadJobStatus(job.job_id);
      await fetchDocuments();
    } finally {
      setIsUploading(false);
    }
  }, [fetchDocuments]);

  const deleteDoc = useCallback(async (fileHash: string) => {
    await deleteDocument(fileHash);
    setDocuments((current) => current.filter((doc) => doc.fileHash !== fileHash));
  }, []);

  const reindex = useCallback(async (fileHash: string) => {
    await reindexDocument(fileHash);
    await fetchDocuments();
  }, [fetchDocuments]);

  return { documents, isUploading, uploadProgress, upload, deleteDoc, reindex, fetchDocuments };
}
