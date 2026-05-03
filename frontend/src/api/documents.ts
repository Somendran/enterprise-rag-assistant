import client from './client';
import type {
  DocumentChunksResponse,
  IngestionJobStatus,
  KnowledgeBaseFilesResponse,
  UploadItemResult,
  UploadProgressHandler,
} from '../types';

export async function listDocuments(): Promise<KnowledgeBaseFilesResponse> {
  const response = await client.get<KnowledgeBaseFilesResponse>('/knowledge-base/files');
  return response.data;
}

export async function upload(files: File[], onUploadProgress?: UploadProgressHandler): Promise<{ job_id: string }> {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  const response = await client.post<{ job_id: string }>('/upload/jobs', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  });
  return response.data;
}

export async function uploadJobStatus(jobId: string): Promise<IngestionJobStatus> {
  const response = await client.get<IngestionJobStatus>(`/upload/jobs/${encodeURIComponent(jobId)}`);
  return response.data;
}

export async function deleteDocument(fileHash: string): Promise<{ chunks_deleted?: number }> {
  const response = await client.delete<{ chunks_deleted?: number }>(
    `/knowledge-base/files/${encodeURIComponent(fileHash)}`,
  );
  return response.data;
}

export async function reindexDocument(fileHash: string): Promise<UploadItemResult> {
  const response = await client.post<UploadItemResult>(
    `/knowledge-base/files/${encodeURIComponent(fileHash)}/reindex`,
  );
  return response.data;
}

export async function documentChunks(
  fileHash: string,
  focusChunkIndex?: number | null,
): Promise<DocumentChunksResponse> {
  const query = typeof focusChunkIndex === 'number'
    ? `?focus_chunk_index=${focusChunkIndex}&neighbor_window=2`
    : '';
  const response = await client.get<DocumentChunksResponse>(
    `/knowledge-base/files/${encodeURIComponent(fileHash)}/chunks${query}`,
  );
  return response.data;
}

export async function updatePermissions(
  fileHash: string,
  visibility: string,
  allowedRoles: string[],
): Promise<KnowledgeBaseFilesResponse> {
  const response = await client.patch<KnowledgeBaseFilesResponse>(
    `/knowledge-base/files/${encodeURIComponent(fileHash)}/permissions`,
    { visibility, allowed_roles: allowedRoles },
  );
  return response.data;
}

export async function resetKnowledgeBase(timeout: number): Promise<{ uploads_deleted?: number }> {
  const response = await client.post<{ uploads_deleted?: number }>(
    '/knowledge-base/reset',
    undefined,
    { timeout },
  );
  return response.data;
}
