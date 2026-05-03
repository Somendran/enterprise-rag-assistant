import client from './client';
import type { AdminOverview, AuditEvent, EvalRun, Message, ModelHealthItem } from '../types';

export async function submitFeedback(
  question: string,
  assistantMessage: Message,
  rating: string,
  reason = '',
): Promise<void> {
  await client.post('/feedback', {
    question,
    answer: assistantMessage.content,
    rating,
    reason,
    confidence_score: assistantMessage.confidence_score,
    sources: assistantMessage.sources || [],
    diagnostics: assistantMessage.diagnostics || undefined,
  });
}

export async function listSessions<T>(): Promise<T[]> {
  const response = await client.get<{ sessions: T[] }>('/chat/sessions');
  return response.data?.sessions || [];
}

export async function createSession<T>(): Promise<T> {
  const response = await client.post<T>('/chat/sessions', { title: 'New chat' });
  return response.data;
}

export async function deleteSession(sessionId: string): Promise<void> {
  await client.delete(`/chat/sessions/${encodeURIComponent(sessionId)}`);
}

export async function listMessages<T>(sessionId: string): Promise<T[]> {
  const response = await client.get<{ messages: T[] }>(
    `/chat/sessions/${encodeURIComponent(sessionId)}/messages`,
  );
  return response.data?.messages || [];
}

export async function saveMessage(sessionId: string, message: Message): Promise<void> {
  await client.post(`/chat/sessions/${encodeURIComponent(sessionId)}/messages`, {
    id: message.id,
    role: message.role,
    content: message.content,
    sources: message.sources || [],
    diagnostics: message.diagnostics || undefined,
    confidence_score: message.confidence_score,
    confidence_level: message.confidence_level,
  });
}

export async function adminOverview(): Promise<AdminOverview> {
  const response = await client.get<AdminOverview>('/admin/overview');
  return response.data;
}

export async function modelHealth(): Promise<ModelHealthItem[]> {
  const response = await client.get<{ checks: ModelHealthItem[] }>('/health/models');
  return response.data?.checks || [];
}

export async function auditLog(): Promise<AuditEvent[]> {
  const response = await client.get<{ events: AuditEvent[] }>('/admin/audit-log');
  return response.data?.events || [];
}

export async function runEval(): Promise<string> {
  const response = await client.post<{ run_id: string }>('/evals/runs');
  return response.data.run_id;
}

export async function evalRun(runId: string): Promise<EvalRun> {
  const response = await client.get<EvalRun>(`/evals/runs/${encodeURIComponent(runId)}`);
  return response.data;
}

export async function evalRuns(): Promise<EvalRun[]> {
  const response = await client.get<{ runs: EvalRun[] }>('/evals/runs');
  return response.data?.runs || [];
}
