import client from './client';
import type { AuthStatus, AuthTokenResponse, AuthUser } from '../types';

export async function authStatus(): Promise<AuthStatus> {
  const response = await client.get<AuthStatus>('/auth/status');
  return response.data;
}

export async function login(email: string, password: string): Promise<AuthTokenResponse> {
  const response = await client.post<AuthTokenResponse>('/auth/login', { email, password });
  return response.data;
}

export async function bootstrap(
  email: string,
  password: string,
  displayName: string,
): Promise<AuthTokenResponse> {
  const response = await client.post<AuthTokenResponse>('/auth/bootstrap', {
    email,
    password,
    display_name: displayName,
  });
  return response.data;
}

export async function currentUser(): Promise<AuthUser> {
  const response = await client.get<{ user: AuthUser }>('/auth/me');
  return response.data.user;
}

export async function logout(): Promise<void> {
  await client.post('/auth/logout');
}
