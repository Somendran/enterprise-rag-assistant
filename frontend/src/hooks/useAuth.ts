import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  authStatus,
  bootstrap as bootstrapRequest,
  currentUser,
  login as loginRequest,
  logout as logoutRequest,
} from '../api/auth';
import { API_KEY, AUTH_TOKEN_STORAGE_KEY, PUBLIC_DEMO_MODE } from '../types';
import type { AuthStatus, AuthUser } from '../types';

export interface UseAuthResult {
  user: AuthUser | null;
  status: AuthStatus | null;
  token: string;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  bootstrap: (email: string, password: string, displayName: string) => Promise<void>;
}

export function useAuth(): UseAuthResult {
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [token, setToken] = useState(() => window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) || '');
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const loadedStatus = await authStatus();
        if (cancelled) return;
        setStatus(loadedStatus);
        if (!PUBLIC_DEMO_MODE && token) {
          setUser(await currentUser());
        }
      } catch (error) {
        console.warn('Failed to load auth state.', error);
        window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
        if (!cancelled) {
          setToken('');
          setUser(null);
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [token]);

  const login = useCallback(async (email: string, password: string) => {
    const response = await loginRequest(email, password);
    window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, response.access_token);
    setToken(response.access_token);
    setUser(response.user);
  }, []);

  const bootstrap = useCallback(async (email: string, password: string, displayName: string) => {
    const response = await bootstrapRequest(email, password, displayName);
    window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, response.access_token);
    setToken(response.access_token);
    setUser(response.user);
    setStatus((current) => current ? { ...current, has_users: true, bootstrap_required: false } : current);
  }, []);

  const logout = useCallback(async () => {
    try {
      await logoutRequest();
    } catch (error) {
      console.warn('Logout request failed.', error);
    }
    window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
    setToken('');
    setUser(null);
  }, []);

  const isAuthenticated = useMemo(
    () => Boolean(PUBLIC_DEMO_MODE || API_KEY || !status?.auth_enabled || user),
    [status, user],
  );

  return { user, status, token, isAuthenticated, isLoading, login, logout, bootstrap };
}
