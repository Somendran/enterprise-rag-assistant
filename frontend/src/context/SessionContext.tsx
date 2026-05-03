/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import client from '../api/client';
import { DEMO_SESSION_STORAGE_KEY, PUBLIC_DEMO_MODE } from '../types';
import type { DemoSessionResponse } from '../types';

interface SessionContextValue {
  demoToken: string;
  isDemoMode: boolean;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [demoToken, setDemoToken] = useState(() => (
    PUBLIC_DEMO_MODE ? window.localStorage.getItem(DEMO_SESSION_STORAGE_KEY) || '' : ''
  ));

  useEffect(() => {
    if (!PUBLIC_DEMO_MODE || demoToken) return;
    let cancelled = false;
    const createDemoSession = async () => {
      try {
        const response = await client.post<DemoSessionResponse>('/demo/session');
        window.localStorage.setItem(DEMO_SESSION_STORAGE_KEY, response.data.token);
        if (!cancelled) setDemoToken(response.data.token);
      } catch (error) {
        console.warn('Failed to create demo session.', error);
        window.localStorage.removeItem(DEMO_SESSION_STORAGE_KEY);
      }
    };
    void createDemoSession();
    return () => {
      cancelled = true;
    };
  }, [demoToken]);

  const value = useMemo<SessionContextValue>(
    () => ({ demoToken, isDemoMode: PUBLIC_DEMO_MODE }),
    [demoToken],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSessionContext(): SessionContextValue {
  const value = useContext(SessionContext);
  if (!value) {
    throw new Error('useSessionContext must be used inside SessionProvider');
  }
  return value;
}
