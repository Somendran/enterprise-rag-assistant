/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext } from 'react';
import type { ReactNode } from 'react';
import { useAuth } from '../hooks/useAuth';
import type { UseAuthResult } from '../hooks/useAuth';

const AuthContext = createContext<UseAuthResult | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const auth = useAuth();
  return <AuthContext.Provider value={auth}>{children}</AuthContext.Provider>;
}

export function useAuthContext(): UseAuthResult {
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error('useAuthContext must be used inside AuthProvider');
  }
  return value;
}
