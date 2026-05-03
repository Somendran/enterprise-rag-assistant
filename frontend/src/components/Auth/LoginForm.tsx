import { useState } from 'react';
import { Loader2 } from 'lucide-react';

interface LoginFormProps {
  authError: string | null;
  isAuthenticating: boolean;
  onSubmit: (email: string, password: string) => void;
}

export function LoginForm({ authError, isAuthenticating, onSubmit }: LoginFormProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  return (
    <form
      className="auth-card"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit(email, password);
      }}
    >
      <h1>RAGiT</h1>
      <p>Sign in to your workspace.</p>
      <label>
        Email
        <input
          type="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          autoComplete="email"
        />
      </label>
      <label>
        Password
        <input
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          autoComplete="current-password"
        />
      </label>
      {authError && <p className="auth-error">{authError}</p>}
      <button type="submit" disabled={isAuthenticating || !email || !password}>
        {isAuthenticating ? <Loader2 size={15} className="animate-spin" /> : null}
        Sign in
      </button>
    </form>
  );
}
