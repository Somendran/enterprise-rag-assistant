import { useState } from 'react';
import { Loader2 } from 'lucide-react';

interface BootstrapFormProps {
  authError: string | null;
  isAuthenticating: boolean;
  onSubmit: (email: string, password: string, displayName: string) => void;
}

export function BootstrapForm({ authError, isAuthenticating, onSubmit }: BootstrapFormProps) {
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');

  return (
    <form
      className="auth-card"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit(email, password, displayName);
      }}
    >
      <h1>RAGiT</h1>
      <p>Create the first admin account.</p>
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
        Display name
        <input
          type="text"
          value={displayName}
          onChange={(event) => setDisplayName(event.target.value)}
          autoComplete="name"
        />
      </label>
      <label>
        Password
        <input
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          autoComplete="new-password"
        />
      </label>
      {authError && <p className="auth-error">{authError}</p>}
      <button type="submit" disabled={isAuthenticating || !email || !password}>
        {isAuthenticating ? <Loader2 size={15} className="animate-spin" /> : null}
        Create admin
      </button>
    </form>
  );
}
