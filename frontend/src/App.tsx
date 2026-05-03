import './App.css';
import AppShell from './components/AppShell';
import { AuthProvider } from './context/AuthContext';
import { SessionProvider } from './context/SessionContext';

function App() {
  return (
    <AuthProvider>
      <SessionProvider>
        <AppShell />
      </SessionProvider>
    </AuthProvider>
  );
}

export default App;