'use client'
import { AuthProvider } from './context/AuthContext'
import { Header } from './components/Header'

export default function App({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <Header />
      {children}
    </AuthProvider>
  )
}