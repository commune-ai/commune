'use client'
import { useState, FormEvent, useEffect } from 'react'
import { X, Shield } from 'lucide-react'

interface MobileLoginModalProps {
  isOpen: boolean
  onClose: () => void
  onLogin: (password: string) => Promise<void>
  defaultPassword: string
}

// Function to generate a random password
const generateRandomPassword = (length: number = 12): string => {
  const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?'
  let password = ''
  for (let i = 0; i < length; i++) {
    password += charset.charAt(Math.floor(Math.random() * charset.length))
  }
  return password
}

export const MobileLoginModal = ({ isOpen, onClose, onLogin, defaultPassword }: MobileLoginModalProps) => {
  const [password, setPassword] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    // If no default password is provided, generate a random one
    if (!defaultPassword || defaultPassword === '') {
      const randomPassword = generateRandomPassword(12)
      setPassword(randomPassword)
    } else {
      setPassword(defaultPassword)
    }
  }, [defaultPassword])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    try {
      await onLogin(password)
      onClose()
    } catch (error) {
      console.error('Login failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80">
      <div className="w-full max-w-sm bg-black border border-green-500/30 rounded-lg shadow-2xl shadow-green-500/20">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-green-400 text-lg font-mono flex items-center gap-2">
              <Shield size={20} />
              $ login
            </h2>
            <button
              onClick={onClose}
              className="text-green-400/60 hover:text-green-400 transition-colors"
            >
              <X size={20} />
            </button>
          </div>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-green-400/60 text-sm mb-2 font-mono">
                password:
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="enter password"
                className="w-full px-4 py-3 bg-black/60 border border-green-500/30 rounded text-green-400 font-mono focus:outline-none focus:border-green-400 transition-colors"
                autoFocus
              />
              {(!defaultPassword || defaultPassword === '') && (
                <p className="mt-2 text-xs text-green-400/40 font-mono">
                  auto-generated password: {password}
                </p>
              )}
            </div>
            
            <button
              type="submit"
              disabled={isLoading || !password}
              className="w-full px-4 py-3 bg-green-900/20 text-green-400 border border-green-500/30 rounded hover:bg-green-900/30 transition-colors font-mono disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'connecting...' : '$ connect wallet'}
            </button>
          </form>
          
          <p className="mt-4 text-xs text-green-400/40 text-center font-mono">
            secure connection via polkadot
          </p>
        </div>
      </div>
    </div>
  )
}

export default MobileLoginModal