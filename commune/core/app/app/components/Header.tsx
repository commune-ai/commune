'use client'
import Link from 'next/link'
import { useState, FormEvent, useEffect } from 'react'
import { Key } from '@/app/key'
import { cryptoWaitReady } from '@polkadot/util-crypto'
import { RefreshCw, LogOut, User, Search, Copy, Filter } from 'lucide-react'
import { UserProfile } from '@/app/user/profile/UserProfile'
import type { User as UserType } from '@/app/types/user'
import { useRouter, usePathname } from 'next/navigation'
import Image from 'next/image'

interface HeaderProps {
  onRefresh?: () => void
}

export const Header = ({ onRefresh }: HeaderProps = {}) => {
  const [password, setPassword] = useState('')
  const [keyInstance, setKeyInstance] = useState<Key | null>(null)
  const [user, setUser] = useState<UserType | null>(null)
  const [showProfile, setShowProfile] = useState(false)
  const [copiedAddress, setCopiedAddress] = useState(false)
  const router = useRouter()
  const pathname = usePathname()

  // Store user session in localStorage to persist across pages
  useEffect(() => {
    const initializeUser = async () => {
      const storedUser = localStorage.getItem('dhub_user')
      const storedPassword = localStorage.getItem('dhub_password')
      
      if (storedUser && storedPassword) {
        try {
          await cryptoWaitReady()
          const userData = JSON.parse(storedUser)
          const key = new Key(storedPassword)
          setUser(userData)
          setKeyInstance(key)
        } catch (error) {
          console.error('Failed to restore user session:', error)
          // Clear invalid session data
          localStorage.removeItem('dhub_user')
          localStorage.removeItem('dhub_password')
        }
      }
    }
    
    initializeUser()
  }, [])

  const handleSignIn = async (e: FormEvent) => {
    e.preventDefault()
    try {
      await cryptoWaitReady()
      const key = new Key(password)
      setKeyInstance(key)
      const userData = {
        address: key.address,
        crypto_type: key.crypto_type,
      }
      setUser(userData)
      
      // Store user session with encrypted password
      localStorage.setItem('dhub_user', JSON.stringify(userData))
      localStorage.setItem('dhub_password', password)
      
      setPassword('')
      setShowProfile(true) // Auto-open profile on login
    } catch (error) {
      console.error('Failed to create key:', error)
    }
  }

  const handleLogout = () => {
    setKeyInstance(null)
    setUser(null)
    setShowProfile(false)
    localStorage.removeItem('dhub_user')
    localStorage.removeItem('dhub_password')
  }

  const handleRefreshClick = () => {
    if (onRefresh) {
      onRefresh()
    } else if (pathname === '/') {
      // Trigger refresh via global event
      window.dispatchEvent(new CustomEvent('refreshModules'))
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedAddress(true)
    setTimeout(() => setCopiedAddress(false), 2000)
  }

  return (
    <>
      <header className="fixed top-0 z-40 w-full bg-black border-b border-green-500 font-mono">
        <nav className="p-4 px-4 mx-auto max-w-7xl">
          <div className="flex items-center justify-between gap-4">
            {/* Left side - Rotating cube logo */}
            <div className="flex items-center gap-3 flex-1">
              <Link href="/" className="group">
                <div className="relative w-12 h-12">
                  <div className="absolute inset-0 animate-spin-slow preserve-3d">
                    <div className="absolute inset-0 bg-green-500 border-2 border-green-400 transform rotate-y-0"></div>
                    <div className="absolute inset-0 bg-green-600 border-2 border-green-400 transform rotate-y-90"></div>
                    <div className="absolute inset-0 bg-green-700 border-2 border-green-400 transform rotate-y-180"></div>
                    <div className="absolute inset-0 bg-green-800 border-2 border-green-400 transform rotate-y-270"></div>
                  </div>
                </div>
              </Link>
            </div>

            {/* Right side - User Section */}
            <div className="flex items-center gap-3">
              {user ? (
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowProfile(!showProfile)}
                    className={`h-12 px-4 border-2 border-green-500 font-mono transition-all uppercase text-base tracking-wider rounded-lg ${
                      showProfile 
                        ? 'bg-green-500 text-black' 
                        : 'text-green-500 hover:bg-green-500 hover:text-black'
                    }`}
                    title={`Address: ${user.address}`}
                  >
                    <span className="font-bold">{user.address.slice(0, 6)}...{user.address.slice(-4)}</span>
                  </button>
                  <button
                    onClick={() => copyToClipboard(user.address)}
                    className="h-12 w-12 flex items-center justify-center border-2 border-green-500 text-green-500 hover:bg-green-500 hover:text-black transition-colors rounded-lg"
                    title={copiedAddress ? 'Copied!' : 'Copy address'}
                  >
                    {copiedAddress ? '✓' : <Copy size={20} />}
                  </button>
                </div>
              ) : (
                <form onSubmit={handleSignIn} className="flex gap-3">
                  <input
                    type="password"
                    placeholder="YOUR SECRET ;)"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="h-12 px-4 bg-black border border-green-500 text-green-500 placeholder-green-500/50 focus:outline-none focus:border-green-400 w-36 font-mono uppercase text-sm rounded-lg"
                  />
                  <button
                    type="submit"
                    className="h-12 px-6 border border-green-500 text-green-500 hover:bg-green-500 hover:text-black font-mono uppercase text-sm tracking-wider transition-colors rounded-lg"
                  >
                    LOGIN
                  </button>
                </form>
              )}
            </div>
          </div>
        </nav>
      </header>
      
      {/* User Profile Sidebar - Right side */}
      {user && keyInstance && (
        <UserProfile
          user={user}
          isOpen={showProfile}
          onClose={() => setShowProfile(false)}
          keyInstance={keyInstance}
          onLogout={handleLogout}
        />
      )}

      <style jsx>{`
        .preserve-3d {
          transform-style: preserve-3d;
        }
        .rotate-y-0 {
          transform: rotateY(0deg);
        }
        .rotate-y-90 {
          transform: rotateY(90deg);
        }
        .rotate-y-180 {
          transform: rotateY(180deg);
        }
        .rotate-y-270 {
          transform: rotateY(270deg);
        }
        @keyframes spin-slow {
          from {
            transform: rotateY(0deg);
          }
          to {
            transform: rotateY(360deg);
          }
        }
        .animate-spin-slow {
          animation: spin-slow 4s linear infinite;
        }
      `}</style>
    </>
  )
}

export default Header