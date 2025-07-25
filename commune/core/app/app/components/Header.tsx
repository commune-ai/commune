'use client'
import Link from 'next/link'
import { useState, FormEvent, useEffect } from 'react'
import { Key } from '@/app/user/key'
import { cryptoWaitReady } from '@polkadot/util-crypto'
import { RefreshCw, LogOut, User, Search, Copy } from 'lucide-react'
import { UserProfile } from '@/app/user/profile/UserProfile'
import type { User as UserType } from '@/app/types/user'
import { useRouter, usePathname } from 'next/navigation'
import Image from 'next/image'

interface HeaderProps {
  onSearch?: (term: string) => void
  onRefresh?: () => void
}

export const Header = ({ onSearch, onRefresh }: HeaderProps = {}) => {
  const [password, setPassword] = useState('')
  const [keyInstance, setKeyInstance] = useState<Key | null>(null)
  const [user, setUser] = useState<UserType | null>(null)
  const [showProfile, setShowProfile] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
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

  const handleSearchSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (onSearch) {
      onSearch(searchQuery)
    } else {
      // Trigger search via global event for module explorer
      window.dispatchEvent(new CustomEvent('searchModules', { detail: searchQuery }))
    }
  }

  const handleSearchChange = (value: string) => {
    setSearchQuery(value)
    // Real-time search
    if (onSearch) {
      onSearch(value)
    } else {
      window.dispatchEvent(new CustomEvent('searchModules', { detail: value }))
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
            {/* Left side - Logo image */}
            <Link href="/" className="flex items-center group">
              <Image
                src="/logo.png"
                alt="dhub logo"
                width={64}
                height={64}
                className="w-16 h-16 object-contain"
                priority
              />
            </Link>

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
    </>
  )
}

export default Header