'use client'
import Link from 'next/link'
import { useState, useRef, FormEvent } from 'react'
import { Copy, Key } from 'lucide-react'
import { UserProfile } from '@/app/user/UserProfile'
import { useRouter, usePathname } from 'next/navigation'
import { useAuth } from '@/app/context/AuthContext'
import {copyToClipboard} from '@/app/utils' // Import the utility function

export const Header = () => {
  const [password, setPassword] = useState('')
  const [showProfile, setShowProfile] = useState(false)
  const [copiedAddress, setCopiedAddress] = useState(false)
  const copyTimerRef = useRef<number | null>(null)

  const { user, keyInstance, signIn, signOut, isLoading } = useAuth()
  const router = useRouter()
  const pathname = usePathname()

  const handleSignIn = async (e: FormEvent) => {
    e.preventDefault()
    try {
      await signIn(password)
      setPassword('')
      setShowProfile(true)
    } catch (error) {
      console.error('Failed to sign in:', error)
    }
  }

  const handleLogout = () => {
    signOut()
    setShowProfile(false)
  }

  const handleCopyAddress = async (e: React.MouseEvent) => {
    e.stopPropagation()
    const addr = user?.address ?? keyInstance?.address
    if (!addr) return
    const ok = await copyToClipboard(addr)
    setCopiedAddress(ok)
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current)
    copyTimerRef.current = window.setTimeout(() => setCopiedAddress(false), 2000)
  }

  if (isLoading) {
    return (
      <header className="fixed top-0 z-40 w-full bg-black border-b border-green-500 font-mono">
        <nav className="p-4 px-4 mx-auto max-w-7xl">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 flex-1">
              <Link href="/" className="group">
                <div className="relative w-12 h-12">
                  <div className="absolute inset-0 animate-spin-slow preserve-3d">
                    <div className="absolute inset-0 bg-green-500 border-2 border-green-400 rotate-y-0"></div>
                    <div className="absolute inset-0 bg-green-600 border-2 border-green-400 rotate-y-90"></div>
                    <div className="absolute inset-0 bg-green-700 border-2 border-green-400 rotate-y-180"></div>
                    <div className="absolute inset-0 bg-green-800 border-2 border-green-400 rotate-y-270"></div>
                  </div>
                </div>
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-12 px-4 flex items-center text-green-500">
                Loading...
              </div>
            </div>
          </div>
        </nav>
      </header>
    )
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
                    <div className="absolute inset-0 bg-green-500 border-2 border-green-400 rotate-y-0"></div>
                    <div className="absolute inset-0 bg-green-600 border-2 border-green-400 rotate-y-90"></div>
                    <div className="absolute inset-0 bg-green-700 border-2 border-green-400 rotate-y-180"></div>
                    <div className="absolute inset-0 bg-green-800 border-2 border-green-400 rotate-y-270"></div>
                  </div>
                </div>
              </Link>
            </div>

            {/* Right side - User Section */}
            <div className="flex items-center gap-3">
              {keyInstance ? (
                <div className="flex items-center gap-2">
                  {/* Wrapper is a div, not a button (prevents nested button issue) */}
                  <div
                    onClick={() => setShowProfile(!showProfile)}
                    className={`cursor-pointer select-none h-12 px-4 pr-2 border-2 border-green-500 font-mono transition-all uppercase text-base tracking-wider rounded-lg flex items-center gap-2 ${
                      showProfile 
                        ? 'bg-green-500 text-black' 
                        : 'text-green-500 hover:bg-green-500 hover:text-black'
                    }`}
                    title={`Address: ${keyInstance.address}`}
                    role="button"
                    aria-pressed={showProfile}
                  >
                    <button
                      onClick={handleCopyAddress}
                      className="p-1 hover:bg-black/20 rounded transition-colors"
                      title={copiedAddress ? 'Copied!' : 'Copy address'}
                      type="button"
                    >
                      {copiedAddress ? '✓' : <Copy size={16} />}
                    </button>
                    <span className="font-bold">
                      {keyInstance.address.slice(0, 6)}...{keyInstance.address.slice(-4)}
                    </span>
                    <Key size={20} />
                  </div>
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
