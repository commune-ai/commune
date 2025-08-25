'use client'
import { useState, useRef } from 'react'
import { Copy, Key } from 'lucide-react'
import { UserProfile } from '@/app/user/User'
import { useUserContext } from '@/app/context/UserContext'
import { CopyButton } from '@/app/components/CopyButton'

export const UserHeader = () => {
  const [password, setPassword] = useState('')
  const [showProfile, setShowProfile] = useState(false)
  const [copiedAddress, setCopiedAddress] = useState(false)

  const { user, keyInstance, signIn, signOut, isLoading } = useUserContext()

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      // if the password is null then generate a random key
      if (password.length === 0) {
        console.warn('No password provided, generating a random key.')
        setPassword(crypto.randomUUID().replace(/-/g, '').slice(0, 16))
      }
      
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


  if (isLoading) {
    return (
      <div className="h-12 px-4 flex items-center text-green-500">
        Loading...
      </div>
    )
  }

  return (
    <>
      <div className="flex items-center gap-4">
        {keyInstance ? (
          <div className="flex items-center gap-2">
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
              <span className="font-bold">
                {keyInstance.address.slice(0, 6)}...{keyInstance.address.slice(-4)}
              </span>
              <CopyButton code={keyInstance.address as string} />
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

      {/* User Profile Sidebar */}
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

export default UserHeader