'use client'

import { useState } from 'react'
import { UserProfile } from '@/app/user/User'
import { useUserContext } from '@/app/context/UserContext'
import { CopyButton } from '@/app/components/CopyButton'

export const UserHeader = () => {
  const [password, setPassword] = useState('')
  const [showProfile, setShowProfile] = useState(false)
  const { user, keyInstance, signIn, signOut, isLoading } = useUserContext()

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      if (!password) {
        setPassword(crypto.randomUUID().replace(/-/g, '').slice(0, 16))
      }
      await signIn(password)
      setPassword('')
      setShowProfile(true)
    } catch (err) {
      console.error('Failed to sign in:', err)
    }
  }

  const handleLogout = () => {
    signOut()
    setShowProfile(false)
  }

  if (isLoading) {
    return (
      <div className="h-12 px-4 flex items-center font-mono text-green-400">
        Loading...
      </div>
    )
  }

  return (
    <>
      <div className="flex items-center gap-3">
        {keyInstance ? (
          <div
            onClick={() => setShowProfile(!showProfile)}
            title={`Address: ${keyInstance.address}`}
            aria-pressed={showProfile}
            role="button"
            className={`cursor-pointer select-none h-11 px-3 border font-mono tracking-wider rounded-lg flex items-center gap-2 transition-all
              ${showProfile
                ? 'bg-green-500 text-black border-green-500'
                : 'bg-black text-green-400 border-green-500 hover:border-green-400 hover:text-green-300'}`}
          >
            {/* Flex row: text and copy button inline */}
            <span className="text-sm leading-none flex items-center">
              {keyInstance.address.slice(0, 6)}…{keyInstance.address.slice(-4)}
            </span>
            <div className="flex items-center justify-center">
              <CopyButton
                code={keyInstance.address as string}
                size="sm"
                className="text-green-400 hover:text-green-300"
              />
            </div>
          </div>
        ) : (
          <form onSubmit={handleSignIn} className="flex gap-2">
            <input
              type="password"
              placeholder="SECRET"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="h-11 px-3 bg-black border border-green-500 text-green-400 placeholder-green-700
                         focus:outline-none focus:border-green-400 w-36 font-mono text-sm rounded-lg"
            />
            <button
              type="submit"
              className="h-11 px-5 border border-green-500 text-green-400 hover:bg-green-500 hover:text-black
                         font-mono text-sm tracking-wider transition-colors rounded-lg"
            >
              LOGIN
            </button>
          </form>
        )}
      </div>

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
