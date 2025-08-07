'use client'
import React, { createContext, useContext, useEffect, useState } from 'react'
import { Key } from '@/app/key'
import { cryptoWaitReady } from '@polkadot/util-crypto'

interface AuthContextType {
  keyInstance: Key | null
  user: { address: string; crypto_type: string } | null
  password: string
  signIn: (password: string) => Promise<void>
  signOut: () => void
  isLoading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [keyInstance, setKeyInstance] = useState<Key | null>(null)
  const [user, setUser] = useState<{ address: string; crypto_type: string } | null>(null)
  const [password, setPassword] = useState('')
  const [isLoading, setIsLoading] = useState(true)

  // Initialize from localStorage on mount
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const storedUser = localStorage.getItem('dhub_user')
        const storedPassword = localStorage.getItem('dhub_password')
        
        if (storedUser && storedPassword) {
          await cryptoWaitReady()
          const userData = JSON.parse(storedUser)
          const key = new Key(storedPassword)
          setUser(userData)
          setKeyInstance(key)
          setPassword(storedPassword)
        }
      } catch (error) {
        console.error('Failed to restore auth session:', error)
        localStorage.removeItem('dhub_user')
        localStorage.removeItem('dhub_password')
      } finally {
        setIsLoading(false)
      }
    }
    
    initializeAuth()
  }, [])

  const signIn = async (newPassword: string) => {
    try {
      await cryptoWaitReady()
      const key = new Key(newPassword)
      const userData = {
        address: key.address,
        crypto_type: key.crypto_type,
      }
      
      setKeyInstance(key)
      setUser(userData)
      setPassword(newPassword)
      
      // Persist to localStorage
      localStorage.setItem('dhub_user', JSON.stringify(userData))
      localStorage.setItem('dhub_password', newPassword)
    } catch (error) {
      console.error('Failed to sign in:', error)
      throw error
    }
  }

  const signOut = () => {
    setKeyInstance(null)
    setUser(null)
    setPassword('')
    localStorage.removeItem('dhub_user')
    localStorage.removeItem('dhub_password')
  }

  return (
    <AuthContext.Provider value={{ keyInstance, user, password, signIn, signOut, isLoading }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}