'use client'
import React, { createContext, useContext, useEffect, useState } from 'react'
import { Key } from '@/app/key'
import { cryptoWaitReady } from '@polkadot/util-crypto'

interface UserContextType {
  keyInstance: Key | null
  user: { address: string; crypto_type: string } | null
  password: string
  signIn: (password: string) => Promise<void>
  signOut: () => void
  isLoading: boolean
}

const UserContext = createContext<UserContextType | undefined>(undefined)

export const UserProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [keyInstance, setKeyInstance] = useState<Key | null>(null)
  const [user, setUser] = useState<{ address: string; crypto_type: string } | null>(null)
  const [password, setPassword] = useState('')
  const [isLoading, setIsLoading] = useState(true)


  // Initialize from localStorage on mount
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const storedUser = localStorage.getItem('user_data')
        const storedPassword = localStorage.getItem('user_password')
        
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
        localStorage.removeItem('user_data')
        localStorage.removeItem('user_password')
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
      localStorage.setItem('user_data', JSON.stringify(userData))
      localStorage.setItem('user_password', newPassword)
    } catch (error) {
      console.error('Failed to sign in:', error)
      throw error
    }
  }

  const signOut = () => {
    setKeyInstance(null)
    setUser(null)
    setPassword('')
    localStorage.removeItem('user_data')
    localStorage.removeItem('user_password')
  }

  return (
    <UserContext.Provider value={{ keyInstance, user, password, signIn, signOut, isLoading }}>
      {children}
    </UserContext.Provider>
  )
}

export const useUserContext = () => {
  const context = useContext(UserContext)
  if (context === undefined) {
    throw new Error('useUserContext must be used within an UserProvider')
  }
  return context
}