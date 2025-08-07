'use client'
import { useContext } from 'react'
import { AuthContext } from '@/app/context/AuthContext'

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export default useAuth