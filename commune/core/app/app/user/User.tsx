'use client'
import { useState, useEffect } from 'react'
import { X, Copy, LogOut, Key as KeyIcon, Shield, Globe, FileSignature, CheckCircle, History, Package, User, Terminal, Lock, Zap } from 'lucide-react'
import { Key } from '@/app/key'
import type { User } from '@/app/types/user'
import { ModuleCaller } from '@/app/user/tabs/ModuleCaller'
import { SignVerifyTab } from './tabs/SignVerifyTab'
import { CopyButton } from '@/app/components/CopyButton'

import { InfoTab } from './tabs/InfoTab'

interface UserProfileProps {
  user: User
  isOpen: boolean
  onClose: () => void
  keyInstance: Key
  onLogout: () => void
}


export const UserProfile = ({ user, isOpen, onClose, keyInstance, onLogout }: UserProfileProps) => {
  const [copiedField, setCopiedField] = useState<string | null>(null)
  const [isAnimating, setIsAnimating] = useState(false)
  const [activeTab, setActiveTab] = useState<'profile' | 'sign' |  'modules'>('profile')
  const [isDragging, setIsDragging] = useState(false)
  const [dragStartX, setDragStartX] = useState(0)
  const [panelWidth, setPanelWidth] = useState(400)
  const MIN_WIDTH = 320
  const MAX_WIDTH = 600

  useEffect(() => {
    if (isOpen) {
      setIsAnimating(true)
    }
  }, [isOpen])

  const handleClose = () => {
    setIsAnimating(false)
    setTimeout(onClose, 200)
  }

  const handleLogout = () => {
    onLogout()
    handleClose()
  }


  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true)
    setDragStartX(e.clientX)
    e.preventDefault()
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return
      
      const deltaX = dragStartX - e.clientX
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, panelWidth + deltaX))
      setPanelWidth(newWidth)
      setDragStartX(e.clientX)
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'ew-resize'
      document.body.style.userSelect = 'none'
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'auto'
      document.body.style.userSelect = 'auto'
    }
  }, [isDragging, dragStartX, panelWidth])

  if (!isOpen && !isAnimating) return null

  const tabIcons = {
    profile: User,
    sign: FileSignature,
    modules: Package
  }

  return (
    <>
      {/* Backdrop */}
      <div 
        className={`fixed inset-0 bg-black/70 backdrop-blur-sm z-40 transition-all duration-300 ${
          isAnimating ? 'opacity-100' : 'opacity-0'
        }`}
        onClick={handleClose}
      />
      
      {/* Sidebar - Right side with enhanced styling */}
      <div className={`fixed top-0 right-0 h-full bg-black/95 backdrop-blur-md border-l border-green-500/50 z-50 transform transition-all duration-300 shadow-2xl shadow-green-500/20 ${
        isAnimating ? 'translate-x-0' : 'translate-x-full'
      }`}
      style={{ width: `${panelWidth}px` }}>
        {/* Drag Handle */}
        <div
          className="absolute top-0 left-0 w-1 h-full cursor-ew-resize bg-gradient-to-b from-green-500/20 via-green-500/50 to-green-500/20 hover:bg-green-500/60 transition-all"
          onMouseDown={handleMouseDown}
        />
        
        {/* Header with gradient background */}
        <div className="relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-green-500/20 via-transparent to-green-600/10" />
          <div className="relative p-6 border-b border-green-500/50">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">

                <div>
                  <CopyButton code={keyInstance.address as string} />
                  <code className="text-green-500/70 text-s font-mono">
                    {keyInstance.address.slice(0, 8)}...{keyInstance.address.slice(-6)}

                  </code>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleLogout}
                  className="p-2 rounded-lg border border-green-500/30 text-green-500 hover:bg-green-500/10 hover:border-green-500 transition-all"
                  title="Logout"
                >
                  <LogOut size={18} />
                </button>
                <button
                  onClick={handleClose}
                  className="p-2 rounded-lg border border-green-500/30 text-green-500 hover:bg-green-500/10 hover:border-green-500 transition-all"
                >
                  <X size={20} />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Tabs */}
        <div className="flex border-b border-green-500/30 bg-black/50">
          {Object.entries(tabIcons).map(([tab, Icon]) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab as any)}
              className={`flex-1 px-4 py-3 text-sm font-mono uppercase transition-all relative group ${
                activeTab === tab 
                  ? 'bg-gradient-to-t from-green-500/20 to-transparent text-green-400' 
                  : 'text-green-600 hover:text-green-400 hover:bg-green-500/5'
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <Icon size={16} />
                <span className="hidden sm:inline">{tab}</span>
              </div>
              {activeTab === tab && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-green-500 to-transparent" />
              )}
            </button>
          ))}
        </div>
        
        <div className="p-6 space-y-6 overflow-y-auto h-[calc(100%-11rem)] custom-scrollbar">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <InfoTab keyInstance={keyInstance} />
          )}

          {/* Sign/Verify Tab */}
          {activeTab === 'sign' && (
         
            <SignVerifyTab keyInstance={keyInstance}/>
          )}
          {/* Modules Tab */}

          {/* Modules Tab */}
          {activeTab === 'modules' && (
            <ModuleCaller keyInstance={keyInstance} />
          )}
        </div>
        
        {/* Visual Resize Indicator */}
        {isDragging && (
          <div className="absolute top-0 left-0 h-full w-full pointer-events-none">
            <div className="absolute left-0 top-0 h-full w-1 bg-green-500 animate-pulse" />
          </div>
        )}
      </div>

    </>
  )
}