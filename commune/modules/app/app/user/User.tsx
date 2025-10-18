'use client'

import { useState, useEffect } from 'react'
import { X, LogOut, FileSignature, Package, User } from 'lucide-react'
import { Key } from '@/app/key'
import type { User as UserType } from '@/app/types/user'
import { ModuleCaller } from '@/app/user/tabs/ModuleCaller'
import { SignVerifyTab } from './tabs/SignVerifyTab'
import { CopyButton } from '@/app/components/CopyButton'
import { InfoTab } from './tabs/InfoTab'

interface UserProfileProps {
  user: UserType
  isOpen: boolean
  onClose: () => void
  keyInstance: Key
  onLogout: () => void
}

export const UserProfile = ({ user, isOpen, onClose, keyInstance, onLogout }: UserProfileProps) => {
  const [isAnimating, setIsAnimating] = useState(false)
  const [activeTab, setActiveTab] = useState<'profile' | 'sign' | 'modules'>('profile')
  const [isDragging, setIsDragging] = useState(false)
  const [dragStartX, setDragStartX] = useState(0)
  const [panelWidth, setPanelWidth] = useState(400)
  const MIN_WIDTH = 320
  const MAX_WIDTH = 600

  useEffect(() => { if (isOpen) setIsAnimating(true) }, [isOpen])

  const handleClose = () => { setIsAnimating(false); setTimeout(onClose, 150) }
  const handleLogout = () => { onLogout(); handleClose() }

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true); setDragStartX(e.clientX); e.preventDefault()
  }

  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!isDragging) return
      const deltaX = dragStartX - e.clientX
      const w = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, panelWidth + deltaX))
      setPanelWidth(w); setDragStartX(e.clientX)
    }
    const up = () => setIsDragging(false)

    if (isDragging) {
      document.addEventListener('mousemove', move)
      document.addEventListener('mouseup', up)
      document.body.style.cursor = 'ew-resize'
      document.body.style.userSelect = 'none'
    }
    return () => {
      document.removeEventListener('mousemove', move)
      document.removeEventListener('mouseup', up)
      document.body.style.cursor = 'auto'
      document.body.style.userSelect = 'auto'
    }
  }, [isDragging, dragStartX, panelWidth])

  if (!isOpen && !isAnimating) return null

  const tabIcons = { profile: User, sign: FileSignature, modules: Package }

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 bg-black/80 z-40 transition-opacity duration-150 ${isAnimating ? 'opacity-100' : 'opacity-0'}`}
        onClick={handleClose}
      />

      {/* Sidebar */}
      <div
        className={`fixed top-0 right-0 h-full z-50 transform transition-transform duration-150
                    ${isAnimating ? 'translate-x-0' : 'translate-x-full'}`}
        style={{ width: `${panelWidth}px` }}
      >
        <div className="h-full bg-black text-green-200 font-mono border-l border-green-500/40">
          {/* Drag handle (thin, retro) */}
          <div
            className="absolute top-0 left-0 h-full w-[2px] bg-green-500/40 cursor-ew-resize"
            onMouseDown={handleMouseDown}
            aria-label="Resize panel"
          />

          {/* Header */}
          <div className="px-4 py-3 border-b border-green-500/40 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CopyButton content={keyInstance.address as string} />
              <code className="text-xs text-green-400/90">
                {keyInstance.address.slice(0, 8)}…{keyInstance.address.slice(-6)}
              </code>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleLogout}
                className="px-2 py-1 border border-green-500/50 text-green-300 hover:bg-green-500/10"
                title="Logout"
              >
                <LogOut size={16} />
              </button>
              <button
                onClick={handleClose}
                className="px-2 py-1 border border-green-500/50 text-green-300 hover:bg-green-500/10"
                title="Close"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Tabs (simple row, underline active) */}
          <div className="flex border-b border-green-500/30">
            {Object.entries(tabIcons).map(([tab, Icon]) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab as any)}
                className={`flex-1 px-3 py-2 text-xs uppercase tracking-wide
                            ${activeTab === tab
                              ? 'text-green-300 border-b-2 border-green-400'
                              : 'text-green-600 hover:text-green-300'}`}
              >
                <span className="inline-flex items-center gap-1.5">
                  <Icon size={14} />
                  {tab}
                </span>
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="p-4 space-y-4 overflow-y-auto h-[calc(100%-9rem)]">
            {activeTab === 'profile' && <InfoTab keyInstance={keyInstance} />}
            {activeTab === 'sign' && <SignVerifyTab keyInstance={keyInstance} />}
            {activeTab === 'modules' && <ModuleCaller keyInstance={keyInstance} />}
          </div>

          {/* Resize indicator (retro blink) */}
          {isDragging && <div className="absolute top-0 left-0 h-full w-[2px] bg-green-400 animate-pulse pointer-events-none" />}
        </div>
      </div>
    </>
  )
}
