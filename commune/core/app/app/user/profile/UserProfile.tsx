'use client'
import { useState, useEffect } from 'react'
import { X, Copy, LogOut, Key as KeyIcon, Shield, Globe, FileSignature, CheckCircle, History, Package, User, Terminal, Lock, Zap } from 'lucide-react'
import { Key } from '@/app/user/key'
import type { User } from '@/app/types/user'

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
  const [activeTab, setActiveTab] = useState<'profile' | 'sign' | 'transactions' | 'modules'>('profile')
  const [signMessage, setSignMessage] = useState('')
  const [signature, setSignature] = useState('')
  const [verifyMessage, setVerifyMessage] = useState('')
  const [verifySignature, setVerifySignature] = useState('')
  const [verifyPublicKey, setVerifyPublicKey] = useState('')
  const [verifyResult, setVerifyResult] = useState<boolean | null>(null)
  const [transactions, setTransactions] = useState<any[]>([])
  const [userModules, setUserModules] = useState<any[]>([])
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

  const copyToClipboard = (text: string, field: string) => {
    navigator.clipboard.writeText(text)
    setCopiedField(field)
    setTimeout(() => setCopiedField(null), 2000)
  }

  const handleSign = async () => {
    if (!signMessage || !keyInstance) return
    try {
      const sig = await keyInstance.sign(signMessage)
      setSignature(sig)
    } catch (error) {
      console.error('Error signing message:', error)
    }
  }

  const handleVerify = async () => {
    if (!verifyMessage || !verifySignature || !verifyPublicKey || !keyInstance) return
    try {
      const result = await keyInstance.verify(verifyMessage, verifySignature, verifyPublicKey)
      setVerifyResult(result)
    } catch (error) {
      console.error('Error verifying signature:', error)
      setVerifyResult(false)
    }
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
    transactions: History,
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
                <div className="w-10 h-10 rounded-full bg-green-500/20 border border-green-500/50 flex items-center justify-center">
                  <Terminal size={20} className="text-green-400" />
                </div>
                <div>
                  <div className="text-green-400 font-mono text-sm uppercase tracking-wider">USER PROFILE</div>
                  <code className="text-green-500/70 text-xs font-mono">
                    {user.address.slice(0, 8)}...{user.address.slice(-6)}
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
            <div className="space-y-6 animate-fadeIn">
              {/* User Avatar Section */}
              <div className="flex justify-center mb-6">
                <div className="relative">
                  <div className="w-24 h-24 rounded-full bg-gradient-to-br from-green-500/20 to-green-600/20 border-2 border-green-500/50 flex items-center justify-center">
                    <User size={40} className="text-green-400" />
                  </div>
                  <div className="absolute -bottom-2 -right-2 w-8 h-8 rounded-full bg-green-500/20 border border-green-500/50 flex items-center justify-center">
                    <Zap size={16} className="text-green-400" />
                  </div>
                </div>
              </div>

              {/* Full Address Section */}
              <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
                <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase">
                  <Globe size={16} className="animate-pulse" />
                  <span>FULL ADDRESS</span>
                </div>
                <div className="flex items-center gap-2">
                  <code className="flex-1 text-green-400 font-mono text-sm break-all bg-black/50 p-3 border border-green-500/30 rounded-lg">
                    {user.address}
                  </code>
                  <button
                    onClick={() => copyToClipboard(user.address, 'address')}
                    className={`p-3 border rounded-lg transition-all ${
                      copiedField === 'address'
                        ? 'border-green-400 bg-green-500/20 text-green-400'
                        : 'border-green-500/30 text-green-500 hover:bg-green-500/10 hover:border-green-500'
                    }`}
                    title="Copy address"
                  >
                    {copiedField === 'address' ? <CheckCircle size={16} /> : <Copy size={16} />}
                  </button>
                </div>
              </div>

              {/* Crypto Type Section */}
              <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
                <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase">
                  <Shield size={16} />
                  <span>CRYPTO TYPE</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-green-400 font-mono text-lg">{user.crypto_type || 'sr25519'}</span>
                  <Lock size={20} className="text-green-500/50" />
                </div>
              </div>

              {/* Public Key Section */}
              {keyInstance && (
                <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
                  <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase">
                    <KeyIcon size={16} />
                    <span>PUBLIC KEY</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 text-green-400 font-mono text-xs break-all bg-black/50 p-3 border border-green-500/30 rounded-lg">
                      {keyInstance.public_key}
                    </code>
                    <button
                      onClick={() => copyToClipboard(keyInstance.public_key, 'publicKey')}
                      className={`p-3 border rounded-lg transition-all ${
                        copiedField === 'publicKey'
                          ? 'border-green-400 bg-green-500/20 text-green-400'
                          : 'border-green-500/30 text-green-500 hover:bg-green-500/10 hover:border-green-500'
                      }`}
                      title="Copy public key"
                    >
                      {copiedField === 'publicKey' ? <CheckCircle size={16} /> : <Copy size={16} />}
                    </button>
                  </div>
                </div>
              )}

              {/* Logout Button */}
              <div className="pt-6">
                <button
                  onClick={handleLogout}
                  className="w-full py-3 border border-green-500/50 text-green-400 hover:bg-green-500/10 hover:border-green-500 transition-all rounded-lg font-mono uppercase flex items-center justify-center gap-2 group"
                >
                  <LogOut size={20} className="group-hover:rotate-12 transition-transform" />
                  <span>LOGOUT</span>
                </button>
              </div>
            </div>
          )}

          {/* Sign/Verify Tab */}
          {activeTab === 'sign' && (
            <div className="space-y-6 animate-fadeIn">
              {/* Sign Section */}
              <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
                <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-3">
                  <FileSignature size={16} />
                  <span>SIGN MESSAGE</span>
                </div>
                <textarea
                  value={signMessage}
                  onChange={(e) => setSignMessage(e.target.value)}
                  placeholder="Enter message to sign..."
                  className="w-full h-24 bg-black/50 border border-green-500/30 rounded-lg p-3 text-green-400 font-mono text-sm placeholder-green-600/50 focus:outline-none focus:border-green-500 transition-all"
                />
                <button
                  onClick={handleSign}
                  disabled={!signMessage}
                  className="w-full py-2 border border-green-500/50 text-green-400 hover:bg-green-500/10 hover:border-green-500 transition-all rounded-lg font-mono uppercase disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Sign Message
                </button>
                {signature && (
                  <div className="space-y-2 mt-4 p-3 bg-green-500/5 rounded-lg border border-green-500/20">
                    <div className="text-green-500/70 text-sm font-mono uppercase">Signature:</div>
                    <div className="flex items-center gap-2">
                      <code className="flex-1 text-green-400 font-mono text-xs break-all bg-black/50 p-3 border border-green-500/30 rounded-lg">
                        {signature}
                      </code>
                      <button
                        onClick={() => copyToClipboard(signature, 'signature')}
                        className={`p-3 border rounded-lg transition-all ${
                          copiedField === 'signature'
                            ? 'border-green-400 bg-green-500/20 text-green-400'
                            : 'border-green-500/30 text-green-500 hover:bg-green-500/10 hover:border-green-500'
                        }`}
                        title="Copy signature"
                      >
                        {copiedField === 'signature' ? <CheckCircle size={16} /> : <Copy size={16} />}
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Verify Section */}
              <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
                <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-3">
                  <CheckCircle size={16} />
                  <span>VERIFY SIGNATURE</span>
                </div>
                <textarea
                  value={verifyMessage}
                  onChange={(e) => setVerifyMessage(e.target.value)}
                  placeholder="Enter message to verify..."
                  className="w-full h-20 bg-black/50 border border-green-500/30 rounded-lg p-3 text-green-400 font-mono text-sm placeholder-green-600/50 focus:outline-none focus:border-green-500 transition-all"
                />
                <input
                  type="text"
                  value={verifySignature}
                  onChange={(e) => setVerifySignature(e.target.value)}
                  placeholder="Enter signature..."
                  className="w-full bg-black/50 border border-green-500/30 rounded-lg p-3 text-green-400 font-mono text-sm placeholder-green-600/50 focus:outline-none focus:border-green-500 transition-all"
                />
                <input
                  type="text"
                  value={verifyPublicKey}
                  onChange={(e) => setVerifyPublicKey(e.target.value)}
                  placeholder="Enter public key (leave empty to use your own)..."
                  className="w-full bg-black/50 border border-green-500/30 rounded-lg p-3 text-green-400 font-mono text-sm placeholder-green-600/50 focus:outline-none focus:border-green-500 transition-all"
                />
                <button
                  onClick={handleVerify}
                  disabled={!verifyMessage || !verifySignature}
                  className="w-full py-2 border border-green-500/50 text-green-400 hover:bg-green-500/10 hover:border-green-500 transition-all rounded-lg font-mono uppercase disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Verify Signature
                </button>
                {verifyResult !== null && (
                  <div className={`text-center p-3 border rounded-lg font-mono transition-all ${
                    verifyResult 
                      ? 'border-green-500 bg-green-500/10 text-green-400' 
                      : 'border-red-500 bg-red-500/10 text-red-400'
                  }`}>
                    {verifyResult ? '✓ SIGNATURE VALID' : '✗ SIGNATURE INVALID'}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Transactions Tab */}
          {activeTab === 'transactions' && (
            <div className="space-y-4 animate-fadeIn">
              <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-4">
                <History size={16} className="animate-spin-slow" />
                <span>TRANSACTION HISTORY</span>
              </div>
              {transactions.length === 0 ? (
                <div className="text-center py-12 text-green-600/50 font-mono">
                  <History size={48} className="mx-auto mb-4 opacity-50" />
                  <p>No transactions found</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {transactions.map((tx, index) => (
                    <div key={index} className="p-4 border border-green-500/30 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent hover:border-green-500/50 transition-all">
                      <div className="flex justify-between items-center">
                        <span className="text-green-400 font-mono text-sm font-bold">{tx.type}</span>
                        <span className="text-green-600/70 font-mono text-xs">{tx.timestamp}</span>
                      </div>
                      <div className="text-green-600/50 font-mono text-xs mt-2">
                        {tx.hash}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Modules Tab */}
          {activeTab === 'modules' && (
            <div className="space-y-4 animate-fadeIn">
              <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-4">
                <Package size={16} />
                <span>MY MODULES</span>
              </div>
              {userModules.length === 0 ? (
                <div className="text-center py-12 text-green-600/50 font-mono">
                  <Package size={48} className="mx-auto mb-4 opacity-50" />
                  <p>No modules created yet</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {userModules.map((module, index) => (
                    <div key={index} className="p-4 border border-green-500/30 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent hover:border-green-500/50 transition-all cursor-pointer group">
                      <div className="flex justify-between items-center">
                        <span className="text-green-400 font-mono text-sm font-bold group-hover:text-green-300">{module.name}</span>
                        <span className="text-green-600/70 font-mono text-xs">{module.version}</span>
                      </div>
                      {module.description && (
                        <div className="text-green-600/70 font-mono text-xs mt-2">
                          {module.description}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
        
        {/* Visual Resize Indicator */}
        {isDragging && (
          <div className="absolute top-0 left-0 h-full w-full pointer-events-none">
            <div className="absolute left-0 top-0 h-full w-1 bg-green-500 animate-pulse" />
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes spin-slow {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }

        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }

        .animate-spin-slow {
          animation: spin-slow 3s linear infinite;
        }

        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }

        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(0, 255, 0, 0.1);
          border-radius: 3px;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(0, 255, 0, 0.3);
          border-radius: 3px;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(0, 255, 0, 0.5);
        }
      `}</style>
    </>
  )
}