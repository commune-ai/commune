'use client'
import { useState } from 'react'
import { User, Copy, CheckCircle, Globe, Shield, Lock, Key as KeyIcon, LogOut, Zap } from 'lucide-react'
import { Key } from '@/app/key'
import type { User as UserType } from '@/app/types/user'
import { copyToClipboard } from '@/app/utils' // Import the utility function

interface ProfileTabProps {
  user: UserType
  keyInstance: Key
  onLogout: () => void
}

export const ProfileTab = ({ user, keyInstance, onLogout }: ProfileTabProps) => {
  const [copiedField, setCopiedField] = useState<string | null>(null)
  return (
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
            {keyInstance.address}
          </code>
          <button
            onClick={() => copyToClipboard(keyInstance.address)}
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
          <span className="text-green-400 font-mono text-lg">{keyInstance.crypto_type || 'sr25519'}</span>
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
              onClick={() => copyToClipboard(keyInstance.public_key)}
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
          onClick={onLogout}
          className="w-full py-3 border border-green-500/50 text-green-400 hover:bg-green-500/10 hover:border-green-500 transition-all rounded-lg font-mono uppercase flex items-center justify-center gap-2 group"
        >
          <LogOut size={20} className="group-hover:rotate-12 transition-transform" />
          <span>LOGOUT</span>
        </button>
      </div>
    </div>
  )
}