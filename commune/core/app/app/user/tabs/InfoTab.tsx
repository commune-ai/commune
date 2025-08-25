
'use client'
import { useState, useEffect } from 'react'
import { X, Copy, LogOut, Key as KeyIcon, Shield, Globe, FileSignature, CheckCircle, History, Package, User, Terminal, Lock, Zap } from 'lucide-react'
import { Key } from '@/app/key'
import type { User } from '@/app/types/user'
import { CopyButton } from '@/app/components/CopyButton'


const shortenAddress = (address: string, length: number=10) => {
    if (!address) return '';
    return `${address.slice(0, length)}...${address.slice(-length)}`;
    }

export const InfoTab = ({ keyInstance }: { keyInstance: Key }) => {

  const [copiedField, setCopiedField] = useState<string | null>(null)

return (
              <div className="space-y-6 animate-fadeIn">
              {/* Full Address Section */}
              <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
                <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase">
                  <Globe size={16} className="animate-pulse" />
                  <span>ADDRESS</span>
                </div>
                <div className="flex items-center gap-2">
                  <code className="flex-1 text-green-400 font-mono text-sm break-all bg-black/50 p-3 border border-green-500/30 rounded-lg">
                    {shortenAddress(keyInstance.address)}
                  </code>
                  <CopyButton
                    code={keyInstance.address as string}
                  />
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
                      {shortenAddress(keyInstance.public_key)}
                    </code>
                    <CopyButton
                      code={keyInstance.public_key as string}

                    />

                  </div>
                </div>
              )}

            </div>
)
}
