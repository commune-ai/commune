'use client'
import { useState } from 'react'
import { Copy, CheckCircle, FileSignature } from 'lucide-react'
import { Key } from '@/app/key'
import { copyToClipboard } from '@/app/utils' // Import the utility function

interface SignVerifyTabProps {
  keyInstance: Key
}

export const SignVerifyTab = ({ keyInstance }: SignVerifyTabProps) => {
  const [copiedField, setCopiedField] = useState<string | null>(null)
  const [signMessage, setSignMessage] = useState('')
  const [signature, setSignature] = useState('')
  const [verifyMessage, setVerifyMessage] = useState('')
  const [verifySignature, setVerifySignature] = useState('')
  const [verifyPublicKey, setVerifyPublicKey] = useState('')
  const [verifyResult, setVerifyResult] = useState<boolean | null>(null)

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

  return (
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
  )
}