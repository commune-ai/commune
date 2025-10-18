'use client'
import { useState, useEffect } from 'react'
import { Key as KeyIcon, Shield, Copy, CheckCircle, AlertCircle, Zap, Terminal, Lock, RefreshCw } from 'lucide-react'
import { Auth } from '@/app/client/auth'
import { Key } from '@/app/key'
import type { AuthHeaders } from '@/app/client/auth'

interface UserAuthProps {
  keyInstance: Key
}

export const UserAuth = ({ keyInstance }: UserAuthProps) => {
  const [authInstance, setAuthInstance] = useState<Auth | null>(null)
  const [functionName, setFunctionName] = useState('')
  const [functionParams, setFunctionParams] = useState('')
  const [authHeaders, setAuthHeaders] = useState<AuthHeaders | null>(null)
  const [verifyData, setVerifyData] = useState('')
  const [verifyHeaders, setVerifyHeaders] = useState('')
  const [verifyResult, setVerifyResult] = useState<{ success: boolean; message: string } | null>(null)
  const [copiedField, setCopiedField] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isVerifying, setIsVerifying] = useState(false)
  const [crypto_type, setCryptoType] = useState('sr25519')
  const [maxStaleness, setMaxStaleness] = useState(60)

  useEffect(() => {
    if (keyInstance) {
      const auth = new Auth(keyInstance, crypto_type, 'sha256', maxStaleness)
      setAuthInstance(auth)
    }
  }, [keyInstance, crypto_type, maxStaleness])


  const generateAuth = async () => {
    if (!authInstance || !functionName) return
    
    setIsGenerating(true)
    try {
      // Parse function parameters
      let params = {}
      if (functionParams) {
        try {
          params = JSON.parse(functionParams)
        } catch (e) {
          // If not valid JSON, treat as string
          params = { data: functionParams }
        }
      }

      // Create the function call data
      const functionData = {
        fn: functionName,
        params: params,
        timestamp: Date.now()
      }

      // Generate auth headers
      const headers = authInstance.generate(functionData)
      setAuthHeaders(headers)
    } catch (error) {
      console.error('Error generating auth:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  const verifyAuth = async () => {
    if (!authInstance || !verifyHeaders) return
    
    setIsVerifying(true)
    try {
      // Parse the headers
      const headers = JSON.parse(verifyHeaders) as AuthHeaders
      
      // Parse the data if provided
      let data = null
      if (verifyData) {
        try {
          data = JSON.parse(verifyData)
        } catch (e) {
          data = verifyData
        }
      }

      // Verify the headers
      authInstance.verify(headers, data)
      setVerifyResult({ success: true, message: 'Authentication headers are valid!' })
    } catch (error: any) {
      setVerifyResult({ success: false, message: error.message || 'Verification failed' })
    } finally {
      setIsVerifying(false)
    }
  }

  const formatHeaders = (headers: AuthHeaders) => {
    return JSON.stringify(headers, null, 2)
  }

  const generateCurlExample = () => {
    if (!authHeaders || !functionName) return ''
    
    return `curl -X POST https://api.commune.ai/call \\
  -H "Content-Type: application/json" \\
  -H "X-Auth-Data: ${authHeaders.data}" \\
  -H "X-Auth-Time: ${authHeaders.time}" \\
  -H "X-Auth-Key: ${authHeaders.key}" \\
  -H "X-Auth-Signature: ${authHeaders.signature}" \\
  ${authHeaders.crypto_type ? `-H "X-Auth-Crypto-Type: ${authHeaders.crypto_type}" \\` : ''}
  -d '{"fn": "${functionName}", "params": ${functionParams || '{}'}}'`
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Auth Configuration */}
      <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
        <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-3">
          <Shield size={16} />
          <span>AUTH CONFIGURATION</span>
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-green-500/70 font-mono uppercase">Crypto Type</label>
            <select
              value={crypto_type}
              onChange={(e) => setCryptoType(e.target.value)}
              className="w-full mt-1 bg-black/50 border border-green-500/30 rounded px-3 py-2 text-green-400 font-mono text-sm focus:outline-none focus:border-green-500"
            >
              <option value="sr25519">SR25519</option>
              <option value="ed25519">ED25519</option>
            </select>
          </div>
          
          <div>
            <label className="text-xs text-green-500/70 font-mono uppercase">Max Staleness (s)</label>
            <input
              type="number"
              value={maxStaleness}
              onChange={(e) => setMaxStaleness(parseInt(e.target.value) || 60)}
              className="w-full mt-1 bg-black/50 border border-green-500/30 rounded px-3 py-2 text-green-400 font-mono text-sm focus:outline-none focus:border-green-500"
              min="10"
              max="3600"
            />
          </div>
        </div>
      </div>

      {/* Generate Auth Section */}
      <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
        <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-3">
          <KeyIcon size={16} />
          <span>GENERATE AUTH HEADERS</span>
        </div>
        
        <div className="space-y-3">
          <div>
            <label className="text-xs text-green-500/70 font-mono uppercase">Function Name</label>
            <input
              type="text"
              value={functionName}
              onChange={(e) => setFunctionName(e.target.value)}
              placeholder="e.g., module.list, user.profile"
              className="w-full mt-1 bg-black/50 border border-green-500/30 rounded px-3 py-2 text-green-400 font-mono text-sm placeholder-green-600/50 focus:outline-none focus:border-green-500"
            />
          </div>
          
          <div>
            <label className="text-xs text-green-500/70 font-mono uppercase">Function Parameters (JSON)</label>
            <textarea
              value={functionParams}
              onChange={(e) => setFunctionParams(e.target.value)}
              placeholder='{"limit": 10, "offset": 0}'
              className="w-full mt-1 h-20 bg-black/50 border border-green-500/30 rounded p-3 text-green-400 font-mono text-sm placeholder-green-600/50 focus:outline-none focus:border-green-500"
            />
          </div>
          
          <button
            onClick={generateAuth}
            disabled={!functionName || isGenerating}
            className="w-full py-2 border border-green-500/50 text-green-400 hover:bg-green-500/10 hover:border-green-500 transition-all rounded font-mono uppercase disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isGenerating ? (
              <>
                <RefreshCw size={16} className="animate-spin" />
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Zap size={16} />
                <span>Generate Auth Headers</span>
              </>
            )}
          </button>
        </div>
        
        {/* Generated Headers Display */}
        {authHeaders && (
          <div className="mt-4 space-y-3">
            <div className="p-3 bg-green-500/5 rounded border border-green-500/20">
              <div className="flex items-center justify-between mb-2">
                <span className="text-green-500/70 text-sm font-mono uppercase">Generated Headers</span>
                <button
                  onClick={() => copyToClipboard(formatHeaders(authHeaders))}
                  className={`p-2 border rounded transition-all ${
                    copiedField === 'headers'
                      ? 'border-green-400 bg-green-500/20 text-green-400'
                      : 'border-green-500/30 text-green-500 hover:bg-green-500/10 hover:border-green-500'
                  }`}
                  title="Copy headers"
                >
                  {copiedField === 'headers' ? <CheckCircle size={14} /> : <Copy size={14} />}
                </button>
              </div>
              <pre className="text-green-400 font-mono text-xs overflow-x-auto">
                {formatHeaders(authHeaders)}
              </pre>
            </div>
            
            {/* CURL Example */}
            <div className="p-3 bg-black/50 rounded border border-green-500/20">
              <div className="flex items-center justify-between mb-2">
                <span className="text-green-500/70 text-sm font-mono uppercase flex items-center gap-2">
                  <Terminal size={14} />
                  CURL Example
                </span>
                <button
                  onClick={() => copyToClipboard(generateCurlExample())}
                  className={`p-2 border rounded transition-all ${
                    copiedField === 'curl'
                      ? 'border-green-400 bg-green-500/20 text-green-400'
                      : 'border-green-500/30 text-green-500 hover:bg-green-500/10 hover:border-green-500'
                  }`}
                  title="Copy CURL command"
                >
                  {copiedField === 'curl' ? <CheckCircle size={14} /> : <Copy size={14} />}
                </button>
              </div>
              <pre className="text-green-400 font-mono text-xs overflow-x-auto whitespace-pre-wrap">
                {generateCurlExample()}
              </pre>
            </div>
          </div>
        )}
      </div>

      {/* Verify Auth Section */}
      <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
        <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-3">
          <Lock size={16} />
          <span>VERIFY AUTH HEADERS</span>
        </div>
        
        <div className="space-y-3">
          <div>
            <label className="text-xs text-green-500/70 font-mono uppercase">Auth Headers (JSON)</label>
            <textarea
              value={verifyHeaders}
              onChange={(e) => setVerifyHeaders(e.target.value)}
              placeholder='Paste auth headers JSON here...'
              className="w-full mt-1 h-24 bg-black/50 border border-green-500/30 rounded p-3 text-green-400 font-mono text-sm placeholder-green-600/50 focus:outline-none focus:border-green-500"
            />
          </div>
          
          <div>
            <label className="text-xs text-green-500/70 font-mono uppercase">Original Data (Optional)</label>
            <textarea
              value={verifyData}
              onChange={(e) => setVerifyData(e.target.value)}
              placeholder='Original data to verify against...'
              className="w-full mt-1 h-20 bg-black/50 border border-green-500/30 rounded p-3 text-green-400 font-mono text-sm placeholder-green-600/50 focus:outline-none focus:border-green-500"
            />
          </div>
          
          <button
            onClick={verifyAuth}
            disabled={!verifyHeaders || isVerifying}
            className="w-full py-2 border border-green-500/50 text-green-400 hover:bg-green-500/10 hover:border-green-500 transition-all rounded font-mono uppercase disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isVerifying ? (
              <>
                <RefreshCw size={16} className="animate-spin" />
                <span>Verifying...</span>
              </>
            ) : (
              <>
                <Shield size={16} />
                <span>Verify Headers</span>
              </>
            )}
          </button>
          
          {verifyResult && (
            <div className={`p-3 border rounded font-mono text-sm flex items-center gap-2 ${
              verifyResult.success
                ? 'border-green-500 bg-green-500/10 text-green-400'
                : 'border-red-500 bg-red-500/10 text-red-400'
            }`}>
              {verifyResult.success ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
              <span>{verifyResult.message}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default UserAuth