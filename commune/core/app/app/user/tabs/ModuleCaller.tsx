'use client'
import { useState } from 'react'
import { Send, Copy, CheckCircle, Terminal, Zap, AlertCircle } from 'lucide-react'
import { Key } from '@/app/key'
import { Auth, AuthHeaders } from '@/app/client/auth'
import { Client } from '@/app/client'
import { copyToClipboard } from '@/app/utils' // Import the utility function
import { CopyButton } from '@/app/components'

interface ModuleCallerProps {
  keyInstance: Key
}

export const ModuleCaller = ({ keyInstance }: ModuleCallerProps) => {
  const [moduleUrl, setModuleUrl] = useState('0.0.0.0:8000')
  const [functionName, setFunctionName] = useState('info')
  const [functionParams, setFunctionParams] = useState('{}')
  const [authHeaders, setAuthHeaders] = useState<AuthHeaders | null>(null)
  const [response, setResponse] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copiedField, setCopiedField] = useState<string | null>(null)

  const generateAuthAndCall = async () => {
    if (!moduleUrl || !functionName || !keyInstance) return
    
    setIsLoading(true)
    setError(null)
    setResponse(null)
    
    try {
      // Parse function parameters
      let params = {}
      if (functionParams) {
        try {
          params = JSON.parse(functionParams)
        } catch (e) {
          params = { data: functionParams }
        }
      }

      // Create auth instance
      const auth = new Auth(keyInstance, 'sha256', 60)
      
      // Create the function call data
      const functionData = {
        fn: functionName,
        params: params,
      }

      // Generate auth headers
      let headers = auth.generate(functionData)
      headers.verified = auth.verify(headers)
      setAuthHeaders(headers)

      const client = new Client(moduleUrl,keyInstance)
      // Make the API call
      const apiResponse = await client.call(functionName, params, headers)


      const responseData = await apiResponse.json()
      console.log('API Response:', responseData)
      setResponse(responseData)
      
      if (!apiResponse.ok) {
        setError(`API Error: ${apiResponse.status} - ${apiResponse.statusText}`)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to call module')
    } finally {
      setIsLoading(false)
    }
  }

  const authHeadersString: string = authHeaders ? JSON.stringify(authHeaders, null, 2) : ''
  const getCurlCommand = () => {
    if (!authHeaders || !moduleUrl || !functionName) return ''
    return `curl -X POST ${moduleUrl}/${functionName} \\
  -H "time: ${authHeaders.time}" \
  -H "key: ${authHeaders.key}" \\
  -H "signature: ${authHeaders.signature}" \\
  -H "crypto_type: ${authHeaders.crypto_type || ''}" \\
  -H "hash_type: ${authHeaders.hash_type || ''}" \\
  -H "data: ${authHeaders.data}" \\
  -d '${functionParams || '{}'}'
`

  }
  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Module Configuration */}
      <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
        <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-3">
          <Terminal size={16} />
          <span>MODULE CALLER</span>
        </div>
        
        <div className="space-y-3">
          <div>
            <label className="text-xs text-green-500/70 font-mono uppercase">Module URL</label>
            <input
              type="text"
              value={moduleUrl}
              onChange={(e) => setModuleUrl(e.target.value)}
              placeholder="https://api.commune.ai/module/endpoint"
              className="w-full mt-1 bg-black/50 border border-green-500/30 rounded px-3 py-2 text-green-400 font-mono text-sm placeholder-green-600/50 focus:outline-none focus:border-green-500"
            />
          </div>
          
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
            onClick={generateAuthAndCall}
            disabled={!moduleUrl || !functionName || isLoading}
            className="w-full py-2 border border-green-500/50 text-green-400 hover:bg-green-500/10 hover:border-green-500 transition-all rounded font-mono uppercase disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Zap size={16} className="animate-spin" />
                <span>Calling Module...</span>
              </>
            ) : (
              <>
                <Send size={16} />
                <span>Call</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Auth Headers Display */}
      {authHeaders && (
        <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase">
              <Zap size={16} />
              <span>Generated Auth Headers</span>
            </div>
            <CopyButton  code={JSON.stringify(authHeaders, null, 2)}/>
          </div>
          <pre className="text-green-400 font-mono text-xs overflow-x-auto bg-black/50 p-3 rounded border border-green-500/20">
            {JSON.stringify(authHeaders, null, 2)}
          </pre>
        </div>
      )}

      {/* CURL Command */}
      {authHeaders && (
        <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase">
              <Terminal size={16} />
              <span>CURL Command</span>
            </div>
            <button
              onClick={() => 
                (getCurlCommand(), 'curl')}
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
          <pre className="text-green-400 font-mono text-xs overflow-x-auto bg-black/50 p-3 rounded border border-green-500/20 whitespace-pre-wrap">
            {getCurlCommand()}
          </pre>
        </div>
      )}

      {/* Response Display */}
      {(response || error) && (
        <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
          <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase">
            {error ? <AlertCircle size={16} className="text-red-500" /> : <CheckCircle size={16} />}
            <span>{error ? 'Error' : 'Response'}</span>
          </div>
          {error ? (
            <div className="text-red-400 font-mono text-sm">{error}</div>
          ) : (
            <pre className="text-green-400 font-mono text-xs overflow-x-auto bg-black/50 p-3 rounded border border-green-500/20">
              {JSON.stringify(response, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

export default ModuleCaller