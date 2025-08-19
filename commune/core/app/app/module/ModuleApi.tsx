'use client'
import { Client } from '@/app/client/client'
import { useState, useMemo } from 'react'
import { CopyButton } from '@/app/components/CopyButton'
import { useUserContext } from '@/app/context/UserContext'
import { Auth } from '@/app/key'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  MagnifyingGlassIcon, 
  CodeBracketIcon, 
  PlayIcon,
  CommandLineIcon,
  DocumentTextIcon,
  ChevronRightIcon,
  XMarkIcon
} from '@heroicons/react/24/outline'

type SchemaType = {
  input: Record<string, {
    value: any
    type: string
  }>
  output: {
    value: any
    type: string
  }
  code?: string
  hash?: string
}

type TabType = 'playground' | 'code'

export const ModuleSchema = ({mod}: Record<string, any>) => {
  const [selectedFunction, setSelectedFunction] = useState<string>('')
  const [params, setParams] = useState<Record<string, any>>({})
  const [response, setResponse] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [authHeaders, setAuthHeaders] = useState<any>(null)
  const [urlParams, setUrlParams] = useState<string>('')
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [activeTab, setActiveTab] = useState<TabType>('playground')
  
  let schema: Record<string, SchemaType> = mod.schema || {}
  
  const { user, keyInstance, signIn, signOut, isLoading } = useUserContext()
  
  // Filter out 'self', 'cls' methods and parameters
  const filteredSchema = Object.entries(schema).reduce((acc, [key, value]) => {
    if (key !== 'self' && key !== 'cls') {
      // Also filter out self and cls from input parameters
      if (value.input) {
        const filteredInput = Object.entries(value.input).reduce((inputAcc, [paramKey, paramValue]) => {
          if (paramKey !== 'self' && paramKey !== 'cls') {
            inputAcc[paramKey] = paramValue
          }
          return inputAcc
        }, {} as typeof value.input)
        acc[key] = { ...value, input: filteredInput }
      } else {
        acc[key] = value
      }
    }
    return acc
  }, {} as Record<string, SchemaType>)

  // Filter functions based on search term
  const searchedFunctions = useMemo(() => {
    if (!searchTerm) return Object.keys(filteredSchema)
    return Object.keys(filteredSchema).filter(fn => 
      fn.toLowerCase().includes(searchTerm.toLowerCase())
    )
  }, [filteredSchema, searchTerm])

  // Handle parameter input change
  const handleParamChange = (paramName: string, value: string) => {
    setParams({ ...params, [paramName]: value })
  }
  
  // Initialize params with defaults when function is selected
  const initializeParams = (fnName: string) => {
    const fnSchema = filteredSchema[fnName]
    if (fnSchema && fnSchema.input) {
      const defaultParams: Record<string, any> = {}
      Object.entries(fnSchema.input).forEach(([param, details]) => {
        if (details.value !== '_empty' && details.value !== undefined) {
          defaultParams[param] = details.value
        }
      })
      setParams(defaultParams)
    }
  }
  
  // Build URL with parameters
  const buildUrlParams = () => {
    const queryParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== '') {
        queryParams.append(key, String(value))
      }
    })
    return queryParams.toString()
  }
  
  // Execute the selected function
  const executeFunction = async () => {
    setLoading(true)
    setError('')
    setAuthHeaders(null)
    try {
      const client = new Client()
      
      // Create auth headers for the request
      const auth = new Auth(keyInstance)
      const headers = auth.generate({
        fn: selectedFunction,
        params: params
      })
      
      setAuthHeaders(headers)
      setUrlParams(buildUrlParams())
      
      const response = await client.call("call", {fn: selectedFunction, params})
      setResponse(response)
    } catch (err: any) {
      setError(err.message || 'Failed to execute function')
    } finally {
      setLoading(false)
    }
  }

  // Get function description
  const getFunctionDescription = (fnName: string) => {
    const fn = filteredSchema[fnName]
    if (!fn) return 'No description available.'
    
    const details = []
    if (fn.hash) details.push(`Hash: ${fn.hash.substring(0, 12)}...`)
    details.push(`Returns: ${fn.output.type}`)
    details.push(`Parameters: ${Object.keys(fn.input).length}`)
    
    return details.join(' | ')
  }

  // Modern dark theme colors
  const colors = {
    primary: '#00ff41',
    secondary: '#008f11',
    accent: '#00d9ff',
    danger: '#ff0040',
    warning: '#ffaa00',
    background: '#000000',
    surface: '#0a0a0a',
    surfaceHover: '#141414',
    border: '#1a1a1a',
    borderActive: '#00ff41',
    text: '#00ff41',
    textSecondary: '#008f11',
    textMuted: '#666666',
    codeBg: '#050505'
  }

  return (
    <div className="flex gap-6 h-full font-mono" style={{ backgroundColor: colors.background }}>
      {/* Left Panel - Function List with Search */}
      <div className="w-96 flex flex-col space-y-4 p-6 rounded-lg" style={{ 
        backgroundColor: colors.surface, 
        borderColor: colors.border, 
        borderWidth: '2px', 
        borderStyle: 'solid',
        boxShadow: `0 0 20px ${colors.primary}20`
      }}>
        <h2 className="text-xl font-bold uppercase tracking-wider flex items-center gap-2" style={{ 
          color: colors.text,
          textShadow: `0 0 10px ${colors.primary}`
        }}>
          <CommandLineIcon className="w-6 h-6" />
          FUNCTIONS
        </h2>
        
        {/* Search Bar */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search functions..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 pl-10 rounded font-mono text-sm"
            style={{
              backgroundColor: colors.codeBg,
              color: colors.text,
              border: `2px solid ${colors.border}`,
              outline: 'none'
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = colors.borderActive
              e.currentTarget.style.boxShadow = `0 0 10px ${colors.primary}40`
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = colors.border
              e.currentTarget.style.boxShadow = 'none'
            }}
          />
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: colors.textMuted }} />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-3 top-1/2 -translate-y-1/2"
            >
              <XMarkIcon className="w-4 h-4" style={{ color: colors.textMuted }} />
            </button>
          )}
        </div>
        
        {/* Function List */}
        <div className="flex-1 overflow-y-auto space-y-2 pr-2" style={{
          scrollbarWidth: 'thin',
          scrollbarColor: `${colors.border} ${colors.background}`
        }}>
          {searchedFunctions.length > 0 ? (
            searchedFunctions.map(fn => (
              <motion.div
                key={fn}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                onClick={() => {
                  setSelectedFunction(fn)
                  initializeParams(fn)
                  setResponse(null)
                  setError('')
                  setAuthHeaders(null)
                  setUrlParams('')
                  setActiveTab('playground')
                }}
                className="p-3 rounded cursor-pointer transition-all group"
                style={{
                  backgroundColor: selectedFunction === fn ? colors.surfaceHover : colors.background,
                  border: `2px solid ${selectedFunction === fn ? colors.borderActive : colors.border}`,
                  color: selectedFunction === fn ? colors.text : colors.textSecondary,
                  boxShadow: selectedFunction === fn ? `0 0 15px ${colors.primary}40` : 'none'
                }}
                whileHover={{
                  scale: 1.02,
                  transition: { duration: 0.2 }
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-sm">{fn}</span>
                  <ChevronRightIcon className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
                <p className="text-xs mt-1 opacity-70">
                  {Object.keys(filteredSchema[fn].input).length} params | {filteredSchema[fn].output.type}
                </p>
              </motion.div>
            ))
          ) : (
            <div className="text-center py-8" style={{ color: colors.textMuted }}>
              {searchTerm ? 'No functions match your search' : 'No functions available'}
            </div>
          )}
        </div>
        
        {/* Module Info */}
        {mod.name && (
          <div className="pt-4" style={{ borderTop: `2px solid ${colors.border}` }}>
            <h3 className="text-sm font-bold mb-2" style={{ color: colors.accent }}>MODULE INFO</h3>
            <p className="text-sm" style={{ color: colors.text }}>{mod.name}</p>
            {mod.description && (
              <p className="text-xs mt-1" style={{ color: colors.textMuted }}>{mod.description}</p>
            )}
          </div>
        )}
      </div>

      {/* Right Panel - Function Details */}
      <div className="flex-1 flex flex-col space-y-4 p-6 rounded-lg" style={{ 
        backgroundColor: colors.surface, 
        borderColor: colors.border, 
        borderWidth: '2px', 
        borderStyle: 'solid',
        boxShadow: `0 0 20px ${colors.accent}20`
      }}>
        {selectedFunction ? (
          <>
            {/* Function Header with Tabs */}
            <div>
              <h2 className="text-xl font-bold uppercase tracking-wider mb-2" style={{ 
                color: colors.text,
                textShadow: `0 0 10px ${colors.accent}`
              }}>
                {selectedFunction}
              </h2>
              <p className="text-sm mb-4" style={{ color: colors.textMuted }}>
                {getFunctionDescription(selectedFunction)}
              </p>
              
              {/* Tabs */}
              <div className="flex gap-2 border-b" style={{ borderColor: colors.border }}>
                <button
                  onClick={() => setActiveTab('playground')}
                  className="px-4 py-2 font-medium text-sm transition-all relative"
                  style={{
                    color: activeTab === 'playground' ? colors.text : colors.textMuted,
                    borderBottom: activeTab === 'playground' ? `2px solid ${colors.primary}` : 'none',
                    marginBottom: '-2px'
                  }}
                >
                  <div className="flex items-center gap-2">
                    <PlayIcon className="w-4 h-4" />
                    Playground
                  </div>
                </button>
                {filteredSchema[selectedFunction].code && (
                  <button
                    onClick={() => setActiveTab('code')}
                    className="px-4 py-2 font-medium text-sm transition-all relative"
                    style={{
                      color: activeTab === 'code' ? colors.text : colors.textMuted,
                      borderBottom: activeTab === 'code' ? `2px solid ${colors.primary}` : 'none',
                      marginBottom: '-2px'
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <CodeBracketIcon className="w-4 h-4" />
                      Code
                    </div>
                  </button>
                )}
              </div>
            </div>

            {/* Tab Content */}
            <AnimatePresence mode="wait">
              {activeTab === 'playground' ? (
                <motion.div
                  key="playground"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex-1 flex flex-col space-y-4"
                >
                  {/* Execute Button at Top */}
                  <button
                    onClick={executeFunction}
                    disabled={loading}
                    className="w-full px-6 py-3 rounded-lg font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2"
                    style={{
                      backgroundColor: loading ? colors.secondary : colors.primary,
                      color: colors.background,
                      cursor: loading ? 'not-allowed' : 'pointer',
                      opacity: loading ? 0.7 : 1,
                      border: `2px solid ${loading ? colors.secondary : colors.primary}`,
                      boxShadow: loading ? 'none' : `0 0 20px ${colors.primary}60`,
                      textShadow: `0 0 5px ${colors.background}`
                    }}
                    onMouseEnter={(e) => {
                      if (!loading) {
                        e.currentTarget.style.backgroundColor = colors.accent
                        e.currentTarget.style.borderColor = colors.accent
                        e.currentTarget.style.boxShadow = `0 0 30px ${colors.accent}80`
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!loading) {
                        e.currentTarget.style.backgroundColor = colors.primary
                        e.currentTarget.style.borderColor = colors.primary
                        e.currentTarget.style.boxShadow = `0 0 20px ${colors.primary}60`
                      }
                    }}
                  >
                    <PlayIcon className="w-5 h-5" />
                    {loading ? 'EXECUTING...' : 'EXECUTE FUNCTION'}
                  </button>

                  {/* Input Parameters */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-bold flex items-center gap-2" style={{ color: colors.accent }}>
                      <DocumentTextIcon className="w-5 h-5" />
                      PARAMETERS
                    </h3>
                    
                    {/* URL Parameters Display */}
                    {urlParams && (
                      <div className="p-3 rounded" style={{
                        backgroundColor: colors.codeBg,
                        border: `1px solid ${colors.border}`
                      }}>
                        <p className="text-xs mb-1" style={{ color: colors.textSecondary }}>URL PARAMS:</p>
                        <code className="text-xs" style={{ color: colors.text }}>?{urlParams}</code>
                      </div>
                    )}
                    
                    {/* Parameter Inputs */}
                    <div className="space-y-3">
                      {Object.entries(filteredSchema[selectedFunction].input).map(([param, details]) => (
                        <div key={param} className="space-y-2">
                          <label className="block text-sm font-bold" style={{ color: colors.text }}>
                            {param}
                            <span className="ml-2 font-normal" style={{ color: colors.accent }}>[{details.type}]</span>
                          </label>
                          {details.value !== '_empty' && details.value !== undefined && (
                            <p className="text-xs" style={{ color: colors.textMuted }}>Default: {String(details.value)}</p>
                          )}
                          <input
                            type="text"
                            value={params[param] !== undefined ? params[param] : ''}
                            onChange={(e) => handleParamChange(param, e.target.value)}
                            placeholder={details.value !== '_empty' && details.value !== undefined ? `Default: ${details.value}` : `Enter ${param}`}
                            className="w-full px-4 py-2 rounded font-mono text-sm"
                            style={{
                              backgroundColor: colors.codeBg,
                              color: colors.text,
                              border: `2px solid ${colors.border}`,
                              outline: 'none'
                            }}
                            onFocus={(e) => {
                              e.currentTarget.style.borderColor = colors.borderActive
                              e.currentTarget.style.boxShadow = `0 0 10px ${colors.primary}40`
                            }}
                            onBlur={(e) => {
                              e.currentTarget.style.borderColor = colors.border
                              e.currentTarget.style.boxShadow = 'none'
                            }}
                          />
                        </div>
                      ))}
                    </div>

                    {/* Auth Headers Display */}
                    {authHeaders && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-sm" style={{ color: colors.textSecondary }}>HEADERS</h4>
                          <CopyButton code={JSON.stringify(authHeaders, null, 2)} />
                        </div>
                        <pre className="p-3 rounded overflow-x-auto max-h-24 text-xs" style={{
                          backgroundColor: colors.codeBg,
                          border: `1px solid ${colors.border}`,
                          color: colors.textSecondary
                        }}>
                          <code>{JSON.stringify(authHeaders, null, 2)}</code>
                        </pre>
                      </div>
                    )}
                  </div>

                  {/* Output Section */}
                  {(response || error) && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-bold" style={{ color: colors.accent }}>OUTPUT</h3>
                      
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-bold text-sm" style={{ color: error ? colors.danger : colors.textSecondary }}>
                            {error ? 'ERROR' : 'RESPONSE'}
                          </h4>
                          <CopyButton code={JSON.stringify(response || error, null, 2)} />
                        </div>
                        <pre className="p-4 rounded overflow-x-auto max-h-64" style={{
                          backgroundColor: colors.codeBg,
                          border: `2px solid ${error ? colors.danger : colors.accent}`,
                          boxShadow: error ? `0 0 15px ${colors.danger}40` : `0 0 15px ${colors.accent}40`
                        }}>
                          <code className="text-sm" style={{ color: error ? colors.danger : colors.text }}>
                            {JSON.stringify(response || error, null, 2)}
                          </code>
                        </pre>
                      </div>
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="code"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex-1"
                >
                  <div className="h-full rounded overflow-hidden" style={{
                    backgroundColor: colors.codeBg,
                    border: `2px solid ${colors.border}`
                  }}>
                    <div className="p-4 border-b flex items-center justify-between" style={{ borderColor: colors.border }}>
                      <h3 className="font-bold text-sm" style={{ color: colors.text }}>FUNCTION CODE</h3>
                      <CopyButton code={filteredSchema[selectedFunction].code || ''} />
                    </div>
                    <pre className="p-4 overflow-auto h-full">
                      <code className="text-sm" style={{ color: colors.text }}>
                        {filteredSchema[selectedFunction].code || 'No code available'}
                      </code>
                    </pre>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center" style={{ color: colors.textMuted }}>
              <CommandLineIcon className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Select a function from the left panel to get started</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ModuleSchema