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

type TabType = 'run' | 'code'

export const ModuleSchema = ({mod}: Record<string, any>) => {
  const [selectedFunction, setSelectedFunction] = useState<string>('')
  const [params, setParams] = useState<Record<string, any>>({})
  const [response, setResponse] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [authHeaders, setAuthHeaders] = useState<any>(null)
  const [urlParams, setUrlParams] = useState<string>('')
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [activeTab, setActiveTab] = useState<TabType>('run')
  
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

  // if none then choose the first function
  if (!selectedFunction && Object.keys(filteredSchema).length > 0) {
    setSelectedFunction(Object.keys(filteredSchema)[0])
    // initialize params
    const firstFn = filteredSchema[Object.keys(filteredSchema)[0]]
    if (firstFn && firstFn.input) {
      const defaultParams: Record<string, any> = {}
      Object.entries(firstFn.input).forEach(([param, details]) => {
        if (details.value !== '_empty' && details.value !== undefined) {
          defaultParams[param] = details.value
        }
      })
      setParams(defaultParams)
    }
  }
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


  // Modern dark theme colors
  
const colors = {
  primary: '#39ff14',       // neon green glow
  secondary: '#00ffaa',     // teal-green accent
  accent: '#00eaff',        // neon cyan
  danger: '#ff0055',        // hot magenta-red
  warning: '#ffb400',       // deep amber
  background: '#000000',    // pure black
  surface: '#0d0d0d',       // slightly lifted dark
  surfaceHover: '#1a1a1a',  // hover contrast
  border: '#262626',        // subtle border
  borderActive: '#39ff14',  // glowing active border
  text: '#e6ffe6',          // pale neon-green text
  textSecondary: '#39ff14', // main neon highlight
  textMuted: '#666666',     // muted gray
  codeBg: '#050505',        // near-black code block
  glow: '#00ffaa44',        // translucent neon aura
  runButtonBg: '#39ff14', // glowing run button
};


  return (
    <div className="flex flex-col h-full font-mono" style={{ backgroundColor: colors.background }}>
      {/* Search Bar at the very top */}
      <div className="p-6 pb-0">
        <div className="relative">
          <input
            type="text"
            placeholder="Search functions..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-3 pl-10 rounded-lg font-mono text-sm"
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
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5" style={{ color: colors.textMuted }} />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-3 top-1/2 -translate-y-1/2"
            >
              <XMarkIcon className="w-5 h-5" style={{ color: colors.textMuted }} />
            </button>
          )}
        </div>
      </div>

      {/* Function Tags */}
      <div className="p-6 pt-4">
        <div className="flex flex-wrap gap-2">
          {searchedFunctions.length > 0 ? (
            searchedFunctions.map(fn => (
              <motion.button
                key={fn}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => {
                  setSelectedFunction(fn)
                  initializeParams(fn)
                  setResponse(null)
                  setError('')
                  setAuthHeaders(null)
                  setUrlParams('')
                  setActiveTab('run')
                }}
                className="px-4 py-2 rounded-full font-medium text-sm transition-all"
                style={{
                  backgroundColor: selectedFunction === fn ? colors.primary : colors.surface,
                  color: selectedFunction === fn ? colors.background : colors.text,
                  border: `2px solid ${selectedFunction === fn ? colors.primary : colors.border}`,
                  boxShadow: selectedFunction === fn ? `0 0 15px ${colors.primary}40` : 'none'
                }}
              >
                {fn}
              </motion.button>
            ))
          ) : (
            <div className="text-center py-4 w-full" style={{ color: colors.textMuted }}>
              {searchTerm ? 'No functions match your search' : 'No functions available'}
            </div>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 px-6 pb-6">
        {selectedFunction ? (
          <div className="h-full flex flex-col rounded-lg" style={{ 
            backgroundColor: colors.surface, 
            borderColor: colors.border, 
            borderWidth: '2px', 
            borderStyle: 'solid',
            boxShadow: `0 0 20px ${colors.accent}20`
          }}>
            {/* Tab Content */}
            <div className="flex-1 overflow-auto">
              <AnimatePresence mode="wait">
                {activeTab === 'run' ? (
                  <motion.div
                    key="run"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="p-6 space-y-4"
                  >


                    {/* Input Parameters */}
                    <div className="space-y-4">
                      <h3 className="text-lg font-bold" style={{ color: colors.accent }}>INPUT</h3>
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
                    </div>

                    {/* Execute Button */}
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
                      {loading ? 'EXECUTING...' : 'RUN'}
                    </button>
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
                    className="h-full"
                  >
                    <div className="h-full rounded overflow-hidden" style={{
                      backgroundColor: colors.codeBg,
                      border: `2px solid ${colors.border}`
                    }}>
                      <div className="p-4 border-b flex items-center justify-between" style={{ borderColor: colors.border }}>
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
            </div>

            {/* Tabs at the bottom */}
            <div className="flex border-t" style={{ borderColor: colors.border }}>
              <button
                onClick={() => setActiveTab('run')}
                className="flex-1 px-4 py-3 font-medium text-sm transition-all flex items-center justify-center gap-2"
                style={{
                  backgroundColor: activeTab === 'run' ? `${colors.primary}20` : 'transparent',
                  color: activeTab === 'run' ? colors.text : colors.textMuted,
                  borderRight: `1px solid ${colors.border}`
                }}
              >
                <PlayIcon className="w-4 h-4" />
                Run
              </button>
              {filteredSchema[selectedFunction].code && (
                <button
                  onClick={() => setActiveTab('code')}
                  className="flex-1 px-4 py-3 font-medium text-sm transition-all flex items-center justify-center gap-2"
                  style={{
                    backgroundColor: activeTab === 'code' ? `${colors.primary}20` : 'transparent',
                    color: activeTab === 'code' ? colors.text : colors.textMuted
                  }}
                >
                  <CodeBracketIcon className="w-4 h-4" />
                  Code
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center rounded-lg" style={{ 
            backgroundColor: colors.surface, 
            borderColor: colors.border, 
            borderWidth: '2px', 
            borderStyle: 'solid'
          }}>
            <div className="text-center" style={{ color: colors.textMuted }}>
              <CommandLineIcon className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Select a function to get started</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ModuleSchema