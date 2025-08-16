'use client'
import { Client } from '@/app/client/client'
import { useState } from 'react'
import { CopyButton } from '@/app/components/CopyButton'

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

export const ModuleSchema = ({mod}: Record<string, any>) => {
  const [selectedFunction, setSelectedFunction] = useState<string>('')
  const [params, setParams] = useState<Record<string, any>>({})
  const [response, setResponse] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [showDescription, setShowDescription] = useState<boolean>(false)
  
  let schema: Record<string, SchemaType> = mod.schema || {}
  
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
  
  // Execute the selected function
  const executeFunction = async () => {
    setLoading(true)
    setError('')
    try {
      const client = new Client()
      const response = await client.call(selectedFunction, params)
      setResponse(response)
    } catch (err: any) {
      setError(err.message || 'Failed to execute function')
    } finally {
      setLoading(false)
    }
  }

  // Get function description (mock - you can enhance this)
  const getFunctionDescription = (fnName: string) => {
    const descriptions: Record<string, string> = {
      'default': 'Select a function to see its description and parameters.',
      // Add more function descriptions here
    }
    return descriptions[fnName] || `Execute ${fnName} with the specified parameters.`
  }

  // IBM Carbon Design inspired styling
  const ibmColors = {
    primary: '#0f62fe',
    secondary: '#393939',
    success: '#24a148',
    danger: '#da1e28',
    warning: '#f1c21b',
    background: '#161616',
    surface: '#262626',
    border: '#393939',
    text: '#f4f4f4',
    textSecondary: '#c6c6c6'
  }

  return (
    <div className="flex gap-4 h-full" style={{ backgroundColor: ibmColors.background }}>
      {/* Left Panel - Function Selection */}
      <div className="w-80 space-y-4 p-6 rounded" style={{ backgroundColor: ibmColors.surface, borderColor: ibmColors.border, borderWidth: '1px', borderStyle: 'solid' }}>
        <h2 className="text-xl font-semibold" style={{ color: ibmColors.text, borderBottom: `1px solid ${ibmColors.border}`, paddingBottom: '8px' }}>
          functions
        </h2>
        
        <div className="space-y-2">
          {Object.keys(filteredSchema).length > 0 ? (
            Object.keys(filteredSchema).map(fn => (
              <div
                key={fn}
                onClick={() => {
                  setSelectedFunction(fn)
                  initializeParams(fn)
                  setResponse(null)
                  setError('')
                  setShowDescription(true)
                }}
                className="p-3 rounded cursor-pointer transition-all"
                style={{
                  backgroundColor: selectedFunction === fn ? ibmColors.primary : ibmColors.background,
                  border: `1px solid ${selectedFunction === fn ? ibmColors.primary : ibmColors.border}`,
                  color: ibmColors.text
                }}
                onMouseEnter={(e) => {
                  if (selectedFunction !== fn) {
                    e.currentTarget.style.backgroundColor = '#353535'
                  }
                }}
                onMouseLeave={(e) => {
                  if (selectedFunction !== fn) {
                    e.currentTarget.style.backgroundColor = ibmColors.background
                  }
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{fn}</span>
                  <svg 
                    className="w-4 h-4 transition-transform"
                    style={{ transform: selectedFunction === fn ? 'rotate(90deg)' : 'none' }}
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
                
                {/* Function Details */}
                {selectedFunction === fn && showDescription && (
                  <div className="mt-3 pt-3" style={{ borderTop: `1px solid ${ibmColors.border}` }}>
                    <p className="text-sm" style={{ color: ibmColors.textSecondary }}>
                      {getFunctionDescription(fn)}
                    </p>
                    <div className="mt-2 space-y-1">
                      <p className="text-xs" style={{ color: ibmColors.textSecondary }}>
                        Parameters: {Object.keys(filteredSchema[fn].input).length}
                      </p>
                      <p className="text-xs" style={{ color: ibmColors.textSecondary }}>
                        Returns: {filteredSchema[fn].output.type}
                      </p>
                      {filteredSchema[fn].hash && (
                        <p className="text-xs" style={{ color: ibmColors.textSecondary }}>
                          Hash: {filteredSchema[fn].hash.substring(0, 8)}...
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="text-center py-8" style={{ color: ibmColors.textSecondary }}>
              No functions available
            </div>
          )}
        </div>
        
        {/* Module Info */}
        {mod.name && (
          <div className="mt-6 pt-6" style={{ borderTop: `1px solid ${ibmColors.border}` }}>
            <h3 className="text-sm font-medium mb-2" style={{ color: ibmColors.textSecondary }}>Module Information</h3>
            <p style={{ color: ibmColors.text }}>{mod.name}</p>
            {mod.description && (
              <p className="text-xs mt-1" style={{ color: ibmColors.textSecondary }}>{mod.description}</p>
            )}
          </div>
        )}
      </div>

      {/* Right Panel - Parameters and Execution */}
      <div className="flex-1 space-y-4 p-6 rounded" style={{ backgroundColor: ibmColors.surface, borderColor: ibmColors.border, borderWidth: '1px', borderStyle: 'solid' }}>
        <h2 className="text-xl font-semibold" style={{ color: ibmColors.text, borderBottom: `1px solid ${ibmColors.border}`, paddingBottom: '8px' }}>
          Function Parameters
        </h2>
        
        {selectedFunction && filteredSchema[selectedFunction] ? (
          <div className="space-y-4">
            {Object.entries(filteredSchema[selectedFunction].input).map(([param, details]) => (
              <div key={param} className="space-y-2">
                <label className="block text-sm font-medium" style={{ color: ibmColors.text }}>
                  {param}
                  <span className="ml-2" style={{ color: ibmColors.textSecondary }}>({details.type})</span>
                </label>
                {details.value !== '_empty' && details.value !== undefined && (
                  <p className="text-xs" style={{ color: ibmColors.textSecondary }}>Default: {String(details.value)}</p>
                )}
                <input
                  type="text"
                  value={params[param] !== undefined ? params[param] : ''}
                  onChange={(e) => handleParamChange(param, e.target.value)}
                  placeholder={details.value !== '_empty' && details.value !== undefined ? `Default: ${details.value}` : `Enter ${param}`}
                  className="w-full px-4 py-2 rounded"
                  style={{
                    backgroundColor: ibmColors.background,
                    color: ibmColors.text,
                    border: `1px solid ${ibmColors.border}`,
                    outline: 'none'
                  }}
                  onFocus={(e) => {
                    e.currentTarget.style.borderColor = ibmColors.primary
                  }}
                  onBlur={(e) => {
                    e.currentTarget.style.borderColor = ibmColors.border
                  }}
                />
              </div>
            ))}

            <button
              onClick={executeFunction}
              disabled={loading}
              className="w-full mt-6 px-4 py-3 rounded font-medium transition-all"
              style={{
                backgroundColor: loading ? ibmColors.secondary : ibmColors.primary,
                color: ibmColors.text,
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.7 : 1
              }}
              onMouseEnter={(e) => {
                if (!loading) {
                  e.currentTarget.style.backgroundColor = '#0353e9'
                }
              }}
              onMouseLeave={(e) => {
                if (!loading) {
                  e.currentTarget.style.backgroundColor = ibmColors.primary
                }
              }}
            >
              {loading ? 'Executing...' : 'Execute Function'}
            </button>
          </div>
        ) : (
          <div className="text-center py-8" style={{ color: ibmColors.textSecondary }}>
            Select a function from the left panel to configure parameters
          </div>
        )}

        {/* Response Display */}
        {(response || error) && (
          <div className="mt-6 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="font-medium" style={{ color: ibmColors.text }}>Response</h3>
              <CopyButton code={JSON.stringify(response || error, null, 2)} />
            </div>
            <pre className="p-4 rounded overflow-x-auto max-h-64" style={{
              backgroundColor: ibmColors.background,
              border: `1px solid ${error ? ibmColors.danger : ibmColors.success}`
            }}>
              <code className="text-sm" style={{ color: error ? ibmColors.danger : ibmColors.text }}>
                {JSON.stringify(response || error, null, 2)}
              </code>
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}

export default ModuleSchema