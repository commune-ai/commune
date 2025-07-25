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
  
  console.log('ModuleSchema props:', mod)
  let schema: Record<string, SchemaType> = mod.schema || {}
  
  // Filter out 'self' method
  const filteredSchema = Object.entries(schema).reduce((acc, [key, value]) => {
    if (key !== 'self') {
      acc[key] = value
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

  return (
    <div className="flex gap-6 h-full">
      {/* Left Panel - Parameters Input */}
      <div className="flex-1 space-y-6 p-6 bg-black/50 border border-green-500/20 rounded-lg">
        <h2 className="text-xl font-bold text-green-400 border-b border-green-500/20 pb-2">
          Parameters
        </h2>
        
        {selectedFunction && filteredSchema[selectedFunction] ? (
          <div className="space-y-4">
            {Object.entries(filteredSchema[selectedFunction].input).map(([param, details]) => (
              <div key={param} className="space-y-2">
                <label className="block text-sm font-medium text-gray-300">
                  {param}
                  <span className="text-green-500/70 ml-2">({details.type})</span>
                </label>
                {details.value !== '_empty' && details.value !== undefined && (
                  <p className="text-xs text-gray-500">Default: {String(details.value)}</p>
                )}
                <input
                  type="text"
                  value={params[param] !== undefined ? params[param] : ''}
                  onChange={(e) => handleParamChange(param, e.target.value)}
                  placeholder={details.value !== '_empty' && details.value !== undefined ? `Default: ${details.value}` : `Enter ${param}`}
                  className="w-full px-4 py-2 bg-black/70 text-green-400 
                           border border-green-500/30 rounded-lg
                           focus:outline-none focus:border-green-400 focus:ring-1 focus:ring-green-400/50
                           placeholder-gray-600"
                />
              </div>
            ))}

            <button
              onClick={executeFunction}
              disabled={loading}
              className="w-full mt-6 px-4 py-3 bg-green-900/20 text-green-400 
                       border border-green-500/30 rounded-lg font-medium
                       hover:bg-green-900/30 hover:border-green-400 transition-all
                       disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Executing..' : 'Execute Function'}
            </button>
          </div>
        ) : (
          <div className="text-gray-500 text-center py-8">
            Select a function from the right panel to configure parameters
          </div>
        )}

        {/* Response Display */}
        {(response || error) && (
          <div className="mt-6 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-green-400 font-medium">Response</h3>
              <CopyButton code={JSON.stringify(response || error, null, 2)} />
            </div>
            <pre className="p-4 bg-black/70 border border-green-500/30 rounded-lg overflow-x-auto max-h-64">
              <code className={`text-sm ${error ? 'text-red-400' : 'text-green-300'}`}>
                {JSON.stringify(response || error, null, 2)}
              </code>
            </pre>
          </div>
        )}
      </div>

      {/* Right Panel - Function Selection */}
      <div className="w-96 space-y-6 p-6 bg-black/50 border border-green-500/20 rounded-lg">
        <h2 className="text-xl font-bold text-green-400 border-b border-green-500/20 pb-2">
          Functions
        </h2>
        
        <div className="space-y-3">
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
                className={`p-4 bg-black/70 border rounded-lg cursor-pointer transition-all
                         hover:bg-green-900/20 hover:border-green-400
                         ${selectedFunction === fn 
                           ? 'border-green-400 bg-green-900/20' 
                           : 'border-green-500/30'}`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-green-400 font-medium">{fn}</span>
                  <svg 
                    className={`w-4 h-4 text-green-500 transition-transform
                              ${selectedFunction === fn ? 'rotate-90' : ''}`}
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
                
                {/* Function Description */}
                {selectedFunction === fn && showDescription && (
                  <div className="mt-3 pt-3 border-t border-green-500/20">
                    <p className="text-sm text-gray-400">
                      {getFunctionDescription(fn)}
                    </p>
                    <div className="mt-2 space-y-1">
                      <p className="text-xs text-gray-500">
                        Parameters: {Object.keys(filteredSchema[fn].input).length}
                      </p>
                      <p className="text-xs text-gray-500">
                        Output type: {filteredSchema[fn].output.type}
                      </p>
                      <p className="text-xs text-gray-500">
                        Code: {filteredSchema[fn].code ? 'Available' : 'NA'}
                      </p>
                      <p className="text-xs text-gray-500">
                        Hash: {filteredSchema[fn].hash ? filteredSchema[fn].hash.substring(0, 8) + '...' : 'NA'}
                      </p>
                    </div>
                    
                    {/* Show code snippet if available */}
                    {filteredSchema[fn].code && (
                      <div className="mt-3">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-gray-400">Code Preview</span>
                          <CopyButton code={filteredSchema[fn].code} />
                        </div>
                        <pre className="p-2 bg-black/70 border border-green-500/20 rounded text-xs overflow-x-auto max-h-32">
                          <code className="text-green-300">
                            {filteredSchema[fn].code.substring(0, 200)}...
                          </code>
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="text-gray-500 text-center py-8">
              No functions available in schema
            </div>
          )}
        </div>
        
        {/* Module Info */}
        {mod.name && (
          <div className="mt-6 pt-6 border-t border-green-500/20">
            <h3 className="text-sm font-medium text-gray-400 mb-2">Module Info</h3>
            <p className="text-green-400">{mod.name}</p>
            {mod.description && (
              <p className="text-xs text-gray-500 mt-1">{mod.description}</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default ModuleSchema