'use client'

import { useState, useCallback } from 'react'
import { ModuleType, DefaultModule } from '@/app/types/module'
import { Client } from '@/app/client/client'
import { XMarkIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline'


interface CreateModuleProps {
  onClose: () => void
  onSuccess: () => void
}

export const CreateModule = ({ onClose, onSuccess }: CreateModuleProps) => {
  const [newModule, setNewModule] = useState<ModuleType>({
    ...DefaultModule,
    name: '',
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [tagInput, setTagInput] = useState('') 
  const [showOptional, setShowOptional] = useState(false)
  const client = new Client()

  const handleFormChange = useCallback((field: string, value: any) => {
    setNewModule((prev) => ({ ...prev, [field]: value }))
    if (error) setError('')
  }, [error])

  // Handle tag input
  const handleTagInput = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault()
      const newTag = tagInput.trim().toLowerCase()
      if (newTag && !newModule.tags.includes(newTag)) {
        handleFormChange('tags', [...newModule.tags, newTag])
        setTagInput('')
      }
    }
  }

  // Function to remove a tag
  const removeTag = (tagToRemove: string) => {
    handleFormChange('tags', newModule.tags.filter(tag => tag !== tagToRemove))
  }

  const validateModule = (): string | null => {
    if (!newModule.name || newModule.name.trim() === '') {
      return 'Module name is required'
    }
    
    // Validate name format
    if (!/^[a-zA-Z0-9-_]+$/.test(newModule.name)) {
      return 'Module name can only contain letters, numbers, hyphens, and underscores'
    }
    
    // Validate URL if provided
    if (newModule.url && !newModule.url.match(/^https?:\/\/.+/)) {
      return 'URL must start with http:// or https://'
    }
    
    return null
  }

  const handleCreate = async () => {
    const validationError = validateModule()
    if (validationError) {
      setError(validationError)
      return
    }

    setLoading(true)
    setError('')
    
    try {
      // Auto-generate URL if not provided
      const moduleToCreate = { ...newModule }
      if (!moduleToCreate.url) {
        moduleToCreate.url = `https://${moduleToCreate.name.toLowerCase().replace(/\s+/g, '-')}.commune.ai`
      }

      await client.call('add_module', moduleToCreate)
      onSuccess()
    } catch (err: any) {
      setError(err.message || 'Failed to create module')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose()
    } else if (e.key === 'Enter' && e.ctrlKey && !loading) {
      handleCreate()
    }
  }

  return (
    <div 
      className="w-full max-w-2xl p-6 bg-black/90 rounded-lg border border-green-500/30 shadow-2xl"
      onKeyDown={handleKeyDown}
      role="dialog"
      aria-labelledby="create-module-title"
      aria-modal="true"
    >
      <div className="flex items-center justify-between mb-6">
        <h2 id="create-module-title" className="text-xl font-semibold text-green-400 flex items-center gap-2">
          <span className="text-yellow-500">$</span> new_module
        </h2>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-green-400 transition-colors"
          aria-label="Close dialog"
        >
          <XMarkIcon className="h-6 w-6" />
        </button>
      </div>

      {/* Form fields */}
      <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); handleCreate(); }}>
        {/* Required Field - Module Name */}
        <div className="space-y-2">
          <label htmlFor="module-name" className="text-green-400 text-sm font-medium">
            Name <span className="text-red-400">*</span>
          </label>
          <input
            id="module-name"
            placeholder="Enter your module name (e.g., my-awesome-module)"
            value={newModule.name || ''}
            onChange={(e) => handleFormChange('name', e.target.value)}
            className="w-full px-4 py-2 bg-black/90 text-green-400 focus:outline-none focus:border-green-400 focus:ring-2 focus:ring-green-400/50 border border-green-500/30 rounded"
            autoFocus
            required
          />
          <p className="text-xs text-gray-500">This is the only required field. We'll figure out the rest!</p>
        </div>

        <div className="space-y-2">
          <label htmlFor="module-name" className="text-green-400 text-sm font-medium">
            Key <span className="text-red-400">*</span>
          </label>
          <input
            id="module-name"
            placeholder="Enter your module name (e.g., my-awesome-module)"
            value={newModule.key || ''}
            onChange={(e) => handleFormChange('key', e.target.value)}
            className="w-full px-4 py-2 bg-black/90 text-green-400 focus:outline-none focus:border-green-400 focus:ring-2 focus:ring-green-400/50 border border-green-500/30 rounded"
            autoFocus
            required
          />
          <p className="text-xs text-gray-500">This is the only required field. We'll figure out the rest!</p>
        </div>


        {/* Toggle Optional Fields */}
        <button
          type="button"
          onClick={() => setShowOptional(!showOptional)}
          className="text-green-400 text-sm hover:text-green-300 flex items-center gap-2 transition-colors"
          aria-expanded={showOptional}
        >
          <span className="transition-transform" style={{ transform: showOptional ? 'rotate(90deg)' : 'rotate(0deg)' }}>▶</span>
          <span>Show optional fields</span>
        </button>

        {/* Optional Fields */}
        {showOptional && (
          <div className="space-y-4 pl-4 border-l-2 border-green-500/20 animate-fade-in">
            {/* URL Field */}
            <div className="space-y-2">
              <label htmlFor="module-url" className="text-green-400 text-sm">
                URL <span className="text-gray-500">(optional)</span>
              </label>
              <input
                id="module-url"
                placeholder="https://your-module.com (we'll auto-generate if empty)"
                value={newModule.url || ''}
                onChange={(e) => handleFormChange('url', e.target.value)}
                className="w-full px-4 py-2 bg-black/90 text-green-400 border focus:outline-none focus:border-green-400 focus:ring-2 focus:ring-green-400/50 border-green-500/30 rounded"
              />
              <p className="text-xs text-gray-500">URL of your server or website</p>
            </div>
            
            {/* Code Repository */}
            <div className="space-y-2">
              <label htmlFor="module-code" className="text-green-400 text-sm">
                Code <span className="text-gray-500">(optional)</span>
              </label>
              <input
                id="module-code"
                placeholder="github.com/username/repo (or IPFS/S3/Arweave link)"
                value={newModule.code || ''}
                onChange={(e) => handleFormChange("code", e.target.value)}
                className="w-full px-4 py-2 bg-black/90 text-green-400 border focus:outline-none focus:border-green-400 focus:ring-2 focus:ring-green-400/50 border-green-500/30 rounded"
              />
              <p className="text-xs text-gray-500">Link to your module's source code</p>
            </div>

            {/* Description */}
            <div className="space-y-2">
              <label htmlFor="module-desc" className="text-green-400 text-sm">
                Description <span className="text-gray-500">(optional)</span>
              </label>
              <textarea
                id="module-desc"
                placeholder="Describe what your module does..."
                value={newModule.desc || ''}
                onChange={(e) => handleFormChange('desc', e.target.value)}
                className="w-full px-4 py-2 bg-black/90 text-green-400 border focus:outline-none focus:border-green-400 focus:ring-2 focus:ring-green-400/50 border-green-500/30 rounded h-20 resize-none"
                rows={3}
              />
              <p className="text-xs text-gray-500">Brief description of your module's functionality</p>
            </div>

            {/* Tags */}
            <div className="space-y-2">
              <label htmlFor="module-tags" className="text-green-400 text-sm">
                Tags <span className="text-gray-500">(optional)</span>
              </label>
              <input
                id="module-tags"
                placeholder="Add tags (press Enter or comma to add)"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={handleTagInput}
                className="w-full px-4 py-2 bg-black/90 text-green-400 border focus:outline-none focus:border-green-400 focus:ring-2 focus:ring-green-400/50 border-green-500/30 rounded"
              />
              <p className="text-xs text-gray-500">Tags help others discover your module</p>
              
              {/* Tags display */}
              {newModule.tags.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2" role="list" aria-label="Selected tags">
                  {newModule.tags.map((tag, index) => (
                    <span 
                      key={index}
                      className="px-2 py-1 bg-green-900/20 text-green-400 rounded-full text-sm flex items-center"
                      role="listitem"
                    >
                      {tag}
                      <button
                        type="button"
                        onClick={() => removeTag(tag)}
                        className="ml-2 text-green-400 hover:text-green-300 focus:outline-none"
                        aria-label={`Remove tag ${tag}`}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </form>

      {error && (
        <div className="mt-4 p-3 bg-red-500/20 border border-red-500 text-red-400 rounded flex items-center gap-2" role="alert">
          <ExclamationTriangleIcon className="h-5 w-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="flex justify-end gap-4 mt-6">
        <button
          onClick={onClose}
          disabled={loading}
          className="px-4 py-2 bg-black/90 text-green-400 border border-green-500/30 rounded hover:bg-green-900/20 transition-colors disabled:opacity-50"
        >
          [ESC] Cancel
        </button>
        <button
          onClick={handleCreate}
          disabled={loading}
          className="px-4 py-2 bg-black/90 text-green-400 border border-green-500/30 rounded hover:bg-green-900/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading ? (
            <>
              <span className="animate-spin h-4 w-4 border-2 border-green-400 border-t-transparent rounded-full" />
              Creating...
            </>
          ) : (
            '[Ctrl+Enter] Create'
          )}
        </button>
      </div>
    </div>
  )
}

// Add this to your global CSS for the fade-in animation
const styles = `
@keyframes fade-in {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in {
  animation: fade-in 0.3s ease-out;
}

.animate-slide-down {
  animation: slide-down 0.3s ease-out;
}

@keyframes slide-down {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
`