'use client'

import { useEffect, useState, useCallback, useMemo } from 'react'
import { Client } from '@/app/client/client'
import { Loading } from '@/app/components/Loading'
import { ModuleType } from '@/app/types/module'
import {
  CodeBracketIcon,
  ServerIcon,
  ArrowLeftIcon,
  ArrowPathIcon,
  DocumentTextIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'
import { CopyButton } from '@/app/components/CopyButton'
import { ModuleCode } from './ModuleCode'
import ModuleSchema from './ModuleSchema'
import Link from 'next/link'

type TabType = 'code' | 'api'

interface ModuleClientProps {
  module_name: string
  code: boolean
  api: boolean
}

const shorten = (str: string): string => {
  if (!str || str.length <= 12) return str
  return `${str.slice(0, 8)}...${str.slice(-4)}`
}

const time2str = (time: number): string => {
  const d = new Date(time * 1000)
  return d.toLocaleString()
}

// Text to color function - generates unique color based on module name
const text2color = (text: string): string => {
  if (!text) return '#00ff00' // Default green
  
  // Create a hash from the text
  let hash = 0
  for (let i = 0; i < text.length; i++) {
    hash = text.charCodeAt(i) + ((hash << 5) - hash)
  }
  
  // Convert hash to HSL color (keeping saturation and lightness consistent for readability)
  const hue = Math.abs(hash) % 360
  const saturation = 70 + (Math.abs(hash >> 8) % 30) // 70-100% saturation
  const lightness = 45 + (Math.abs(hash >> 16) % 15) // 45-60% lightness
  
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`
}

export default function ModuleClient({ module_name, code, api }: ModuleClientProps) {
  // Create client instance once using useMemo to prevent recreation
  const client = useMemo(() => new Client(), [])
  
  const [module, setModule] = useState<ModuleType | undefined>()
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [syncing, setSyncing] = useState<boolean>(false)
  const [codeMap, setCodeMap] = useState<Record<string, string>>({})
  const initialTab: TabType = code ? 'code' : api ? 'api' : 'code'
  const [activeTab, setActiveTab] = useState<TabType>(initialTab)
  
  // Track if we've already fetched to prevent duplicate calls
  const [hasFetched, setHasFetched] = useState(false)

  const fetchModule = useCallback(async (update = false) => {
    try {
      if (update) {
        setSyncing(true)
      } else {
        setLoading(true)
      }
      
      const params = { module: module_name, update: update, code: true }
      const foundModule = await client.call('module', params)
      
      if (foundModule) {
        setModule(foundModule)
        if (foundModule.code && typeof foundModule.code === 'object') {
          setCodeMap(foundModule.code as Record<string, string>)
        }
        setError('')
      } else {
        setError(`Module ${module_name} not found`)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch module')
    } finally {
      setLoading(false)
      setSyncing(false)
    }
  }, [module_name, client])

  // Initial fetch - only run once
  useEffect(() => {
    if (!hasFetched) {
      setHasFetched(true)
      fetchModule(false)
    }
  }, [hasFetched, fetchModule])

  const handleSync = useCallback(() => {
    fetchModule(true)
  }, [fetchModule])

  if (loading) return <Loading />
  
  if (error || !module) {
    return (
      <div className='flex min-h-screen items-center justify-center bg-black'>
        <div className='max-w-md text-center'>
          <ExclamationTriangleIcon className='mx-auto h-12 w-12 text-red-500 mb-4' />
          <h2 className='text-xl font-semibold text-red-500 mb-2'>Error Loading Module</h2>
          <p className='text-gray-400 mb-6'>{error}</p>
          <Link
            href='/'
            className='inline-flex items-center gap-2 rounded-lg border border-green-500/30 bg-black/90 px-4 py-2 text-green-400 hover:bg-green-900/20'
          >
            <ArrowLeftIcon className='h-4 w-4' />
            Back to Modules
          </Link>
        </div>
      </div>
    )
  }

  // Generate the module color based on its name
  const moduleColor = text2color(module.name)

  const tabs = [
    { id: 'code', label: 'CODE', icon: CodeBracketIcon },
    { id: 'api', label: 'SCHEMA', icon: ServerIcon },
  ]

  return (
    <div className='min-h-screen bg-gradient-to-b from-black to-gray-950 p-6 font-mono'>
      <div className='mx-auto max-w-7xl space-y-6'>
        {/* Header with key info */}
        <div className='space-y-4'>
          {/* Module name with sync */}
          <div className='flex items-center justify-between'>
            <button
              onClick={handleSync}
              disabled={syncing}
              className='group flex items-center space-x-3 text-left transition-all duration-300'
              style={{ color: moduleColor }}
            >
              <h1 className='text-3xl font-bold group-hover:underline'>
                {module.name}
              </h1>
              <ArrowPathIcon className={`h-6 w-6 opacity-0 group-hover:opacity-100 ${syncing ? 'animate-spin' : ''}`} />
            </button>
            <Link
              href='/'
              className='flex items-center gap-2 rounded-lg border bg-black/90 px-4 py-2 transition-all hover:bg-black/20'
              style={{ 
                borderColor: `${moduleColor}4D`,
                color: moduleColor
              }}
            >
              <ArrowLeftIcon className='h-4 w-4' />
              <span>Back</span>
            </Link>
          </div>

          {/* Key info row */}
          <div className='flex flex-wrap gap-4 text-sm'>
            {module.key && (
              <div className='flex items-center gap-2'>
                <span className='text-gray-400'>key:</span>
                <span className='font-mono' style={{ color: `${moduleColor}CC` }}>
                  {shorten(module.key)}
                </span>
                <CopyButton code={module.key} />
              </div>
            )}
            {module.cid && (
              <div className='flex items-center gap-2'>
                <span className='text-gray-400'>cid:</span>
                <span className='font-mono' style={{ color: `${moduleColor}CC` }}>
                  {shorten(module.cid)}
                </span>
                <CopyButton code={module.cid} />
              </div>
            )}
            <div className='flex items-center gap-2'>
              <span className='text-gray-400'>created:</span>
              <span className='font-mono' style={{ color: `${moduleColor}CC` }}>
                {time2str(module.time)}
              </span>
            </div>
          </div>

          {/* Description and tags */}
          {module.desc && (
            <p className='max-w-3xl text-gray-400'>
              {module.desc}
            </p>
          )}
          {module.tags?.length > 0 && (
            <div className='flex flex-wrap gap-2'>
              {module.tags.map((tag, i) => (
                <span
                  key={i}
                  className='rounded-full border px-3 py-1 text-sm'
                  style={{ 
                    borderColor: `${moduleColor}33`,
                    backgroundColor: `${moduleColor}0D`,
                    color: `${moduleColor}CC`
                  }}
                >
                  #{tag}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Main content card */}
        <div className='overflow-hidden rounded-2xl border bg-black/90 shadow-xl backdrop-blur-sm'
             style={{ 
               borderColor: `${moduleColor}4D`,
               boxShadow: `0 4px 20px ${moduleColor}1A`
             }}>
          {/* Tabs */}
          <div className='flex border-b' style={{ borderColor: `${moduleColor}33` }}>
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as TabType)}
                className={`flex items-center space-x-2 px-8 py-4 transition-all ${
                  activeTab === id
                    ? 'border-b-2'
                    : 'hover:bg-black/10'
                }`}
                style={{
                  borderColor: activeTab === id ? moduleColor : 'transparent',
                  backgroundColor: activeTab === id ? `${moduleColor}0D` : 'transparent',
                  color: activeTab === id ? moduleColor : `${moduleColor}80`
                }}
              >
                <Icon className='h-5 w-5' />
                <span>{label}</span>
              </button>
            ))}
          </div>

          {/* Content */}
          <div className='p-8'>
            {activeTab === 'code' && (
              <ModuleCode
                files={codeMap}
                title=''
                showSearch={true}
                showFileTree={Object.keys(codeMap).length > 3}
                compactMode={false}
              />
            )}
            {activeTab === 'api' && <ModuleSchema mod={module} />}
          </div>
        </div>

        {/* Footer Actions */}
        <div className='flex flex-wrap items-center justify-center gap-4'>
          {module.url && (
            <a
              href={module.url}
              target='_blank'
              rel='noopener noreferrer'
              className='flex items-center gap-2 rounded-xl border bg-black/90 px-6 py-3 transition-all hover:bg-black/20'
              style={{ 
                borderColor: `${moduleColor}4D`,
                color: moduleColor
              }}
            >
              <span>Visit App</span>
            </a>
          )}
          <button className='flex items-center gap-2 rounded-xl border bg-black/90 px-6 py-3 transition-all hover:bg-black/20'
                  style={{ 
                    borderColor: `${moduleColor}4D`,
                    color: moduleColor
                  }}>
            <DocumentTextIcon className='h-5 w-5' />
            <span>Documentation</span>
          </button>
          <button className='flex items-center gap-2 rounded-xl border bg-black/90 px-6 py-3 transition-all hover:bg-black/20'
                  style={{ 
                    borderColor: `${moduleColor}4D`,
                    color: moduleColor
                  }}>
            <ExclamationTriangleIcon className='h-5 w-5' />
            <span>Report Issue</span>
          </button>
        </div>
      </div>
    </div>
  )
}