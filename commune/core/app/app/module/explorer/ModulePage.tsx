'use client'

import { useEffect, useState, useCallback } from 'react'
import { Client } from '@/app/client/client'
import { Loading } from '@/app/components/Loading'
import { ModuleType } from '@/app/types/module'
import {
  CodeBracketIcon,
  ServerIcon,
  GlobeAltIcon,
  BeakerIcon,
  ArrowLeftIcon,
  ArrowPathIcon,
  DocumentTextIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'
import { CopyButton } from '@/app/components/CopyButton'
import { ModuleCode } from '../page/ModuleCode'
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

// Text to color function - generates unique color based on module name (same as ModuleCard)
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

interface InfoCardProps {
  label: string
  value: string
  showCopy?: boolean
  moduleColor: string
}

const InfoCard = ({ label, value, showCopy = true, moduleColor }: InfoCardProps) => (
  <div className='rounded-xl border bg-black/60 p-4 backdrop-blur-sm transition-all duration-300'
       style={{ borderColor: `${moduleColor}4D` }}>
    <div className='mb-2 flex items-center justify-between'>
      <span className='text-gray-400'>{label}</span>
      {showCopy && <CopyButton code={value} />}
    </div>
    <div className='truncate font-mono text-sm' style={{ color: `${moduleColor}CC` }}>
      {label === 'created' ? value : shorten(value)}
    </div>
  </div>
)

export default function ModuleClient({ module_name, code, api }: ModuleClientProps) {
  const client = new Client()
  const [module, setModule] = useState<ModuleType | undefined>()
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [syncing, setSyncing] = useState<boolean>(false)
  const [codeMap, setCodeMap] = useState<Record<string, string>>({})
  const initialTab: TabType = code ? 'code' : api ? 'api' : 'code'
  const [activeTab, setActiveTab] = useState<TabType>(initialTab)

  const fetchModule = useCallback(async (update = false) => {
    try {
      if (update) setSyncing(true)

      if (!update) {
        return setModule(undefined)
      }
      const params = { module: module_name, update , code: true}
      const foundModule = await client.call('module', params)
      console.log('Fetched module:', foundModule,params)
      
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
  }, [])

  useEffect(() => {
    fetchModule()
  }, [fetchModule])

  const handleSync = () => {
    fetchModule(true)
  }

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
        {/* Header */}
        <div className='overflow-hidden rounded-2xl border bg-black/90 shadow-xl backdrop-blur-sm'
             style={{ 
               borderColor: `${moduleColor}4D`,
               boxShadow: `0 4px 20px ${moduleColor}1A`
             }}>
          <div className='space-y-6 border-b p-8' style={{ borderColor: `${moduleColor}33` }}>
            <div className='flex items-center justify-between'>
              <div className='flex items-center space-x-4'>
                <BeakerIcon className='h-8 w-8' style={{ color: moduleColor }} />
                <div>
                  <h1 className='text-3xl font-bold' style={{ color: moduleColor }}>
                    {module.name}
                  </h1>
                  <p className='mt-1 text-gray-400'>
                    {module.network || 'commune'}
                  </p>
                </div>
              </div>
              <div className='flex space-x-4'>
                <button 
                  className='flex items-center space-x-2 rounded-lg border bg-black/20 p-2 hover:bg-black/40 disabled:opacity-50 md:px-6 md:py-3 transition-all duration-300'
                  style={{ 
                    borderColor: `${moduleColor}4D`,
                    color: moduleColor
                  }}
                  onClick={handleSync}
                  disabled={syncing}
                  aria-label='Sync module data'
                >
                  <ArrowPathIcon className={`h-5 w-5 ${syncing ? 'animate-spin' : ''}`} />
                  <span className='hidden md:inline'>{syncing ? 'syncing...' : 'sync'}</span>
                </button>
              </div>
            </div>

            <p className='max-w-3xl text-gray-400'>
              {module.desc || 'No description available'}
            </p>

            {module.tags?.length > 0 && (
              <div className='flex flex-wrap gap-2' role='list' aria-label='Module tags'>
                {module.tags.map((tag, i) => (
                  <span
                    key={i}
                    className='rounded-full border px-3 py-1 text-sm'
                    style={{ 
                      borderColor: `${moduleColor}33`,
                      backgroundColor: `${moduleColor}0D`,
                      color: `${moduleColor}CC`
                    }}
                    role='listitem'
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            )}

            <div className='mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3'>
              {module.key && <InfoCard label='key' value={module.key} moduleColor={moduleColor} />}
              {module.cid && <InfoCard label='cid' value={module.cid} moduleColor={moduleColor} />}
              <InfoCard label='created' value={time2str(module.time)} showCopy={false} moduleColor={moduleColor} />
            </div>
          </div>

          {/* Tabs */}
          <div className='flex border-b' role='tablist' style={{ borderColor: `${moduleColor}33` }}>
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
                role='tab'
                aria-selected={activeTab === id}
                aria-controls={`tabpanel-${id}`}
              >
                <Icon className='h-5 w-5' />
                <span>{label}</span>
              </button>
            ))}
          </div>

          {/* Content */}
          <div className='p-8' role='tabpanel' id={`tabpanel-${activeTab}`}>
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
        <nav className='flex flex-wrap items-center justify-between gap-2 sm:gap-4' aria-label='Module actions'>
          <Link
            href='/'
            className='flex w-full items-center justify-center space-x-2 rounded-xl border bg-black/90 px-6 py-3 text-center transition-all hover:bg-black/20 sm:w-auto'
            style={{ 
              borderColor: `${moduleColor}4D`,
              color: moduleColor
            }}
          >
            <ArrowLeftIcon className='h-5 w-5' />
            <span>Back to Modules</span>
          </Link>

          <div className='flex w-full flex-wrap justify-center gap-2 sm:w-auto sm:justify-end sm:gap-4'>
            {module.url && (
              <a
                href={module.url}
                target='_blank'
                rel='noopener noreferrer'
                className='w-full rounded-xl border bg-black/90 px-6 py-3 text-center transition-all hover:bg-black/20 sm:w-auto flex items-center justify-center gap-2'
                style={{ 
                  borderColor: `${moduleColor}4D`,
                  color: moduleColor
                }}
              >
                <GlobeAltIcon className='h-5 w-5' />
                <span>Visit App</span>
              </a>
            )}
            <button className='w-full rounded-xl border bg-black/90 px-6 py-3 text-center transition-all hover:bg-black/20 sm:w-auto flex items-center justify-center gap-2'
                    style={{ 
                      borderColor: `${moduleColor}4D`,
                      color: moduleColor
                    }}>
              <DocumentTextIcon className='h-5 w-5' />
              <span>Documentation</span>
            </button>
            <button className='w-full rounded-xl border bg-black/90 px-6 py-3 text-center transition-all hover:bg-black/20 sm:w-auto flex items-center justify-center gap-2'
                    style={{ 
                      borderColor: `${moduleColor}4D`,
                      color: moduleColor
                    }}>
              <ExclamationTriangleIcon className='h-5 w-5' />
              <span>Report Issue</span>
            </button>
          </div>
        </nav>
      </div>
    </div>
  )
}