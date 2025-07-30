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
  ChartBarIcon,
  CubeIcon,
  SparklesIcon,
  CommandLineIcon,
  GlobeAltIcon,
  ClockIcon,
  HashtagIcon,
  TagIcon,
  LinkIcon,
  DocumentDuplicateIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline'
import { CopyButton } from '@/app/components/CopyButton'
import { ModuleCode } from './ModuleCode'
import ModuleSchema from './ModuleSchema'
import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'

type TabType = 'code' | 'api' | 'history' | 'links'

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
  const now = new Date()
  const diff = now.getTime() - d.getTime()
  
  if (diff < 60000) return 'just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`
  
  return d.toLocaleDateString('en-US', { 
    month: 'short', 
    day: 'numeric',
    year: d.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
  })
}

// Enhanced color generation with better distribution
const text2color = (text: string): string => {
  if (!text) return '#00ff00'
  
  let hash = 0
  for (let i = 0; i < text.length; i++) {
    hash = text.charCodeAt(i) + ((hash << 5) - hash)
  }
  
  // Use golden ratio for better color distribution
  const golden_ratio = 0.618033988749895
  const hue = (hash * golden_ratio * 360) % 360
  const saturation = 65 + (Math.abs(hash >> 8) % 35)
  const lightness = 50 + (Math.abs(hash >> 16) % 20)
  
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`
}

// Generate dynamic background pattern
const generatePattern = (color: string, seed: string) => {
  const svg = `
    <svg width="400" height="400" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 40" fill="none" stroke="${color}" stroke-width="0.5" opacity="0.1"/>
        </pattern>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
      <rect width="400" height="400" fill="url(#grid)"/>
      ${Array.from({ length: 5 }, (_, i) => {
        const x = (parseInt(seed.slice(i * 2, i * 2 + 2), 16) / 255) * 350 + 25
        const y = (parseInt(seed.slice(i * 2 + 10, i * 2 + 12), 16) / 255) * 350 + 25
        const r = 3 + (i % 3) * 2
        return `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" opacity="0.6" filter="url(#glow)"/>`
      }).join('')}
    </svg>
  `
  return `data:image/svg+xml;base64,${btoa(svg)}`
}

interface HistoryItemProps {
  version: string
  date: string
  changes: string[]
  color: string
}

const HistoryItem = ({ version, date, changes, color }: HistoryItemProps) => (
  <motion.div
    initial={{ opacity: 0, x: -20 }}
    animate={{ opacity: 1, x: 0 }}
    className="relative pl-8 pb-8 group"
  >
    {/* Timeline line */}
    <div className="absolute left-3 top-8 bottom-0 w-0.5 bg-gray-800 group-last:hidden" />
    
    {/* Timeline dot */}
    <div 
      className="absolute left-1.5 top-2 h-4 w-4 rounded-full border-2 bg-black"
      style={{ borderColor: color }}
    >
      <div 
        className="absolute inset-0.5 rounded-full animate-pulse"
        style={{ backgroundColor: color }}
      />
    </div>
    
    {/* Content */}
    <div 
      className="rounded-lg border bg-black/60 p-4 backdrop-blur-sm hover:bg-black/80 transition-all"
      style={{ borderColor: `${color}33` }}
    >
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-bold text-white" style={{ color }}>{version}</h4>
        <span className="text-sm text-gray-400">{date}</span>
      </div>
      <ul className="space-y-1">
        {changes.map((change, i) => (
          <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
            <span style={{ color }} className="mt-1">•</span>
            <span>{change}</span>
          </li>
        ))}
      </ul>
    </div>
  </motion.div>
)

interface LinkedModuleProps {
  name: string
  description: string
  type: 'dependency' | 'dependent' | 'related'
  color: string
}

const LinkedModule = ({ name, description, type, color }: LinkedModuleProps) => {
  const typeColors = {
    dependency: '#ff6b6b',
    dependent: '#4ecdc4',
    related: '#45b7d1'
  }
  
  const typeLabels = {
    dependency: 'Depends on',
    dependent: 'Used by',
    related: 'Related to'
  }
  
  return (
    <Link href={`/module/${name}`}>
      <motion.div
        whileHover={{ scale: 1.02, y: -2 }}
        className="relative overflow-hidden rounded-lg border bg-black/60 p-4 backdrop-blur-sm cursor-pointer group"
        style={{ borderColor: `${typeColors[type]}33` }}
      >
        <div className="absolute inset-0 opacity-5" 
             style={{ backgroundImage: `url(${generatePattern(typeColors[type], name)})` }} />
        
        <div className="relative z-10">
          <div className="flex items-start justify-between mb-2">
            <h4 className="font-bold text-white group-hover:text-opacity-90 transition-colors"
                style={{ color: typeColors[type] }}>
              {name}
            </h4>
            <span className="text-xs px-2 py-1 rounded-full"
                  style={{ 
                    backgroundColor: `${typeColors[type]}20`,
                    color: typeColors[type]
                  }}>
              {typeLabels[type]}
            </span>
          </div>
          <p className="text-sm text-gray-400 line-clamp-2">{description}</p>
        </div>
      </motion.div>
    </Link>
  )
}

export default function ModuleClient({ module_name, code, api }: ModuleClientProps) {
  const client = useMemo(() => new Client(), [])
  
  const [module, setModule] = useState<ModuleType | undefined>()
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [syncing, setSyncing] = useState<boolean>(false)
  const [codeMap, setCodeMap] = useState<Record<string, string>>({})
  const initialTab: TabType = code ? 'code' : api ? 'api' : 'history'
  const [activeTab, setActiveTab] = useState<TabType>(initialTab)
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
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className='max-w-md text-center'
        >
          <ExclamationTriangleIcon className='mx-auto h-16 w-16 text-red-500 mb-6' />
          <h2 className='text-2xl font-bold text-red-500 mb-3'>Error Loading Module</h2>
          <p className='text-gray-400 mb-8'>{error}</p>
          <Link
            href='/'
            className='inline-flex items-center gap-2 rounded-lg border border-green-500/30 bg-black/90 px-6 py-3 text-green-400 hover:bg-green-900/20 transition-all'
          >
            <ArrowLeftIcon className='h-5 w-5' />
            Back to Modules
          </Link>
        </motion.div>
      </div>
    )
  }

  const moduleColor = text2color(module.name)
  const patternBg = generatePattern(moduleColor, module.key || module.name)

  const tabs = [
    { id: 'history', label: 'HISTORY', icon: ClockIcon },
    { id: 'links', label: 'LINKS', icon: LinkIcon },
    { id: 'code', label: 'CODE', icon: CodeBracketIcon },
    { id: 'api', label: 'API', icon: ServerIcon },
  ]

  // Mock data - replace with real data from your API
  const historyData = [
    {
      version: 'v2.1.0',
      date: 'Nov 15, 2024',
      changes: [
        'Added WebSocket support for real-time updates',
        'Improved error handling and retry logic',
        'Performance optimizations for large datasets'
      ]
    },
    {
      version: 'v2.0.0',
      date: 'Oct 28, 2024',
      changes: [
        'Complete UI redesign with new color scheme',
        'Migrated to TypeScript for better type safety',
        'Added comprehensive test coverage'
      ]
    },
    {
      version: 'v1.5.2',
      date: 'Oct 10, 2024',
      changes: [
        'Fixed memory leak in data processing',
        'Updated dependencies to latest versions',
        'Added support for custom themes'
      ]
    }
  ]

  const linkedModules = {
    dependencies: [
      { name: 'auth', description: 'Handles user authentication and authorization' },
      { name: 'database', description: 'Manages data persistence and queries' },
    ],
    dependents: [
      { name: 'dashboard', description: 'Main application dashboard interface' },
      { name: 'analytics', description: 'Data analytics and visualization module' },
    ],
    related: [
      { name: 'logger', description: 'Centralized logging and monitoring service' },
      { name: 'cache', description: 'High-performance caching layer' },
    ]
  }

  return (
    <div className='min-h-screen bg-gradient-to-br from-black via-gray-950 to-black'>
      {/* Animated background */}
      <div className='fixed inset-0 opacity-20'>
        <div className='absolute inset-0' 
             style={{ 
               backgroundImage: `url(${patternBg})`,
               backgroundSize: '400px 400px',
               animation: 'slide 20s linear infinite'
             }} />
      </div>
      
      <div className='relative z-10'>
        <div className='mx-auto max-w-7xl'>
          {/* Enhanced Header */}
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className='space-y-4 px-6 pt-4 pb-6'
          >
            {/* Top Navigation */}
            <div className='flex items-center justify-between'>
              <Link
                href='/'
                className='group flex items-center gap-2 text-gray-400 hover:text-white transition-colors'
              >
                <ArrowLeftIcon className='h-4 w-4 group-hover:-translate-x-1 transition-transform' />
                <span>All Modules</span>
              </Link>
              
              <button
                onClick={handleSync}
                disabled={syncing}
                className='group flex items-center gap-2 rounded-full border px-4 py-2 transition-all'
                style={{ 
                  borderColor: `${moduleColor}4D`,
                  color: moduleColor,
                  backgroundColor: syncing ? `${moduleColor}10` : 'transparent'
                }}
              >
                <ArrowPathIcon className={`h-4 w-4 ${syncing ? 'animate-spin' : 'group-hover:rotate-180 transition-transform duration-500'}`} />
                <span className='text-sm font-medium'>{syncing ? 'Syncing...' : 'Sync'}</span>
              </button>
            </div>

            {/* Module Header */}
            <div className='flex items-start gap-4'>
              <motion.div 
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className='p-3 rounded-2xl'
                style={{ 
                  backgroundColor: `${moduleColor}15`,
                  boxShadow: `0 0 30px ${moduleColor}30`
                }}
              >
                <CubeIcon className='h-10 w-10' style={{ color: moduleColor }} />
              </motion.div>
              
              <div className='flex-1'>
                <h1 className='text-4xl font-bold mb-1' 
                    style={{ 
                      color: moduleColor,
                      textShadow: `0 0 20px ${moduleColor}50`
                    }}>
                  {module.name}
                </h1>
                <p className='text-lg text-gray-400 mb-3'>
                  {module.desc || 'A powerful module in the Commune ecosystem'}
                </p>
                
                {/* Tags */}
                {module.tags?.length > 0 && (
                  <div className='flex flex-wrap gap-2'>
                    {module.tags.map((tag, i) => (
                      <motion.span
                        key={i}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.05 }}
                        className='inline-flex items-center gap-1 rounded-full px-3 py-1 text-sm'
                        style={{ 
                          backgroundColor: `${moduleColor}10`,
                          color: moduleColor,
                          border: `1px solid ${moduleColor}30`
                        }}
                      >
                        <TagIcon className='h-3 w-3' />
                        {tag}
                      </motion.span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Key Info Cards */}
            <div className='grid grid-cols-1 md:grid-cols-3 gap-3'>
              {module.key && (
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className='flex items-center gap-3 rounded-xl border bg-black/60 p-3 backdrop-blur-sm'
                  style={{ borderColor: `${moduleColor}33` }}
                >
                  <HashtagIcon className='h-5 w-5' style={{ color: moduleColor }} />
                  <div className='flex-1'>
                    <p className='text-xs text-gray-400'>Module Key</p>
                    <p className='font-mono text-sm' style={{ color: moduleColor }}>
                      {shorten(module.key)}
                    </p>
                  </div>
                  <CopyButton code={module.key} />
                </motion.div>
              )}
              
              {module.cid && (
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 }}
                  className='flex items-center gap-3 rounded-xl border bg-black/60 p-3 backdrop-blur-sm'
                  style={{ borderColor: `${moduleColor}33` }}
                >
                  <CommandLineIcon className='h-5 w-5' style={{ color: moduleColor }} />
                  <div className='flex-1'>
                    <p className='text-xs text-gray-400'>Content ID</p>
                    <p className='font-mono text-sm' style={{ color: moduleColor }}>
                      {shorten(module.cid)}
                    </p>
                  </div>
                  <CopyButton code={module.cid} />
                </motion.div>
              )}
              
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                className='flex items-center gap-3 rounded-xl border bg-black/60 p-3 backdrop-blur-sm'
                style={{ borderColor: `${moduleColor}33` }}
              >
                <ClockIcon className='h-5 w-5' style={{ color: moduleColor }} />
                <div className='flex-1'>
                  <p className='text-xs text-gray-400'>Last Updated</p>
                  <p className='text-sm' style={{ color: moduleColor }}>
                    {time2str(module.time)}
                  </p>
                </div>
              </motion.div>
            </div>
          </motion.div>

          {/* Main Content */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className='mx-6 overflow-hidden rounded-3xl border bg-black/80 shadow-2xl backdrop-blur-md'
            style={{ 
              borderColor: `${moduleColor}33`,
              boxShadow: `0 20px 60px ${moduleColor}15`
            }}
          >
            {/* Enhanced Tabs */}
            <div className='flex border-b backdrop-blur-sm' style={{ borderColor: `${moduleColor}20` }}>
              {tabs.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id as TabType)}
                  className='group relative flex-1 flex items-center justify-center gap-2 px-6 py-5 transition-all'
                >
                  {activeTab === id && (
                    <motion.div
                      layoutId="activeTab"
                      className='absolute inset-0'
                      style={{ backgroundColor: `${moduleColor}10` }}
                    />
                  )}
                  <Icon className='h-5 w-5 transition-transform group-hover:scale-110' 
                       style={{ color: activeTab === id ? moduleColor : `${moduleColor}80` }} />
                  <span className='font-medium' 
                        style={{ color: activeTab === id ? moduleColor : `${moduleColor}80` }}>
                    {label}
                  </span>
                  {activeTab === id && (
                    <motion.div
                      layoutId="activeTabBorder"
                      className='absolute bottom-0 left-0 right-0 h-0.5'
                      style={{ backgroundColor: moduleColor }}
                    />
                  )}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <AnimatePresence mode='wait'>
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                className='p-8'
              >
                {activeTab === 'history' && (
                  <div className='space-y-6'>
                    <div className='flex items-center gap-3 mb-8'>
                      <ArrowTrendingUpIcon className='h-6 w-6' style={{ color: moduleColor }} />
                      <h2 className='text-2xl font-bold text-white'>Version History</h2>
                    </div>
                    <div className='relative'>
                      {historyData.map((item, i) => (
                        <HistoryItem key={i} {...item} color={moduleColor} />
                      ))}
                    </div>
                  </div>
                )}
                
                {activeTab === 'links' && (
                  <div className='space-y-8'>
                    <div className='flex items-center gap-3 mb-6'>
                      <LinkIcon className='h-6 w-6' style={{ color: moduleColor }} />
                      <h2 className='text-2xl font-bold text-white'>Linked Modules</h2>
                    </div>
                    
                    {/* Dependencies */}
                    <div>
                      <h3 className='text-lg font-semibold text-white mb-4 flex items-center gap-2'>
                        <DocumentDuplicateIcon className='h-5 w-5 text-red-400' />
                        Dependencies
                      </h3>
                      <div className='grid grid-cols-1 md:grid-cols-2 gap-4'>
                        {linkedModules.dependencies.map((mod, i) => (
                          <LinkedModule key={i} {...mod} type='dependency' color={moduleColor} />
                        ))}
                      </div>
                    </div>
                    
                    {/* Dependents */}
                    <div>
                      <h3 className='text-lg font-semibold text-white mb-4 flex items-center gap-2'>
                        <DocumentDuplicateIcon className='h-5 w-5 text-teal-400' />
                        Used By
                      </h3>
                      <div className='grid grid-cols-1 md:grid-cols-2 gap-4'>
                        {linkedModules.dependents.map((mod, i) => (
                          <LinkedModule key={i} {...mod} type='dependent' color={moduleColor} />
                        ))}
                      </div>
                    </div>
                    
                    {/* Related */}
                    <div>
                      <h3 className='text-lg font-semibold text-white mb-4 flex items-center gap-2'>
                        <DocumentDuplicateIcon className='h-5 w-5 text-blue-400' />
                        Related Modules
                      </h3>
                      <div className='grid grid-cols-1 md:grid-cols-2 gap-4'>
                        {linkedModules.related.map((mod, i) => (
                          <LinkedModule key={i} {...mod} type='related' color={moduleColor} />
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                
                {activeTab === 'code' && (
                  <ModuleCode
                    files={codeMap}
                    title='Source Code'
                    showSearch={true}
                    showFileTree={Object.keys(codeMap).length > 3}
                    compactMode={false}
                  />
                )}
                
                {activeTab === 'api' && <ModuleSchema mod={module} />}
              </motion.div>
            </AnimatePresence>
          </motion.div>

          {/* Enhanced Footer Actions */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className='flex flex-wrap items-center justify-center gap-4 p-6'
          >
            {module.url && (
              <motion.a
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                href={module.url}
                target='_blank'
                rel='noopener noreferrer'
                className='flex items-center gap-2 rounded-full border bg-black/80 px-8 py-3 backdrop-blur-sm transition-all'
                style={{ 
                  borderColor: `${moduleColor}4D`,
                  color: moduleColor,
                  boxShadow: `0 4px 20px ${moduleColor}20`
                }}
              >
                <GlobeAltIcon className='h-5 w-5' />
                <span className='font-medium'>Visit Application</span>
              </motion.a>
            )}
            
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className='flex items-center gap-2 rounded-full border bg-black/80 px-8 py-3 backdrop-blur-sm transition-all'
              style={{ 
                borderColor: `${moduleColor}4D`,
                color: moduleColor
              }}
            >
              <DocumentTextIcon className='h-5 w-5' />
              <span className='font-medium'>Documentation</span>
            </motion.button>
          </motion.div>
        </div>
      </div>
      
      <style jsx>{`
        @keyframes slide {
          0% { transform: translate(0, 0); }
          100% { transform: translate(-400px, -400px); }
        }
      `}</style>
    </div>
  )
}