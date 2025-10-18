'use client'

import { useEffect, useState, useCallback } from 'react'
import { Client } from '@/app/client/client'
import { Loading } from '@/app/components/Loading'
import { ModuleType } from '@/app/types/module'
import { useUserContext } from '@/app/context/UserContext'
import {
  CodeBracketIcon,
  ServerIcon,
  ArrowPathIcon,
  TagIcon,
  ClockIcon,
  KeyIcon,
  ComputerDesktopIcon,
  CubeIcon,
} from '@heroicons/react/24/outline'
import { CopyButton } from '@/app/components/CopyButton'
import { ModuleContent } from './ModuleContent'
import ModuleSchema from './ModuleApi'
import { ModuleApp } from './ModuleApp'
import { motion, AnimatePresence } from 'framer-motion'

type TabType = 'app' | 'api' | 'content'

interface ModuleProps {
  module_name: string
  content?: boolean
  api?: boolean
}

const shorten = (str: string): string => {
  if (!str || str.length <= 12) return str
  return `${str.slice(0, 8)}...${str.slice(-4)}`
}

const time2str = (time: number): string => {
  const d = new Date(time * 1000)
  const now = new Date()
  const diff = now.getTime() - d.getTime()
  if (diff < 60_000) return 'now'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`
  if (diff < 604_800_000) return `${Math.floor(diff / 86_400_000)}d ago`
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: d.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
  })
}

const text2color = (text: string): string => {
  if (!text) return '#00ff00'
  let hash = 0
  for (let i = 0; i < text.length; i++) hash = text.charCodeAt(i) + ((hash << 5) - hash)
  const golden_ratio = 0.618033988749895
  const hue = (hash * golden_ratio * 360) % 360
  const saturation = 65 + (Math.abs(hash >> 8) % 35)
  const lightness = 50 + (Math.abs(hash >> 16) % 20)
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`
}

export default function Module({ module_name }: ModuleProps) {
  const { keyInstance, authLoading } = useUserContext()

  const [module, setModule] = useState<ModuleType | undefined>()
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [syncing, setSyncing] = useState<boolean>(false)
  const [activeTab, setActiveTab] = useState<TabType>('api')
  const [hasFetched, setHasFetched] = useState(false)

  const fetchModule = useCallback(async (update = false) => {
    try {
      update ? setSyncing(true) : setLoading(true)
      const client = new Client(undefined, keyInstance)
      const params = { module: module_name, update:update, content: true , public: true, schema: true}
      const foundModule = await client.call('module', params)
      if (foundModule) {
        setModule(foundModule as ModuleType)
        setError('')
      } else {
        setError(`Module ${module_name} not found`)
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch module')
    } finally {
      setLoading(false)
      setSyncing(false)
    }
  }, [module_name, keyInstance])

  useEffect(() => {
    if ((!hasFetched && !authLoading) || module === undefined) {
      setHasFetched(true)
      fetchModule(false)
    }
  }, [hasFetched, fetchModule, authLoading, module])

  const handleSync = useCallback(() => {
    fetchModule(true)
  }, [fetchModule])

  if (authLoading || loading || module === undefined) return <Loading />

  const moduleColor = text2color(module.name)

  const tabs: Array<{ id: TabType; icon: any }> = [
    { id: 'api', icon: ServerIcon },
    { id: 'app',  icon: ComputerDesktopIcon },
    { id: 'content', icon: CodeBracketIcon },
  ]
  
  return (
    <div className="min-h-screen bg-black text-white module-page">
      <div className="w-full">
        {/* Compact Professional Header */}
        <div className="w-full px-4 py-3 border-b border-white/10 bg-gradient-to-r from-black via-gray-900/50 to-black">
          <div className="flex flex-wrap items-center gap-3">
            <span
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-2xl font-bold"
              style={{ color: moduleColor, backgroundColor: `${moduleColor}14`, border: `2px solid ${moduleColor}33` }}
            >
              <CubeIcon className="h-7 w-7" />
              {module.name}
            </span>

            {Array.isArray(module.tags) && module.tags.map((tag, i) => (
              <span
                key={`${tag}-${i}`}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-base border"
                style={{ color: moduleColor, backgroundColor: `${moduleColor}0f`, borderColor: `${moduleColor}33` }}
              >
                <TagIcon className="h-4 w-4" />
                {tag}
              </span>
            ))}

            <div className="flex-1 min-w-[8px]" />

            {module.key && (
              <div
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-base border bg-black/60"
                style={{ borderColor: `${moduleColor}33` }}
              >
                <KeyIcon className="h-4 w-4" style={{ color: moduleColor }} />
                <span className="font-mono">{shorten(module.key)}</span>
                <CopyButton size="sm" content={module.key} />
              </div>
            )}

            <div
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-base border bg-black/60"
              style={{ borderColor: `${moduleColor}33` }}
            >
              <ClockIcon className="h-4 w-4" style={{ color: moduleColor }} />
              <span className="font-medium">{time2str(module.time)}</span>
            </div>

            <button
              onClick={handleSync}
              disabled={syncing}
              className="inline-flex items-center justify-center h-10 w-10 rounded-lg border-2 transition font-bold text-lg"
              style={{ borderColor: `${moduleColor}4D`, color: moduleColor, backgroundColor: syncing ? `${moduleColor}10` : 'transparent' }}
              title="Sync"
            >
              <ArrowPathIcon className={`h-5 w-5 ${syncing ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Professional Tabs */}
        <div className="flex border-b border-white/10 bg-black/90">
          {tabs.map(({ id, icon: Icon }) => {
            const active = activeTab === id
            return (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className="group relative px-6 py-3 text-base font-bold flex items-center gap-2 transition-all"
              >
                {active && (
                  <motion.div layoutId="activeTab" className="absolute inset-0" style={{ backgroundColor: `${moduleColor}10` }} />
                )}
                <Icon className="h-5 w-5 relative z-10" style={{ color: active ? moduleColor : `${moduleColor}80` }} />
                <span className="relative z-10 uppercase tracking-wide" style={{ color: active ? moduleColor : `${moduleColor}80` }}>
                  {id}
                </span>
                {active && (
                  <motion.div layoutId="activeTabBorder" className="absolute bottom-0 left-0 right-0 h-1" style={{ backgroundColor: moduleColor }} />
                )}
              </button>
            )
          })}
        </div>

        {/* Content Area */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="p-4"
          >
            {activeTab === 'app' && (
              module.url_app ? (
                <div className="rounded-lg border border-white/10 overflow-hidden">
                  <ModuleApp module={module} moduleColor={moduleColor} />
                </div>
              ) : (
                <div className="h-[400px] flex items-center justify-center text-xl text-white/70">
                  <div className="text-center">
                    <ComputerDesktopIcon className="h-16 w-16 mx-auto mb-4 opacity-70" />
                    <p className="font-bold">No Application Available</p>
                  </div>
                </div>
              )
            )}
            {activeTab === 'content' && (
              <ModuleContent
                files={module.content || {}}
                showSearch={true}
                compactMode={false}
              />
            )}

            {activeTab === 'api' && (
              <div className="rounded-lg border border-white/10">
                <ModuleSchema mod={module} />
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
