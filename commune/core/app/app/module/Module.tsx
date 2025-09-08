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
  GlobeAltIcon,
  ClockIcon,
  KeyIcon,
  ComputerDesktopIcon,
  CubeIcon,
} from '@heroicons/react/24/outline'
import { CopyButton } from '@/app/components/CopyButton'
import { ModuleCode } from './ModuleCode'
import ModuleSchema from './ModuleApi'
import { ModuleApp } from './ModuleApp'
import { motion, AnimatePresence } from 'framer-motion'

type TabType = 'app' | 'api' | 'code'

interface ModuleProps {
  module_name: string
  code?: boolean
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
  if (diff < 60_000) return 'just now'
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
      const params = { module: module_name, update, code: true }
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

  const tabs: Array<{ id: TabType; label: string; icon: any }> = [
    { id: 'api', label: 'API', icon: ServerIcon },
    { id: 'app', label: 'APP', icon: ComputerDesktopIcon },
    { id: 'code', label: 'CODE', icon: CodeBracketIcon },
  ]

  return (
    <div className="min-h-screen bg-black text-white m-0 p-0">
      <div className="w-full">
        {/* Compact Header Row (everything on one line, wraps as needed) */}
        <div className="w-full px-3 py-2 border-b border-white/10">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-sm font-semibold leading-none"
              style={{ color: moduleColor, backgroundColor: `${moduleColor}14`, border: `1px solid ${moduleColor}33` }}
            >
              <CubeIcon className="h-4 w-4" />
              {module.name}
            </span>

            {/* Tags inline on same row */}
            {Array.isArray(module.tags) && module.tags.map((tag, i) => (
              <span
                key={`${tag}-${i}`}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs leading-none border"
                style={{ color: moduleColor, backgroundColor: `${moduleColor}0f`, borderColor: `${moduleColor}33` }}
              >
                <TagIcon className="h-3 w-3" />
                {tag}
              </span>
            ))}

            {/* Spacer to push attributes to the right if lots of tags */}
            <div className="flex-1 min-w-[8px]" />

            {/* Attributes (compact) */}
            {module.key && (
              <div
                className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs leading-none border bg-black/60"
                style={{ borderColor: `${moduleColor}33` }}
              >
                <KeyIcon className="h-3 w-3" style={{ color: moduleColor }} />
                <span className="truncate max-w-[140px]">{shorten(module.key)}</span>
                <CopyButton size="xs" code={module.key} />
              </div>
            )}

            <div
              className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs leading-none border bg-black/60"
              style={{ borderColor: `${moduleColor}33` }}
            >
              <ClockIcon className="h-3 w-3" style={{ color: moduleColor }} />
              <span>{time2str(module.time)}</span>
              <CopyButton size="xs" code={time2str(module.time)} />
            </div>

            {/* Sync button — ultra compact */}
            <button
              onClick={handleSync}
              disabled={syncing}
              className="inline-flex items-center justify-center h-7 w-7 rounded-md border transition"
              style={{ borderColor: `${moduleColor}4D`, color: moduleColor, backgroundColor: syncing ? `${moduleColor}10` : 'transparent' }}
              title="Sync"
            >
              <ArrowPathIcon className={`h-4 w-4 ${syncing ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Compact Tabs */}
        <div className="flex border-b border-white/10">
          {tabs.map(({ id, label, icon: Icon }) => {
            const active = activeTab === id
            return (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className="group relative px-3 py-2 text-xs font-medium flex items-center gap-1"
              >
                {active && (
                  <motion.div layoutId="activeTab" className="absolute inset-0" style={{ backgroundColor: `${moduleColor}10` }} />
                )}
                <Icon className="h-4 w-4 relative z-10" style={{ color: active ? moduleColor : `${moduleColor}80` }} />
                <span className="relative z-10" style={{ color: active ? moduleColor : `${moduleColor}80` }}>
                  {label}
                </span>
                {active && (
                  <motion.div layoutId="activeTabBorder" className="absolute bottom-0 left-0 right-0 h-0.5" style={{ backgroundColor: moduleColor }} />
                )}
              </button>
            )
          })}
          <div className="flex-1" />
          {/* Optional: Visit button kept small and tight if url exists */}
          {module.url && (
            <a
              href={module.url}
              target="_blank"
              rel="noopener noreferrer"
              className="mx-2 my-1 inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs"
              style={{ borderColor: `${moduleColor}40`, color: moduleColor }}
            >
              <GlobeAltIcon className="h-4 w-4" />
              Visit
            </a>
          )}
        </div>

        {/* Content — dense */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -12 }}
            transition={{ duration: 0.15 }}
            className="p-3"
          >
            {activeTab === 'app' && (
              module.url_app ? (
                <div className="rounded-md border border-white/10 overflow-hidden">
                  <ModuleApp module={module} moduleColor={moduleColor} />
                </div>
              ) : (
                <div className="h-[300px] flex items-center justify-center text-sm text-white/70">
                  <div className="text-center">
                    <ComputerDesktopIcon className="h-10 w-10 mx-auto mb-2 opacity-70" />
                    No Application Available
                  </div>
                </div>
              )
            )}

            {activeTab === 'code' && (
              <>
                {/* force compact in child */}
                <ModuleCode
                  files={module.source?.code || {}}
                  title=""
                  showSearch={true}
                  compactMode={true}
                />
              </>
            )}

            {activeTab === 'api' && (
              <div className="rounded-md border border-white/10">
                <ModuleSchema mod={module} />
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
