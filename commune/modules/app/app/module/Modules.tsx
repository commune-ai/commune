'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Client } from '@/app/client/client'
import { Loading } from '@/app/components/Loading'
import ModuleCard from '@/app/module/ModuleCard'
import { CreateModule } from '@/app/module/ModuleCreate'
import { ModuleType } from '@/app/types/module'
import { Footer } from '@/app/components/Footer'
import { useSearchContext } from '@/app/context/SearchContext'
import { useUserContext } from '@/app/context/UserContext'
import { ChevronLeft, ChevronRight, RefreshCcw, Plus, X } from 'lucide-react'

type SortKey = 'recent' | 'name' | 'author'

interface ModulesState {
  modules: ModuleType[]
  loading: boolean
  error: string | null
  page: number
  hasMore: boolean
}

export default function Modules() {
  const { keyInstance } = useUserContext()
  const client = useMemo(() => new Client(undefined, keyInstance), [keyInstance])
  const { searchFilters } = useSearchContext()

  const [showCreateForm, setShowCreateForm] = useState(false)
  const [sort, setSort] = useState<SortKey>('recent')
  const [maxCols, setMaxCols] = useState(4)

  const incCols = () => setMaxCols((c) => Math.min(6, c + 1))
  const decCols = () => setMaxCols((c) => Math.max(2, c - 1))

  const cardMinPx = useMemo(() => {
    const map: Record<number, number> = { 2: 600, 3: 400, 4: 320, 5: 280, 6: 240 }
    return map[Math.min(6, Math.max(2, maxCols))]
  }, [maxCols])

  const gapPx = 16
  const maxGridWidth = '100%'

  const userPageSize = searchFilters.pageSize || 24
  const searchTerm = searchFilters.searchTerm?.trim() || ''

  const [state, setState] = useState<ModulesState>({
    modules: [],
    loading: false,
    error: null,
    page: 1,
    hasMore: true,
  })

  const abortRef = useRef<AbortController | null>(null)
  const sentinelRef = useRef<HTMLDivElement | null>(null)

  const sortModules = useCallback(
    (mods: ModuleType[]) => {
      switch (sort) {
        case 'name':
          return [...mods].sort((a, b) => (a.name || '').localeCompare(b.name || ''))
        case 'author':
          return [...mods].sort((a, b) => (a.author || '').localeCompare(b.author || ''))
        case 'recent':
        default:
          return [...mods].sort((a, b) => (b.time || 0) - (a.time || 0))
      }
    },
    [sort]
  )

  const resetAndFetch = useCallback(() => {
    setState({ modules: [], loading: false, error: null, page: 1, hasMore: true })
  }, [])

  const fetchPage = useCallback(
    async (page: number, append: boolean) => {
      if (!keyInstance || state.loading || !state.hasMore && append) return
      abortRef.current?.abort()
      const ac = new AbortController()
      abortRef.current = ac
      setState((p) => ({ ...p, loading: true, error: null }))

      try {
        const params: Record<string, any> = {
          page,
          page_size: userPageSize,
          ...(searchTerm ? { search: searchTerm } : {}),
        }

        const raw = (await client.call('modules', params, { signal: ac.signal })) as ModuleType[]
        const pageItems = Array.isArray(raw) ? raw : []
        const nextList = append ? [...state.modules, ...pageItems] : pageItems
        const dedup = new Map(nextList.map((m) => [m.key, m]))
        const sorted = sortModules([...dedup.values()])

        setState({
          modules: sorted,
          loading: false,
          error: null,
          page,
          hasMore: pageItems.length >= userPageSize,
        })
      } catch (err: any) {
        if (err.name === 'AbortError') return
        setState((p) => ({ ...p, loading: false, error: err?.message || 'Failed to fetch modules' }))
      }
    },
    [client, keyInstance, searchTerm, userPageSize, sortModules, state.modules, state.hasMore, state.loading]
  )

  useEffect(() => {
    if (!keyInstance) return
    resetAndFetch()
  }, [keyInstance, searchTerm, sort, userPageSize, resetAndFetch])

  useEffect(() => {
    if (!keyInstance) return
    fetchPage(1, false)
  }, [keyInstance, searchTerm, sort, userPageSize])

  useEffect(() => {
    if (!sentinelRef.current) return
    const el = sentinelRef.current
    const io = new IntersectionObserver((entries) => {
      const first = entries[0]
      if (first?.isIntersecting && !state.loading && state.hasMore) {
        fetchPage(state.page + 1, true)
        setState((p) => ({ ...p, page: p.page + 1 }))
      }
    }, { rootMargin: '1200px 0px' })

    io.observe(el)
    return () => io.disconnect()
  }, [state.loading, state.hasMore, state.page, fetchPage])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === '[') { e.preventDefault(); decCols() }
      if (e.key === ']') { e.preventDefault(); incCols() }
      if (e.key.toLowerCase() === 'r') { e.preventDefault(); fetchPage(1, false) }
      if (e.key.toLowerCase() === 'c') { e.preventDefault(); setShowCreateForm(true) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [fetchPage])

  const retry = () => fetchPage(Math.max(1, state.page), !!state.modules.length)

  return (
    <div className="min-h-screen bg-black text-green-500 font-mono">
      <header className="sticky top-0 z-40 border-b border-green-900/40 bg-black/80 backdrop-blur">
        <div className="mx-auto px-3 py-2 flex items-center gap-3">
          <div className="flex-1 truncate">
            <span className="tracking-wide">modules</span>
            <span className="mx-2 text-green-400/70">•</span>
            <span className="text-green-400/80">{state.modules.length}</span>
            {searchTerm && (
              <>
                <span className="mx-2 text-green-400/70">•</span>
                <span className="text-green-400/70">query:</span>
                <span className="ml-1">{searchTerm}</span>
              </>
            )}
          </div>

          <div className="hidden md:flex items-center gap-2">
            <label className="text-xs text-green-400/70">density</label>
            <button
              aria-label="decrease columns"
              onClick={decCols}
              className="p-1 border border-green-600 rounded hover:bg-green-900/30"
            >
              <ChevronLeft size={16} />
            </button>
            <div className="px-2 tabular-nums">{maxCols}</div>
            <button
              aria-label="increase columns"
              onClick={incCols}
              className="p-1 border border-green-600 rounded hover:bg-green-900/30"
            >
              <ChevronRight size={16} />
            </button>
          </div>

          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}
            className="bg-black border border-green-700 text-green-400 text-sm rounded px-2 py-1"
            aria-label="sort modules"
          >
            <option value="recent">recent</option>
            <option value="name">name</option>
            <option value="author">author</option>
          </select>

          <button
            onClick={() => fetchPage(1, false)}
            className="inline-flex items-center gap-2 border border-green-600 rounded px-2 py-1 hover:bg-green-900/30"
          >
            <RefreshCcw size={16} /> refresh
          </button>

          <button
            onClick={() => setShowCreateForm(true)}
            className="inline-flex items-center gap-2 border border-green-500 bg-green-900/20 rounded px-2 py-1 hover:bg-green-900/40"
          >
            <Plus size={16} /> create
          </button>
        </div>
      </header>

      {state.error && (
        <div className="mx-auto px-3">
          <div className="mt-3 flex items-center justify-between p-2 border border-red-500 text-red-400 rounded">
            <span>error: {state.error}</span>
            <div className="flex items-center gap-2">
              <button onClick={retry} className="px-2 py-1 border border-red-500 rounded hover:bg-red-900/20">
                retry
              </button>
              <button
                onClick={() => setState((p) => ({ ...p, error: null }))}
                className="px-2 py-1 border border-red-500 rounded hover:bg-red-900/20"
              >
                <X size={14} />
              </button>
            </div>
          </div>
        </div>
      )}

      {showCreateForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
          <div className="border border-green-500 bg-black p-3 rounded w-full max-w-3xl">
            <CreateModule onClose={() => setShowCreateForm(false)} onSuccess={() => setShowCreateForm(false)} />
          </div>
        </div>
      )}

      <main className="p-3" role="main">
        {!state.loading && state.modules.length === 0 && !state.error && (
          <div className="mx-auto max-w-3xl my-20 text-center">
            <div className="text-green-300/80 mb-3">
              {searchTerm ? 'no modules match your search.' : 'no modules yet.'}
            </div>
            <button
              onClick={() => setShowCreateForm(true)}
              className="inline-flex items-center gap-2 border border-green-600 rounded px-3 py-2 hover:bg-green-900/30"
            >
              <Plus size={16} /> create your first module
            </button>
          </div>
        )}

        <div
          role="list"
          className="grid gap-4 mx-auto"
          style={{
            gridTemplateColumns: `repeat(auto-fill, minmax(${cardMinPx}px, 1fr))`,
            maxWidth: maxGridWidth,
          }}
        >
          {state.modules.map((m) => (
            <div key={m.key} role="listitem">
              <ModuleCard module={m} />
            </div>
          ))}

          {state.loading &&
            Array.from({ length: Math.max(3, Math.min(userPageSize, maxCols * 2)) }).map((_, i) => (
              <div
                key={`skeleton-${i}`}
                className="h-40 rounded border border-green-900/40 bg-gradient-to-r from-green-900/10 via-green-800/10 to-green-900/10 animate-pulse"
              />
            ))}
        </div>

        <div ref={sentinelRef} className="h-8" />

        {state.loading && state.modules.length === 0 && (
          <div className="py-10 text-center">
            <Loading />
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}