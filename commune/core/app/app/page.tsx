'use client'

import { useEffect, useMemo, useState } from 'react'
import { Client } from '@/app/client/client'
import { Loading } from '@/app/components/Loading'
import ModuleCard from '@/app/module/ModuleCard'
import { CreateModule } from '@/app/module/ModuleCreate'
import { ModuleType } from '@/app/types/module'
import { Footer } from '@/app/components/Footer'
import { useSearchContext } from '@/app/context/SearchContext'
import { useUserContext } from '@/app/context/UserContext'

interface ModulesState {
  modules: ModuleType[]
  n: number
  loading: boolean
  error: string | null
}

export default function Modules() {
  const { keyInstance } = useUserContext()
  const client = useMemo(() => new Client(undefined, keyInstance), [keyInstance])
  const { searchFilters } = useSearchContext()

  const [showCreateForm, setShowCreateForm] = useState(false)
  const [state, setState] = useState<ModulesState>({ modules: [], n: 0, loading: false, error: null })

  // Slider: desired MAX columns (1..4). Grid auto-fits up to this count.
  const [maxCols, setMaxCols] = useState(3)

  // Choose a minimum card width by density target (tweak if you want tighter/looser)
  const cardMinPx = useMemo(() => {
    const map: Record<number, number> = { 1: 680, 2: 460, 3: 340, 4: 260 }
    return map[Math.min(4, Math.max(1, maxCols))]
  }, [maxCols])

  // cap container width so we never exceed maxCols visually
  const gapPx = 16 // matches Tailwind gap-4
  const maxGridWidth = `calc(${cardMinPx}px * ${maxCols} + ${gapPx}px * (${maxCols} - 1))`

  const page = searchFilters.page || 1
  const pageSize = searchFilters.pageSize || 20

  const fetchModules = async () => {
    setState(p => ({ ...p, loading: true, error: null }))
    try {
      if (!keyInstance) throw new Error('No key instance available')
      const params: any = {
        page,
        page_size: pageSize,
        ...(searchFilters.searchTerm ? { search: searchFilters.searchTerm } : {})
      }
      const modulesData: ModuleType[] = await client.call('modules', params)
      setState({ modules: modulesData, n: modulesData.length, loading: false, error: null })
    } catch (err: any) {
      setState({ modules: [], n: 0, loading: false, error: err.message || 'Failed to fetch modules' })
    }
  }

  const handleCreateSuccess = () => {
    setShowCreateForm(false)
    setState(p => ({ ...p, error: null }))
  }

  useEffect(() => {
    if (keyInstance) fetchModules()
  }, [searchFilters.searchTerm, page, pageSize, keyInstance])

  return (
    <div className="min-h-screen bg-black text-green-500 font-mono">
      {state.error && (
        <div className="flex items-center justify-between p-2 border border-red-500 text-red-500" role="alert">
          <span>ERROR: {state.error}</span>
          <button onClick={() => setState(p => ({ ...p, error: null }))} className="px-2">[X]</button>
        </div>
      )}

      {showCreateForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black">
          <div className="border border-green-500 bg-black p-3">
            <CreateModule onClose={() => setShowCreateForm(false)} onSuccess={handleCreateSuccess} />
          </div>
        </div>
      )}

      <main className="p-2" role="main">
        {/* minimal slider */}

        {state.loading && (
          <div className="py-6 text-center"><Loading /></div>
        )}

        {!state.loading && state.modules.length === 0 && (
          <div className="py-6 text-center">
            {searchFilters.searchTerm ? 'NO MODULES MATCH YOUR SEARCH.' : 'NO MODULES AVAILABLE.'}
          </div>
        )}

        {!state.loading && state.modules.length > 0 && (
          <div
            role="list"
            className="grid gap-4 mx-auto"
            style={{
              // auto-fit into as many tracks as will fit, each at least cardMinPx wide,
              // but never exceed maxCols by capping the grid's max width.
              gridTemplateColumns: `repeat(auto-fit, minmax(${cardMinPx}px, 1fr))`,
              maxWidth: maxGridWidth,
            }}
          >
            {state.modules.map((m) => (
              <div key={m.key} role="listitem">
                <ModuleCard module={m} />
              </div>
            ))}
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}
