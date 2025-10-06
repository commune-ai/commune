'use client'
import { useState } from 'react'
import { Sidebar } from './Sidebar'

export function ClientSidebar({ children }: { children: React.ReactNode }) {
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(false)

  return (
    <>
      <Sidebar 
        isExpanded={isSidebarExpanded} 
        onToggleExpand={() => setIsSidebarExpanded(!isSidebarExpanded)} 
      />
      <main className={`pt-28 transition-all duration-300 ${isSidebarExpanded ? 'ml-64' : 'ml-16'}`}>
        <div className="min-h-screen">
          {children}
        </div>
      </main>
    </>
  )
}