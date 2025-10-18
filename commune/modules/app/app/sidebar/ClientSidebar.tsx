'use client'
import { Sidebar } from './Sidebar'
import { useSidebarContext } from '../context/SidebarContext'

export function ClientSidebar({ children }: { children: React.ReactNode }) {
  const { isSidebarExpanded, toggleSidebar } = useSidebarContext()

  return (
    <>
      <Sidebar 
        isExpanded={isSidebarExpanded} 
        onToggleExpand={toggleSidebar} 
      />
      <main className={`pt-28 transition-all duration-300 ${isSidebarExpanded ? 'ml-64 pl-8' : 'ml-16 pl-8'}`}>
        <div className="min-h-screen">
          {children}
        </div>
      </main>
    </>
  )
}