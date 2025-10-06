'use client'
import { createContext, useContext, useState, ReactNode } from 'react'

interface SidebarContextType {
  isSidebarExpanded: boolean
  toggleSidebar: () => void
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined)

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(false)

  const toggleSidebar = () => {
    setIsSidebarExpanded(!isSidebarExpanded)
  }

  return (
    <SidebarContext.Provider value={{ isSidebarExpanded, toggleSidebar }}>
      {children}
    </SidebarContext.Provider>
  )
}

export function useSidebarContext() {
  const context = useContext(SidebarContext)
  if (context === undefined) {
    throw new Error('useSidebarContext must be used within a SidebarProvider')
  }
  return context
}