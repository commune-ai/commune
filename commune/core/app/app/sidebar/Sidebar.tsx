'use client'
import { useState, useEffect } from 'react'
import { Home, Package, Info, ChevronLeft, ChevronRight } from 'lucide-react'

interface NavItem {
  icon: any
  label: string
  href: string
}

interface SidebarProps {
  isExpanded: boolean
  onToggleExpand: () => void
}

const NAV_ITEMS: NavItem[] = [
  { icon: Home, label: 'HOME', href: '/' },
  { icon: Package, label: 'MODULES', href: '/modules' },
  { icon: Info, label: 'ABOUT', href: '/about' },
]

export const Sidebar = ({ isExpanded, onToggleExpand }: SidebarProps) => {
  return (
    <div 
      className={`fixed left-0 top-0 h-full bg-black border-r border-green-500 z-40 transition-all duration-300 ${
        isExpanded ? 'w-64' : 'w-16'
      }`}
    >
      <button
        onClick={onToggleExpand}
        className="absolute -right-3 top-20 bg-black border border-green-500 rounded-full p-1 hover:bg-green-500 hover:text-black transition-colors z-10"
      >
        {isExpanded ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
      </button>

      <nav className="p-4 mt-20">
        <ul className="space-y-4">
          {NAV_ITEMS.map((item, index) => {
            const Icon = item.icon
            return (
              <li key={`${item.href}-${index}`}>
                <a 
                  href={item.href} 
                  className="flex items-center gap-3 text-green-500 hover:text-green-400 font-mono uppercase transition-colors"
                >
                  <Icon size={20} className="flex-shrink-0" />
                  {isExpanded && (
                    <span className="transition-opacity duration-300">
                      {item.label}
                    </span>
                  )}
                </a>
              </li>
            )
          })}
        </ul>
      </nav>
    </div>
  )
}

export default Sidebar