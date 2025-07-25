'use client'
import { useState, useEffect, useRef } from 'react'
import { X, Menu, ChevronLeft, ChevronRight, RefreshCw, Search } from 'lucide-react'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  isCollapsed?: boolean
  onToggleCollapse?: () => void
  // Pagination props
  currentPage?: number
  totalPages?: number
  onPageChange?: (page: number) => void
  // Sync props
  onSync?: () => void
  isSyncing?: boolean
}

export const Sidebar = ({ 
  isOpen, 
  onClose, 
  isCollapsed = false, 
  onToggleCollapse,
  currentPage = 1,
  totalPages = 1,
  onPageChange,
  onSync,
  isSyncing = false
}: SidebarProps) => {
  const [isAnimating, setIsAnimating] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [dragStartX, setDragStartX] = useState(0)
  const [sidebarWidth, setSidebarWidth] = useState(isCollapsed ? 64 : 320)
  const [showPageSearch, setShowPageSearch] = useState(false)
  const [pageSearchQuery, setPageSearchQuery] = useState('')
  const sidebarRef = useRef<HTMLDivElement>(null)
  const MIN_WIDTH = 64
  const MAX_WIDTH = 480
  const COLLAPSED_WIDTH = 64
  const EXPANDED_WIDTH = 320
  const PAGE_THRESHOLD = 10 // Show search when more than 10 pages

  useEffect(() => {
    if (isOpen) {
      setIsAnimating(true)
    }
  }, [isOpen])

  useEffect(() => {
    setSidebarWidth(isCollapsed ? COLLAPSED_WIDTH : EXPANDED_WIDTH)
  }, [isCollapsed])

  const handleClose = () => {
    setIsAnimating(false)
    setTimeout(onClose, 200)
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    if (isCollapsed) return
    setIsDragging(true)
    setDragStartX(e.clientX)
    e.preventDefault()
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || isCollapsed) return
      
      const deltaX = e.clientX - dragStartX
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, sidebarWidth - deltaX))
      setSidebarWidth(newWidth)
      setDragStartX(e.clientX)
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'ew-resize'
      document.body.style.userSelect = 'none'
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'auto'
      document.body.style.userSelect = 'auto'
    }
  }, [isDragging, dragStartX, sidebarWidth, isCollapsed])

  // Generate visible page numbers based on search
  const getVisiblePages = () => {
    if (!pageSearchQuery) {
      // Show limited pages when no search
      if (totalPages <= PAGE_THRESHOLD) {
        return Array.from({ length: totalPages }, (_, i) => i + 1)
      }
      // Show first few, current area, and last few
      const pages = new Set<number>()
      // First 3 pages
      for (let i = 1; i <= Math.min(3, totalPages); i++) pages.add(i)
      // Current page area
      for (let i = Math.max(1, currentPage - 1); i <= Math.min(totalPages, currentPage + 1); i++) pages.add(i)
      // Last 3 pages
      for (let i = Math.max(totalPages - 2, 1); i <= totalPages; i++) pages.add(i)
      return Array.from(pages).sort((a, b) => a - b)
    }
    
    // Filter pages based on search query
    const query = pageSearchQuery.toLowerCase()
    return Array.from({ length: totalPages }, (_, i) => i + 1)
      .filter(page => page.toString().includes(query))
  }

  // Don't render if not open and not animating
  if (!isOpen && !isAnimating) return null

  const visiblePages = getVisiblePages()

  return (
    <>
      {/* Backdrop - clicking this will close the sidebar */}
      <div 
        className={`fixed inset-0 bg-black/50 z-40 transition-opacity duration-200 ${
          isAnimating && isOpen ? 'opacity-100' : 'opacity-0'
        }`}
        onClick={handleClose}
      />
      
      {/* Sidebar */}
      <div 
        ref={sidebarRef}
        className={`fixed top-0 left-0 h-full bg-black border-r border-green-500 z-50 transform transition-transform duration-200 ${
          isAnimating && isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
        style={{ width: `${sidebarWidth}px` }}
      >
        {/* Left Edge Click Area - New addition for closing sidebar */}
        <div
          className="absolute top-0 left-0 w-8 h-full cursor-pointer hover:bg-green-500/10 transition-colors"
          onClick={handleClose}
          title="Click to close sidebar"
        >
          <div className="flex items-center justify-center h-full opacity-0 hover:opacity-100 transition-opacity">
            <ChevronLeft className="text-green-500" size={20} />
          </div>
        </div>
        
        {/* Drag Handle */}
        {!isCollapsed && (
          <div
            className="absolute top-0 right-0 w-1 h-full cursor-ew-resize hover:bg-green-500/50 transition-colors"
            onMouseDown={handleMouseDown}
          />
        )}
        
        <div className="p-4 border-b border-green-500">
          <div className="flex items-center justify-between">
            {!isCollapsed && <h2 className="text-green-500 text-xl font-mono uppercase">MENU</h2>}
            <div className="flex items-center gap-2">
              {/* Sync Button */}
              {onSync && (
                <button
                  onClick={onSync}
                  disabled={isSyncing}
                  className="text-green-500 hover:text-green-400 transition-colors disabled:opacity-50"
                  title="Sync modules"
                >
                  <RefreshCw size={20} className={isSyncing ? 'animate-spin' : ''} />
                </button>
              )}
              <button
                onClick={handleClose}
                className="text-green-500 hover:text-green-400 transition-colors"
                title="Close"
              >
                <X size={24} />
              </button>
            </div>
          </div>
        </div>
        
        <nav className="p-4 flex-1">
          <ul className="space-y-4">
            <li>
              <a href="/" className={`block text-green-500 hover:text-green-400 font-mono uppercase transition-colors ${isCollapsed ? 'text-center' : ''}`}>
                {isCollapsed ? 'H' : 'HOME'}
              </a>
            </li>
            <li>
              <a href="/modules" className={`block text-green-500 hover:text-green-400 font-mono uppercase transition-colors ${isCollapsed ? 'text-center' : ''}`}>
                {isCollapsed ? 'M' : 'MODULES'}
              </a>
            </li>
            <li>
              <a href="/about" className={`block text-green-500 hover:text-green-400 font-mono uppercase transition-colors ${isCollapsed ? 'text-center' : ''}`}>
                {isCollapsed ? 'A' : 'ABOUT'}
              </a>
            </li>
            <li>
              <a href="/contact" className={`block text-green-500 hover:text-green-400 font-mono uppercase transition-colors ${isCollapsed ? 'text-center' : ''}`}>
                {isCollapsed ? 'C' : 'CONTACT'}
              </a>
            </li>
          </ul>
        </nav>
        
        {/* Enhanced Pagination Controls */}
        {onPageChange && totalPages > 1 && !isCollapsed && (
          <div className="p-4 border-t border-green-500 space-y-3">
            {/* Compact Page Navigation */}
            <div className="flex items-center justify-between text-green-500 font-mono text-sm">
              <button
                onClick={() => onPageChange(currentPage - 1)}
                disabled={currentPage <= 1}
                className="p-1 border border-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed transition-none"
                title="Previous page"
              >
                <ChevronLeft size={16} />
              </button>
              
              {/* Page selector area */}
              <div className="flex-1 mx-2">
                {totalPages > PAGE_THRESHOLD ? (
                  <div className="relative">
                    {/* Search toggle button */}
                    <button
                      onClick={() => setShowPageSearch(!showPageSearch)}
                      className="w-full px-2 py-1 border border-green-500 hover:bg-green-500/10 text-xs flex items-center justify-center gap-1"
                    >
                      <Search size={12} />
                      <span>{currentPage}/{totalPages}</span>
                    </button>
                    
                    {/* Expandable search */}
                    {showPageSearch && (
                      <div className="absolute bottom-full left-0 right-0 mb-2 bg-black border border-green-500 p-2">
                        <input
                          type="text"
                          placeholder="SEARCH PAGE..."
                          value={pageSearchQuery}
                          onChange={(e) => setPageSearchQuery(e.target.value)}
                          className="w-full px-2 py-1 bg-black border border-green-500/50 text-green-500 placeholder-green-500/30 text-xs focus:outline-none focus:border-green-400"
                          autoFocus
                        />
                        <div className="mt-2 max-h-32 overflow-y-auto">
                          <div className="grid grid-cols-5 gap-1">
                            {visiblePages.map((page) => (
                              <button
                                key={page}
                                onClick={() => {
                                  onPageChange(page)
                                  setShowPageSearch(false)
                                  setPageSearchQuery('')
                                }}
                                className={`px-1 py-1 text-xs border transition-none ${
                                  page === currentPage
                                    ? 'bg-green-500 text-black border-green-500'
                                    : 'border-green-500/50 text-green-500 hover:bg-green-500/20'
                                }`}
                              >
                                {page}
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  /* Direct page buttons for few pages */
                  <div className="flex items-center justify-center gap-1">
                    {visiblePages.map((page, idx) => (
                      <React.Fragment key={page}>
                        {idx > 0 && visiblePages[idx - 1] < page - 1 && (
                          <span className="text-green-500/50 text-xs">...</span>
                        )}
                        <button
                          onClick={() => onPageChange(page)}
                          className={`px-2 py-1 text-xs border transition-none ${
                            page === currentPage
                              ? 'bg-green-500 text-black border-green-500'
                              : 'border-green-500/50 text-green-500 hover:bg-green-500/20'
                          }`}
                        >
                          {page}
                        </button>
                      </React.Fragment>
                    ))}
                  </div>
                )}
              </div>
              
              <button
                onClick={() => onPageChange(currentPage + 1)}
                disabled={currentPage >= totalPages}
                className="p-1 border border-green-500 hover:bg-green-500 hover:text-black disabled:opacity-50 disabled:cursor-not-allowed transition-none"
                title="Next page"
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
        
        {/* Visual Resize Indicator */}
        {isDragging && !isCollapsed && (
          <div className="absolute top-0 left-0 h-full w-full pointer-events-none">
            <div className="absolute right-0 top-0 h-full w-1 bg-green-500 animate-pulse" />
          </div>
        )}
      </div>
    </>
  )
}

export default Sidebar