'use client'
import { UserHeader } from './UserHeader'
import { SearchHeader } from './SearchHeader'
import { LogoHeader } from './LogoHeader'
import { useSidebarContext } from '../context/SidebarContext'

export const Header = () => {
  const { toggleSidebar } = useSidebarContext()
  
  return (
    <header className="fixed top-0 z-[100] w-full bg-black font-mono">
      <nav className="p-4 px-1 mx-auto">
        <div className="flex items-center justify-between gap-2">
          <LogoHeader onToggleSidebar={toggleSidebar} />
          <div className="flex-1 mx-4 flex-1 items-center">
            <SearchHeader />
          </div>
          <UserHeader />
        </div>
      </nav>
    </header>
  )
}

export default Header