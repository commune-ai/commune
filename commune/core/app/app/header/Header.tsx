'use client'
import { UserHeader } from './UserHeader'
import { SearchHeader } from './SearchHeader'
import { useUserContext } from '@/app/context/UserContext'
import { LogoHeader } from './LogoHeader'
export const Header = () => {
  return (
    <header className="fixed top-0 z-40 w-full bg-black font-mono">
      <nav className="p-4 px-1 mx-auto">
        <div className="flex items-center justify-between gap-2">
          {/* Left side - Search */}
          <LogoHeader />
          <div className="flex-1 mx-4 flex-1 items-center">
            <SearchHeader />
          </div>

          {/* Right side - User Section */}
          <UserHeader />
        </div>
      </nav>
    </header>
  )
}

export default Header