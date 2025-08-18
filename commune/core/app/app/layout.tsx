import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Header } from './header/Header'
import { UserProvider } from './context/UserContext'
import { SearchProvider } from './context/SearchContext'
import { Sidebar } from './sidebar/Sidebar'
import "react-responsive-modal/styles.css"
import "@/app/globals.css"

export const metadata: Metadata = {
  title: "dhub",
  description: "dhub - the hub for modules. built with commune-ai/commune",
  robots: 'all',
  icons: [{ rel: 'icon', url: '/favicon.ico' }]
}

const inter = Inter({ subsets: ['latin'] })

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">

      <body className={`${inter.className} h-full relative bg-black`}>
        <UserProvider>
          <SearchProvider>
          <Header />
                          {/* <Sidebar /> */}

          {/* Main content area */}
          <main className="pt-28">
            <div className="min-h-screen">
              {children}
            </div>
          </main>
          </SearchProvider>
        </UserProvider>
      </body>
    </html>
  )
}