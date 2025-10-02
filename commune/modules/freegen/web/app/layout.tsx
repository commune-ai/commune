import type { Metadata } from 'next';
import './globals.css';
import { Providers } from '@/components/Providers';

export const metadata: Metadata = {
  title: 'Neon Oracle Markets',
  description: 'Cyberpunk oracle-agnostic prediction markets'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <div className="min-h-dvh grid-bg">
            <header className="max-w-6xl mx-auto p-6 flex items-center justify-between">
              <h1 className="text-2xl font-semibold ae">NEON ▷ MARKETS</h1>
              <div id="wallet-slot" />
            </header>
            <main className="max-w-6xl mx-auto p-6">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );}
