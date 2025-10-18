import config from '@/config.json'
import Image from 'next/image'
import React from 'react'

const navigation = {
  social: [
      {
        name: 'GitHub',
        href: config.github,
        icon: '/github.svg',
      },
      {
        name: 'Discord',
        href: config.discord,
        icon: '/discord.svg',
      },
      {
        name: 'X',
        href: config.x,
        icon: '/x.svg',
      },
      {
        name: 'Telegram',
        href: config.telegram,
        icon: '/telegram.svg',
      },
  ],
}

export const Footer = () => {
  return (
    <footer className='mt-8 bg-black border-t border-green-500'>
      <div className='mx-auto flex max-w-7xl flex-col items-center px-6 py-6 lg:px-8'>
        <div className="flex items-center space-x-4">
          {navigation.social.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className="text-green-500 hover:bg-green-500 hover:text-black p-2 border border-green-500"
              aria-label={item.name}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span className="font-mono uppercase">[{item.name}]</span>
            </a>
          ))}
        </div>
        <div className="mt-4 text-green-500 font-mono text-xs">
          © 
        </div>
      </div>
    </footer>
  )
}