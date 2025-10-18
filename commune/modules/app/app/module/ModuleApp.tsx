'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { GlobeAltIcon, ArrowLeftIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline'
import Link from 'next/link'
import { ModuleType } from '@/app/types/module'

interface ModuleAppProps {
  module: ModuleType
  moduleColor: string
}

export function ModuleApp({ module, moduleColor }: ModuleAppProps) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Check if the URL is valid
    if (!module.url_app) {
      setError('No application URL provided')
      setLoading(false)
      return
    }

    // Validate URL format
    try {
      const url = new URL(module.url_app)
      // URL is valid, iframe will handle loading
      setLoading(false)
    } catch (e) {
      setError('Invalid application URL')
      setLoading(false)
    }
  }, [module.url_app])

  if (error) {
    return (
      <div className='flex min-h-[600px] items-center justify-center'>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className='text-center'
        >
          <ExclamationTriangleIcon className='mx-auto h-16 w-16 text-red-500 mb-4' />
          <h3 className='text-xl font-semibold text-red-500 mb-2'>Application Error</h3>
          <p className='text-gray-400 mb-6'>{error}</p>
          <Link
            href={`/module/${module.name}`}
            className='inline-flex items-center gap-2 rounded-lg border px-4 py-2 transition-all'
            style={{ 
              borderColor: `${moduleColor}4D`,
              color: moduleColor
            }}
          >
            <ArrowLeftIcon className='h-4 w-4' />
            Back to Module
          </Link>
        </motion.div>
      </div>
    )
  }

  return (
    <div className='relative w-full h-full min-h-[600px]'>
      {loading && (
        <div className='absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm z-10'>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className='h-8 w-8 rounded-full border-2 border-t-transparent'
            style={{ borderColor: moduleColor }}
          />
        </div>
      )}
      
      <iframe
        src={module.url_app}
        className='w-full h-full border-0'
        title={`${module.name} Application`}
        onLoad={() => setLoading(false)}
        onError={() => {
          setError('Failed to load application')
          setLoading(false)
        }}
        allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture'
        allowFullScreen
      />
      
      {/* External link indicator */}
      <motion.a
        href={module.url_app}
        target='_blank'
        rel='noopener noreferrer'
        className='absolute top-4 right-4 flex items-center gap-2 rounded-full bg-black/80 px-4 py-2 text-sm backdrop-blur-sm transition-all hover:scale-105'
        style={{ 
          borderColor: `${moduleColor}4D`,
          color: moduleColor,
          border: '1px solid'
        }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <GlobeAltIcon className='h-4 w-4' />
        Open in new tab
      </motion.a>
    </div>
  )
}
