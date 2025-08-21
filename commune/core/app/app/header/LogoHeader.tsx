      
      
    

import { useDebounce } from '@/app/hooks/useDebounce'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { CubeIcon } from '@heroicons/react/24/solid'



export const LogoHeader = () => {
    const router = useRouter()

    const moduleColor = '#10b981' // Tailwind's emerald-500 color
    return (
    <div className="relative w-10 h-10 flex-shrink-0">
              <motion.div 
                onClick={() => {router.push('/')}}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.8 }}
                transition={{ duration: 0.2 }}
                className=' rounded-2xl'
                style={{ 
                  backgroundColor: `${moduleColor}15`,
                  boxShadow: `0 0 30px ${moduleColor}30`
                }}
              >
                <CubeIcon className='h-10 w-10' style={{ color: moduleColor }} onClick={() => {router.push('/')}} />
              </motion.div>
              
    </div>
    )
}
