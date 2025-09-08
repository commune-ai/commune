'use client'

import { DocumentDuplicateIcon, CheckIcon } from '@heroicons/react/20/solid'
import { useState } from 'react'
import { copyToClipboard } from '@/app/utils' // Import the utility function

type TCodeComponentProps = { code: string }

export const CopyButton = (props: TCodeComponentProps, className=null) => {
  const { code } = props
  const [copied, setCopied] = useState(false)
  if (!className) {
    className = "p-1 hover:bg-black/20 rounded transition-colors flex items-center justify-center"
  }

  async function copyTextToClipboard(text: string) {
    copyToClipboard(text)
    setCopied(true)
    setTimeout(() => {
      setCopied(false)
    }, 1000)

    if ('clipboard' in navigator) {
      return await navigator.clipboard.writeText(text)
    } else {
      return document.execCommand('copy', true, text)
    }
  }

  return (

    
    // stop propagation on button click
    <div onClick={(e) => e.stopPropagation()}>
    <button
      className={className}
      onClick={() => copyTextToClipboard(code)}
      title={copied ? 'Copied!' : 'Copy'}
    >
      {!copied && <DocumentDuplicateIcon height={16} />}
      {copied && <CheckIcon height={16} />}
    </button>
    </div>
  )

}