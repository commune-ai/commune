'use client'

import { DocumentDuplicateIcon, CheckIcon } from '@heroicons/react/20/solid'
import { useState } from 'react'

type TCodeComponentProps = { code: string }

export const CopyButton = (props: TCodeComponentProps) => {
  const { code } = props
  const [copied, setCopied] = useState(false)

  async function copyTextToClipboard(text: string) {
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
    <button
      className={`h-8 px-2 border ${copied ? 'border-green-400 bg-green-500/10' : 'border-green-500 hover:bg-green-500 hover:text-black'} text-green-400 transition-colors`}
      onClick={() => copyTextToClipboard(code)}
      title={copied ? 'Copied!' : 'Copy'}
    >
      {!copied && <DocumentDuplicateIcon height={16} />}
      {copied && <CheckIcon height={16} />}
    </button>
  )
}