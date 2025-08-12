'use client'
import { useState, useEffect } from 'react'
import { Activity, Cpu, Database, Globe, Hash, Layers, Network, Server, Zap } from 'lucide-react'
import type { Client } from '@/app/client/client'

interface ChainInfoProps {
  client: Client
}

interface ChainData {
  blockNumber?: number
  blockHash?: string
  chainId?: string
  networkName?: string
  gasPrice?: string
  totalSupply?: string
  validators?: number
  transactionCount?: number
  averageBlockTime?: number
  lastBlockTime?: string
}

export const ChainInfo = ({ client }: ChainInfoProps) => {
  const [chainData, setChainData] = useState<ChainData>({})
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchChainInfo()
    const interval = setInterval(fetchChainInfo, 10000) // Update every 10 seconds
    return () => clearInterval(interval)
  }, [client])

  const fetchChainInfo = async () => {
    try {
      setIsLoading(true)
      // Fetch chain information from the client
      const info = await client.call('chain/info')
      setChainData(info)
      setError(null)
    } catch (err) {
      console.error('Error fetching chain info:', err)
      setError('Failed to fetch chain information')
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading && Object.keys(chainData).length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-green-500 font-mono">
          <Zap className="animate-spin mx-auto mb-4" size={32} />
          <p>Loading chain information...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12 text-red-500 font-mono">
        <Network size={48} className="mx-auto mb-4 opacity-50" />
        <p>{error}</p>
      </div>
    )
  }

  const infoItems = [
    {
      icon: Layers,
      label: 'BLOCK HEIGHT',
      value: chainData.blockNumber?.toLocaleString() || 'N/A',
      color: 'text-green-400'
    },
    {
      icon: Hash,
      label: 'LATEST BLOCK HASH',
      value: chainData.blockHash ? `${chainData.blockHash.slice(0, 10)}...${chainData.blockHash.slice(-8)}` : 'N/A',
      color: 'text-green-500',
      fullValue: chainData.blockHash
    },
    {
      icon: Globe,
      label: 'NETWORK',
      value: chainData.networkName || 'Unknown',
      color: 'text-green-400'
    },
    {
      icon: Database,
      label: 'CHAIN ID',
      value: chainData.chainId || 'N/A',
      color: 'text-green-500'
    },
    {
      icon: Activity,
      label: 'GAS PRICE',
      value: chainData.gasPrice || 'N/A',
      color: 'text-green-400'
    },
    {
      icon: Server,
      label: 'VALIDATORS',
      value: chainData.validators?.toString() || 'N/A',
      color: 'text-green-500'
    },
    {
      icon: Cpu,
      label: 'TOTAL TRANSACTIONS',
      value: chainData.transactionCount?.toLocaleString() || 'N/A',
      color: 'text-green-400'
    },
    {
      icon: Zap,
      label: 'AVG BLOCK TIME',
      value: chainData.averageBlockTime ? `${chainData.averageBlockTime}s` : 'N/A',
      color: 'text-green-500'
    }
  ]

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-6">
        <Network size={16} className="animate-pulse" />
        <span>BLOCKCHAIN INFORMATION</span>
      </div>

      {/* Chain Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {infoItems.map((item, index) => {
          const Icon = item.icon
          return (
            <div
              key={index}
              className="p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20 hover:border-green-500/40 transition-all"
            >
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-green-500/10 border border-green-500/30">
                  <Icon size={20} className={item.color} />
                </div>
                <div className="flex-1">
                  <div className="text-green-600/70 text-xs font-mono uppercase mb-1">
                    {item.label}
                  </div>
                  <div className={`${item.color} font-mono text-sm font-bold break-all`}>
                    {item.value}
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Live Status Indicator */}
      <div className="mt-6 p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-green-400 font-mono text-sm">LIVE STATUS</span>
          </div>
          <span className="text-green-600/70 font-mono text-xs">
            Last updated: {new Date().toLocaleTimeString()}
          </span>
        </div>
      </div>

      {/* Additional Chain Metrics */}
      {chainData.totalSupply && (
        <div className="p-4 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent border border-green-500/20">
          <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-2">
            <Database size={16} />
            <span>TOTAL SUPPLY</span>
          </div>
          <div className="text-green-400 font-mono text-lg font-bold">
            {chainData.totalSupply}
          </div>
        </div>
      )}
    </div>
  )
}