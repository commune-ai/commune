'use client'
import { History } from 'lucide-react'

interface Transaction {
  type: string
  timestamp: string
  hash: string
}

interface TransactionsTabProps {
  transactions: Transaction[]
}

export const TransactionsTab = ({ transactions }: TransactionsTabProps) => {
  return (
    <div className="space-y-4 animate-fadeIn">
      <div className="flex items-center gap-2 text-green-500/70 text-sm font-mono uppercase mb-4">
        <History size={16} className="animate-spin-slow" />
        <span>TRANSACTION HISTORY</span>
      </div>
      {transactions.length === 0 ? (
        <div className="text-center py-12 text-green-600/50 font-mono">
          <History size={48} className="mx-auto mb-4 opacity-50" />
          <p>No transactions found</p>
        </div>
      ) : (
        <div className="space-y-2">
          {transactions.map((tx, index) => (
            <div key={index} className="p-4 border border-green-500/30 rounded-lg bg-gradient-to-br from-green-500/5 to-transparent hover:border-green-500/50 transition-all">
              <div className="flex justify-between items-center">
                <span className="text-green-400 font-mono text-sm font-bold">{tx.type}</span>
                <span className="text-green-600/70 font-mono text-xs">{tx.timestamp}</span>
              </div>
              <div className="text-green-600/50 font-mono text-xs mt-2">
                {tx.hash}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}