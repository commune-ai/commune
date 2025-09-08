import { ConnectButton } from '@rainbow-me/rainbowkit'
import { useAccount, useContractRead, useContractWrite, usePrepareContractWrite } from 'wagmi'
import { useState, useEffect } from 'react'
import { ethers } from 'ethers'
import contractData from '../contract-address.json'
import StableCoinVaultABI from '../abi/StableCoinVault.json'

export default function Home() {
  const { address, isConnected } = useAccount()
  const [selectedToken, setSelectedToken] = useState('')
  const [amount, setAmount] = useState('')
  const [userBalances, setUserBalances] = useState<any>({});
  
  const contractAddress = contractData.address
  
  // Read accepted tokens
  const { data: acceptedTokens } = useContractRead({
    address: contractAddress as `0x${string}`,
    abi: StableCoinVaultABI,
    functionName: 'getAcceptedTokens',
  })
  
  // Prepare deposit transaction
  const { config: depositConfig } = usePrepareContractWrite({
    address: contractAddress as `0x${string}`,
    abi: StableCoinVaultABI,
    functionName: 'deposit',
    args: [selectedToken, ethers.utils.parseUnits(amount || '0', 6)],
    enabled: !!selectedToken && !!amount && Number(amount) > 0,
  })
  
  const { write: deposit } = useContractWrite(depositConfig)
  
  // Prepare withdraw transaction
  const { config: withdrawConfig } = usePrepareContractWrite({
    address: contractAddress as `0x${string}`,
    abi: StableCoinVaultABI,
    functionName: 'withdraw',
    args: [selectedToken, ethers.utils.parseUnits(amount || '0', 6)],
    enabled: !!selectedToken && !!amount && Number(amount) > 0,
  })
  
  const { write: withdraw } = useContractWrite(withdrawConfig)
  
  // Token names mapping
  const tokenNames: any = {
    '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48': 'USDC',
    '0xdAC17F958D2ee523a2206206994597C13D831ec7': 'USDT',
    '0x6B175474E89094C44Da98b954EedeAC495271d0F': 'DAI'
  }
  
  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <h1 className="text-2xl font-bold">StableCoin Vault</h1>
            <ConnectButton />
          </div>
        </div>
      </nav>
      
      <main className="max-w-4xl mx-auto mt-8 p-6">
        {isConnected ? (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Manage Your Stablecoins</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Token
                </label>
                <select
                  value={selectedToken}
                  onChange={(e) => setSelectedToken(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select a token</option>
                  {acceptedTokens?.map((token: any) => (
                    <option key={token} value={token}>
                      {tokenNames[token] || token}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Amount
                </label>
                <input
                  type="number"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  placeholder="0.00"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <div className="flex space-x-4">
                <button
                  onClick={() => deposit?.()}
                  disabled={!deposit}
                  className="flex-1 bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
                >
                  Deposit
                </button>
                <button
                  onClick={() => withdraw?.()}
                  disabled={!withdraw}
                  className="flex-1 bg-red-500 text-white py-2 px-4 rounded-md hover:bg-red-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
                >
                  Withdraw
                </button>
              </div>
            </div>
            
            <div className="mt-8">
              <h3 className="text-lg font-semibold mb-4">Your Balances</h3>
              <div className="space-y-2">
                {acceptedTokens?.map((token: any) => (
                  <div key={token} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                    <span className="font-medium">{tokenNames[token] || token}</span>
                    <span>{userBalances[token] || '0.00'}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow p-6 text-center">
            <h2 className="text-xl font-semibold mb-4">Welcome to StableCoin Vault</h2>
            <p className="text-gray-600 mb-6">Connect your wallet to start managing your stablecoins</p>
            <ConnectButton />
          </div>
        )}
      </main>
    </div>
  )
}