"use client";
import { ReactNode } from 'react';
import { WagmiProvider, createConfig, http } from 'wagmi';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { base, baseSepolia, mainnet, sepolia, defineChain } from 'viem/chains';

const CHAIN_ID = Number(process.env.NEXT_PUBLIC_CHAIN_ID || '0');
const RPC = process.env.NEXT_PUBLIC_RPC_HTTP || '';

// Allow any EVM endpoint by env
const custom = defineChain({
  id: CHAIN_ID || baseSepolia.id,
  name: 'Custom',
  nativeCurrency: { name: 'ETH', symbol: 'ETH', decimals: 18 },
  rpcUrls: { default: { http: [RPC||'http://localhost:8545'] } },
});

const config = createConfig({
  chains: [custom],
  transports: { [custom.id]: http(RPC||'http://localhost:8545') },
});

const qc = new QueryClient();

export function Providers({ children }: { children: ReactNode }) {
  return (
    <WagmiProvider config={config}>
      <QueryClientProvider client={qc}>{children}</QueryClientProvider>
    </WagmiProvider>
  );
}
