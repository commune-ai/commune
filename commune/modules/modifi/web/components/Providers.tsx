"use client";
import { ReactNode } from 'react';
import { WagmiProvider, createConfig, http } from 'wagmi';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { defineChain } from 'viem/chains';

const CHAIN_ID = Number(process.env.NEXT_PUBLIC_CHAIN_ID||'0');
const RPC = process.env.NEXT_PUBLIC_RPC_HTTP || 'http://localhost:8545';
const custom = defineChain({ id: CHAIN_ID||31337, name:'Custom', nativeCurrency:{name:'ETH',symbol:'ETH',decimals:18}, rpcUrls:{ default:{ http:[RPC] } } });
const config = createConfig({ chains:[custom], transports:{ [custom.id]: http(RPC) } });
const qc = new QueryClient();
export function Providers({ children }:{ children: ReactNode }){
  return (<WagmiProvider config={config}><QueryClientProvider client={qc}>{children}</QueryClientProvider></WagmiProvider>);
}
