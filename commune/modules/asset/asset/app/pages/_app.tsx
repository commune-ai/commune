import '../styles/globals.css'
import type { AppProps } from 'next/app'
import { WagmiConfig, createConfig, configureChains, mainnet } from 'wagmi'
import { localhost } from 'wagmi/chains'
import { publicProvider } from 'wagmi/providers/public'
import { RainbowKitProvider, getDefaultWallets } from '@rainbow-me/rainbowkit'
import '@rainbow-me/rainbowkit/styles.css'

const { chains, publicClient, webSocketPublicClient } = configureChains(
  [localhost],
  [publicProvider()]
)

const { connectors } = getDefaultWallets({
  appName: 'StableCoin Vault',
  projectId: 'YOUR_PROJECT_ID',
  chains
})

const config = createConfig({
  autoConnect: true,
  connectors,
  publicClient,
  webSocketPublicClient,
})

export default function App({ Component, pageProps }: AppProps) {
  return (
    <WagmiConfig config={config}>
      <RainbowKitProvider chains={chains}>
        <Component {...pageProps} />
      </RainbowKitProvider>
    </WagmiConfig>
  )
}