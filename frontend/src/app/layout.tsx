"use client";
import '@rainbow-me/rainbowkit/styles.css';
import { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import {
  RainbowKitProvider,
  darkTheme,
  connectorsForWallets
} from '@rainbow-me/rainbowkit';
import {
  rainbowWallet,
  walletConnectWallet,
  trustWallet,
  okxWallet,
  ledgerWallet,
  metaMaskWallet
} from '@rainbow-me/rainbowkit/wallets';
import { ConfigProvider} from 'antd';
import localFont from 'next/font/local'
import { Provider } from "react-redux";
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { configureChains, createConfig, sepolia, WagmiConfig } from 'wagmi';
import {
  mainnet,
  polygon,
  optimism,
  arbitrum,
  base,
  zora,
  goerli,
} from 'wagmi/chains';
import { alchemyProvider } from 'wagmi/providers/alchemy';
import { publicProvider } from 'wagmi/providers/public';
import "@fontsource/source-code-pro";
import Providers from "@/app/provider"
import Loading from '@/components/molecules/bittensor/loading';
import NavigationBar from '@/components/organisms/navbar/navigation-bar';
import Footer from '@/components/templates/footer/footer';
import Head from '@/components/templates/head';
import { ColorProvider } from '@/context/color-widget-provider';
import ThemeProvider from '@/context/toggle-theme-provider';
import './globals.css';
import 'reactflow/dist/style.css';
import { store } from "@/store/index"
import { projectId } from "../config"


const { chains, publicClient } = configureChains(
  [mainnet, polygon, optimism, arbitrum, base, zora, sepolia, goerli],
  [
    alchemyProvider({ apiKey: 'Pg7_v8x8SlXaP0ZsI90QrGFxOEEJBCtA' }),
    publicProvider()
  ]
);

const connectors = connectorsForWallets([
  {
    groupName: 'Recommended',
    wallets: [
      metaMaskWallet({ projectId, chains }), // Metamask
      ...(projectId ? [walletConnectWallet({ projectId, chains })] : []),
      ...(projectId ? [trustWallet({ projectId, chains })] : []),
    ],
  },
  {
    groupName: 'Other',
    wallets: [
      ...(projectId ? [rainbowWallet({ projectId, chains })] : []),
      ...(projectId ? [okxWallet({ projectId, chains })] : []),
      ...(projectId ? [ledgerWallet({ projectId, chains })] : []),

    ],
  },
]);

export const wagmiConfig = createConfig({
  autoConnect: true,
  connectors,
  publicClient,
});

const myFont = localFont({ src: '../../public/fonts/VCR_OSD_MONO_1.001.ttf' })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {

  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const handleLoad = () => {
      setIsLoading(false);
    };

    window.addEventListener('load', handleLoad);

    return () => {
      // Cleanup: Remove the event listener when the component unmounts
      window.removeEventListener('load', handleLoad);
    };
  }, []);

  const pathName = usePathname()

  console.log('-----------This is the react component---------', pathName);


  return (
    <html lang="en">

      <Head />

      <body>
        {
          isLoading ?
            <Loading />
            :
            <WagmiConfig config={wagmiConfig}>
              <RainbowKitProvider chains={chains} coolMode theme={darkTheme()}>
                <Providers>
                  <Provider store={store}>
                    <ThemeProvider>
                      <ColorProvider>
                        {/* <main className={myFont.className}> */}
                        <ConfigProvider
                            theme={{
                              token: {
                                colorBgElevated: "#111838",
                                colorText:"white"

                              },
                              
                              components: {
                                Input: {
                                  colorPrimary: '#eb2f96',
                                  colorBgElevated:'#eeeeee',
                                  colorBgContainer:'#eeeeee',
                                  colorBgBase:'#eeeeee',
                                  colorBgLayout:'#eeeeee',
                                },
                                Button:{
                                  colorText:'#111838'
                                }
                              },
                              
                              
                            }}
                          >
                        <NavigationBar />
                        {children}
                        </ConfigProvider>
                      </ColorProvider>
                      {/* </main> */}
                      {/* <Footer /> */}
                    </ThemeProvider>
                  </Provider>
                </Providers>
                <ToastContainer />
              </RainbowKitProvider>
            </WagmiConfig>
        }

      </body>
    </html>
  );
}
