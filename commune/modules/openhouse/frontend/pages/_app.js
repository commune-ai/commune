 # start of file
import { ChakraProvider, extendTheme } from '@chakra-ui/react';
import { Web3Provider } from '../hooks/useWeb3';
import '../styles/globals.css';

// Extend the theme to include custom colors, fonts, etc
const theme = extendTheme({
  colors: {
    brand: {
      50: '#e6f7ff',
      100: '#b3e0ff',
      200: '#80caff',
      300: '#4db3ff',
      400: '#1a9dff',
      500: '#0080ff',
      600: '#0066cc',
      700: '#004d99',
      800: '#003366',
      900: '#001a33',
    },
  },
  fonts: {
    heading: 'Poppins, sans-serif',
    body: 'Inter, sans-serif',
  },
});

function MyApp({ Component, pageProps }) {
  return (
    <ChakraProvider theme={theme}>
      <Web3Provider>
        <Component {...pageProps} />
      </Web3Provider>
    </ChakraProvider>
  );
}

export default MyApp;
