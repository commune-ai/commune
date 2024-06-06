export const isTestnet = true;

export const projectId = 'a0dd23157746b39315f34b62eb614eae';

export const externalLinks = {
    whitepaper: 'https://ai-secure.github.io/DMLW2022/assets/papers/7.pdf',
    exchangeApi: 'https://api.comswap.io/orders/public/marketinfo',
};

export const acceptableChains = [
    'ethereum',
    'polygon',
    'sepolia',
    'polygon mumbai'
];

export const tokenAddresses = {
    'ethereum': {
        usdt: '0xdAC17F958D2ee523a2206206994597C13D831ec7',
        usdc: '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
    },
    'sepolia': {
        usdt: '0x7169D38820dfd117C3FA1f22a697dBA58d90BA06',
        usdc: '0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238',
    },
    'polygon': {
        usdt: '0xc2132d05d31c914a87c6611c10748aeb04b58e8f',
        usdc: '0x2791bca1f2de4661ed88a30c99a7a9449aa84174',
    },
    'polygon mumbai': {
        usdt: '0x1fdE0eCc619726f4cD597887C9F3b4c8740e19e2',
        usdc: '0x9999f7fea5938fd3b1e26a12c3f2fb024e194f97',
    }
};
