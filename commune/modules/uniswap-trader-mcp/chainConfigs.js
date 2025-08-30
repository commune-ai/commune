// chainConfigs.js
require('dotenv').config();

const INFURA_KEY = process.env.INFURA_KEY;
if (!INFURA_KEY) {
  throw new Error("INFURA_KEY environment variable is required");
}

const CHAIN_CONFIGS = {
  1: { // Ethereum Mainnet
    rpcUrl: `https://mainnet.infura.io/v3/${INFURA_KEY}`,
    swapRouter: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    poolFactory: "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    weth: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    name: "Ethereum"
  },
  10: { // Optimism
    rpcUrl: `https://optimism-mainnet.infura.io/v3/${INFURA_KEY}`,
    swapRouter: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    poolFactory: "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    weth: "0x4200000000000000000000000000000000000006",
    name: "Optimism"
  },
  137: { // Polygon
    rpcUrl: `https://polygon-mainnet.infura.io/v3/${INFURA_KEY}`,
    swapRouter: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    poolFactory: "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    weth: "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
    name: "Polygon"
  },
  42161: { // Arbitrum One
    rpcUrl: `https://arbitrum-mainnet.infura.io/v3/${INFURA_KEY}`,
    swapRouter: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    poolFactory: "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    weth: "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    name: "Arbitrum One"
  },
  42220: { // Celo
    rpcUrl: `https://celo-mainnet.infura.io/v3/${INFURA_KEY}`,
    swapRouter: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    poolFactory: "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    weth: "0x471EcE3750Da237f93B8E339c536989b8978a438", // CELO (not WETH)
    name: "Celo"
  },
  56: { // BNB Chain
    rpcUrl: "https://bsc-dataseed.binance.org/",
    swapRouter: "0xB971eF87edeb8e677893eAf6B013cA363c0eB0B2",
    poolFactory: "0xdB1d10011AD0Ff90774D0C6Bb92e5C5c8b4461F7",
    weth: "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c", // WBNB
    name: "BNB Chain"
  },
  43114: { // Avalanche
    rpcUrl: `https://avalanche-mainnet.infura.io/v3/${INFURA_KEY}`,
    swapRouter: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    poolFactory: "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    weth: "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7", // WAVAX
    name: "Avalanche"
  },
  8453: { // Base
    rpcUrl: `https://base-mainnet.infura.io/v3/${INFURA_KEY}`,
    swapRouter: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    poolFactory: "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    weth: "0x4200000000000000000000000000000000000006",
    name: "Base"
  }
};

module.exports = CHAIN_CONFIGS;