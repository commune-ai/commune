require('@nomicfoundation/hardhat-toolbox');
require('dotenv').config();

module.exports = {
  solidity: '0.8.20',
  networks: {
    base: {
      url: process.env.BASE_RPC_URL || 'https://mainnet.base.org',
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 8453
    },
    baseGoerli: {
      url: process.env.BASE_GOERLI_RPC_URL || 'https://goerli.base.org',
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 84531
    }
  },
  etherscan: {
    apiKey: {
      base: process.env.BASESCAN_API_KEY || ''
    }
  }
};