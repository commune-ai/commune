require("@nomiclabs/hardhat-waffle");

module.exports = {
  solidity: {
    version: "0.8.0",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    localhost: {
      url: process.env.RPC_URL || "http://localhost:8545",
      chainId: 1337
    },
    hardhat: {
      chainId: 1337
    }
  },
  paths: {
    sources: "./",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  }
};