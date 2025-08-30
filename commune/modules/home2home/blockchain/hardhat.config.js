 # start of file
require("@nomiclabs/hardhat-waffle");
require("dotenv").config();

module.exports = {
  solidity: {
    version: "0.8.17",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    localhost: {
      url: process.env.NETWORK_URL || "http://ganache:8545",
    },
    hardhat: {
      chainId: 1337
    }
  },
  paths: {
    sources: "../contracts",
    artifacts: "./artifacts",
    cache: "./cache",
    tests: "./test"
  }
};
