require("@nomicfoundation/hardhat-toolbox");

/** @type import('hardhat/config').HardhatUserConfig */
let contract_path = './commune/web3/evm/contract/data';
module.exports = {
  solidity: "0.8.9",
  paths: {
    sources: `${contract_path}/contracts`,
    artifacts: `${contract_path}/artifacts`
  }
};