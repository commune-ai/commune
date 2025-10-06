import { HardhatUserConfig } from 'hardhat/config';
import '@nomicfoundation/hardhat-toolbox';
import * as dotenv from 'dotenv';
dotenv.config();

const accounts = process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : undefined;

const config: HardhatUserConfig = {
  solidity: { version: '0.8.26', settings: { optimizer: { enabled: true, runs: 500 }, viaIR: true } },
  networks: {
    localhost: { url: 'http://127.0.0.1:8545' },
    external:  { url: process.env.EVM_RPC_HTTP || '', accounts }
  }
};
export default config;
