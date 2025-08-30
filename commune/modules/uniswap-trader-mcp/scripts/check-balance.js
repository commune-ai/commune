#!/usr/bin/env node

/**
 * Script to check wallet balance on supported chains
 * Usage: node scripts/check-balance.js [chainId]
 */

require('dotenv').config();
const { ethers } = require('ethers');
const { CHAIN_CONFIGS } = require('../chainConfigs');

// Check for required environment variables
if (!process.env.WALLET_PRIVATE_KEY) {
  console.error('Error: WALLET_PRIVATE_KEY environment variable is required');
  process.exit(1);
}

async function checkBalance(chainId) {
  const config = CHAIN_CONFIGS[chainId];
  if (!config) {
    console.error(`Chain ID ${chainId} not supported`);
    console.log('Supported chains:');
    Object.entries(CHAIN_CONFIGS).forEach(([id, config]) => {
      console.log(`  ${id} - ${config.name}`);
    });
    return;
  }

  console.log(`Checking balance on ${config.name} (Chain ID: ${chainId})`);
  
  const provider = new ethers.providers.JsonRpcProvider(config.rpcUrl);
  const wallet = new ethers.Wallet(process.env.WALLET_PRIVATE_KEY, provider);
  
  // Get native token balance
  const balance = await provider.getBalance(wallet.address);
  console.log(`Native token balance: ${ethers.utils.formatEther(balance)} ${config.nativeSymbol || 'ETH'}`);
  
  console.log(`Wallet address: ${wallet.address}`);
}

// Get chain ID from command line args or check all chains
const requestedChainId = process.argv[2];

if (requestedChainId) {
  checkBalance(requestedChainId);
} else {
  console.log('Checking balances on all supported chains...');
  Object.keys(CHAIN_CONFIGS).forEach(chainId => {
    checkBalance(chainId);
  });
}
