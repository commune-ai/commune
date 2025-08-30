const express = require('express');
const bodyParser = require('body-parser');
const { McpServer } = require('@modelcontextprotocol/sdk');
const { AlphaRouter } = require('@uniswap/smart-order-router');
const { ethers } = require('ethers');
const { getPrice, executeSwap, suggestSwap, createToken } = require('./index');
const { CHAIN_CONFIGS } = require('./chainConfigs');

// Check for required environment variables
if (!process.env.INFURA_KEY) {
  console.error('Error: INFURA_KEY environment variable is required');
  process.exit(1);
}

if (!process.env.WALLET_PRIVATE_KEY) {
  console.error('Error: WALLET_PRIVATE_KEY environment variable is required');
  process.exit(1);
}

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(bodyParser.json());

// Create providers and wallets for each chain
const providers = {};
const wallets = {};
const routers = {};

Object.entries(CHAIN_CONFIGS).forEach(([chainId, config]) => {
  const provider = new ethers.providers.JsonRpcProvider(config.rpcUrl);
  providers[chainId] = provider;
  
  const wallet = new ethers.Wallet(process.env.WALLET_PRIVATE_KEY, provider);
  wallets[chainId] = wallet;
  
  const router = new AlphaRouter({
    chainId: parseInt(chainId),
    provider,
  });
  routers[chainId] = router;
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok' });
});

// Get supported chains endpoint
app.get('/chains', (req, res) => {
  const chains = Object.entries(CHAIN_CONFIGS).map(([chainId, config]) => ({
    chainId: parseInt(chainId),
    name: config.name
  }));
  res.json(chains);
});

// Price quote endpoint
app.post('/quote', async (req, res) => {
  try {
    const { chainId, tokenIn, tokenOut, amount, exactOut } = req.body;
    
    if (!chainId || !tokenIn || !tokenOut || !amount) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }
    
    const provider = providers[chainId];
    if (!provider) {
      return res.status(400).json({ error: `Unsupported chain ID: ${chainId}` });
    }
    
    const result = await getPrice({
      chainId: parseInt(chainId),
      tokenIn,
      tokenOut,
      amount,
      exactOut: !!exactOut,
      provider,
      router: routers[chainId]
    });
    
    res.json(result);
  } catch (error) {
    console.error('Error getting price quote:', error);
    res.status(500).json({ error: error.message });
  }
});

// Execute swap endpoint
app.post('/swap', async (req, res) => {
  try {
    const { chainId, tokenIn, tokenOut, amount, exactOut, slippageTolerance } = req.body;
    
    if (!chainId || !tokenIn || !tokenOut || !amount) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }
    
    const provider = providers[chainId];
    const wallet = wallets[chainId];
    const router = routers[chainId];
    
    if (!provider || !wallet || !router) {
      return res.status(400).json({ error: `Unsupported chain ID: ${chainId}` });
    }
    
    const result = await executeSwap({
      chainId: parseInt(chainId),
      tokenIn,
      tokenOut,
      amount,
      exactOut: !!exactOut,
      slippageTolerance: slippageTolerance || 0.5,
      provider,
      wallet,
      router
    });
    
    res.json(result);
  } catch (error) {
    console.error('Error executing swap:', error);
    res.status(500).json({ error: error.message });
  }
});

// Suggest swap endpoint
app.post('/suggest', async (req, res) => {
  try {
    const { chainId, amount, token, tradeType } = req.body;
    
    if (!chainId || !amount || !token || !tradeType) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }
    
    // This is a placeholder - we need to implement the suggestion logic
    // based on the suggestSwap function from the MCP
    const suggestion = await suggestSwap({
      chainId: parseInt(chainId),
      amount,
      token,
      tradeType
    });
    
    res.json(suggestion);
  } catch (error) {
    console.error('Error suggesting swap:', error);
    res.status(500).json({ error: error.message });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Uniswap Trader API server running on port ${PORT}`);
  console.log(`Supported chains: ${Object.values(CHAIN_CONFIGS).map(c => c.name).join(', ')}`);
});
