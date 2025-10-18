const express = require('express');
const cors = require('cors');
const { ethers } = require('ethers');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3001;

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', network: 'base' });
});

app.get('/api/contract-info', (req, res) => {
  res.json({
    address: process.env.CONTRACT_ADDRESS,
    network: 'base',
    chainId: 8453
  });
});

app.post('/api/mint', async (req, res) => {
  try {
    const { address, uri } = req.body;
    // Add minting logic here
    res.json({ success: true, message: 'Minting initiated' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Backend running on port ${PORT}`);
});