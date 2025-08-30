#!/bin/bash

# Helper script to set up the Uniswap Trader MCP environment

echo "Setting up Uniswap Trader MCP environment..."

# Check if .env file exists
if [ ! -f .env ]; then
  echo "Creating .env file template..."
  cat > .env << EOL
# Required environment variables
INFURA_KEY=your_infura_key_here
WALLET_PRIVATE_KEY=your_wallet_private_key_here

# Optional configuration
PORT=3000
EOL
  echo "⚠️  Please edit .env with your actual API keys before running the server."
fi

# Install dependencies
echo "Installing dependencies..."
npm install

echo "Setup complete! Run './run.sh' or 'make start' to start the server."
