# Uniswap Trader MCP

A Model Context Protocol (MCP) integration for Uniswap trading. This service enables programmatic trading on Uniswap through a simple API interface.

## Requirements

- Node.js (v14 or higher)
- npm or yarn
- Infura API key
- Ethereum wallet private key

## Setup

1. Clone this repository
2. Create a `.env` file with your credentials (see below)
3. Install dependencies: `make install` or `npm install`
4. Start the server: `make start` or `./run.sh`

## Environment Variables

Create a `.env` file with the following variables:

```
# Required environment variables
INFURA_KEY=your_infura_key_here
WALLET_PRIVATE_KEY=your_wallet_private_key_here

# Optional configuration
PORT=3000
```

## Available Commands

This project includes a Makefile with the following commands:

- `make install` - Install dependencies
- `make start` - Start the MCP server
- `make build-docker` - Build Docker image
- `make run-docker` - Run Docker container
- `make clean` - Remove node_modules and logs

## Docker Support

You can build and run this service using Docker:

```bash
# Build the Docker image
make build-docker

# Run the Docker container
make run-docker
```

## License

MIT
