#!/bin/bash

# Script to run a local testnet

set -e

echo "Starting local Substrate testnet..."

# Function to cleanup on exit
cleanup() {
    echo "\nShutting down testnet..."
    docker-compose down
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found! Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "docker-compose not found! Please install docker-compose first."
    exit 1
fi

# Start the testnet
echo "Starting nodes with docker-compose..."
docker-compose up -d

# Wait for nodes to start
echo "Waiting for nodes to start..."
sleep 10

# Show logs
echo "\nTestnet is running!"
echo "Alice node RPC: http://localhost:9933"
echo "Alice node WebSocket: ws://localhost:9944"
echo "Bob node RPC: http://localhost:9934"
echo "Bob node WebSocket: ws://localhost:9945"
echo "\nView logs with: docker-compose logs -f"
echo "Stop testnet with: docker-compose down"
echo "\nPress Ctrl+C to stop the testnet"

# Keep script running and show logs
docker-compose logs -f