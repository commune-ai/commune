 # start of file
#!/bin/bash
set -e

# Build and run the Docker container
docker-compose up -d

echo "Node is running in Docker container"
echo "WebSocket endpoint: ws://localhost:9944"
echo "RPC endpoint: http://localhost:9933"
echo "UI available at: http://localhost:8080"
