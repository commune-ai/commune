#!/bin/bash
set -e

# Function to handle container shutdown
cleanup() {
    echo "Container is shutting down..."
    exit 0
}

# Setup signal trapping
trap cleanup SIGTERM SIGINT

# Start SSH server
sudo service ssh start

# Print container info
echo "Container ready for SSH connections"
echo "Workspace directory: /workspace"
echo "User: devuser"

# Keep container running
while true; do
    sleep 1
done