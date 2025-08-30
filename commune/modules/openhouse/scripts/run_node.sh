#!/bin/bash

# Run a rollup node

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Parse command line arguments
NODE_ID="node1"
NETWORK="testnet"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --node-id)
        NODE_ID="$2"
        shift
        shift
        ;;
        --network)
        NETWORK="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Run the node
python rollup_node.py --node-id "$NODE_ID" --network "$NETWORK"
