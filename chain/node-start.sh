#!/bin/sh
set -e
wget https://github.com/agicommies/communex/raw/main/docs/bootnodes.txt
wget https://github.com/agicommies/subspace/raw/current/specs/main.json

NODE_DATA_DIR="${NODE_DATA_DIR:-/node-data}"

BOOTNODES="$(< "$NODE_DATA_DIR/bootnodes.txt" xargs echo)"

# shellcheck disable=SC2086
node-subspace \
    --base-path "$NODE_DATA_DIR" --chain "$NODE_DATA_DIR/specs/main.json" \
    --rpc-external --rpc-cors=all --port 30333 --rpc-port 9944 \
    --telemetry-url 'ws://telemetry.communeai.net:8001/submit 0' \
    --rpc-max-connections 2000 --bootnodes $BOOTNODES "$@"
