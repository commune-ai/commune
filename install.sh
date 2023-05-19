#!/bin/bash
./scripts/install_python_env.sh
sudo ./scripts/install_npm_env.sh
sudo npm install -g pm2
# ./scripts/install_rust_env.sh