#/bin/bash

# install node
export NODE_VERSION=16.17.1
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
export NVM_DIR=/root/.nvm
. "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
. "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
. "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
export PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"

# Install node packages
npm i -g pm2
# npm install --save-dev hardhat
# npm install --save-dev @nomicfoundation/hardhat-toolbox
# npx hardhat
# npm install @openzeppelin/contracts