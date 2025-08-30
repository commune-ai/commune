 # start of file
#!/bin/bash

echo "Deploying smart contracts to local network..."

# Enter the blockchain directory
cd blockchain

# Run the deployment script
npx hardhat run scripts/deploy.js --network localhost

echo "Smart contracts deployed successfully!"
