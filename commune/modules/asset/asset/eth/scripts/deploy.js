const hre = require("hardhat");

async function main() {
  console.log("Deploying StableCoinVault...");
  
  const StableCoinVault = await hre.ethers.getContractFactory("StableCoinVault");
  const vault = await StableCoinVault.deploy();
  
  await vault.deployed();
  
  console.log("StableCoinVault deployed to:", vault.address);
  
  // Get accepted tokens
  const acceptedTokens = await vault.getAcceptedTokens();
  console.log("Default accepted tokens:", acceptedTokens);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });