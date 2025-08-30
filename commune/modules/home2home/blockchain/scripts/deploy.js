 # start of file
const fs = require('fs');
const path = require('path');
const hre = require("hardhat");

async function main() {
  console.log("Starting deployment of Home2Home contracts...");

  // Get the contract factories
  const PropertyToken = await hre.ethers.getContractFactory("PropertyToken");
  const RentToOwnAgreement = await hre.ethers.getContractFactory("RentToOwnAgreement");
  const Home2HomeRegistry = await hre.ethers.getContractFactory("Home2HomeRegistry");

  // Deploy the registry first
  const registry = await Home2HomeRegistry.deploy();
  await registry.deployed();
  console.log("Home2HomeRegistry deployed to:", registry.address);

  // Deploy a sample property token
  const propertyDetails = {
    squareFeet: 2000,
    bedrooms: 3,
    bathrooms: 2,
    yearBuilt: 2010,
    propertyType: "single-family"
  };

  const [deployer] = await hre.ethers.getSigners();
  
  const propertyToken = await PropertyToken.deploy(
    "123 Main St, Anytown, USA",
    hre.ethers.utils.parseEther("350000"), // $350,000 property
    deployer.address,
    propertyDetails
  );
  await propertyToken.deployed();
  console.log("Sample PropertyToken deployed to:", propertyToken.address);

  // Register the property in the registry
  await registry.registerProperty(
    propertyToken.address,
    "123 Main St, Anytown, USA",
    hre.ethers.utils.parseEther("350000")
  );
  console.log("Property registered in the registry");

  // Write the deployment info to a file that can be accessed by the frontend
  const deploymentInfo = {
    registryAddress: registry.address,
    samplePropertyTokenAddress: propertyToken.address,
    network: hre.network.name,
    chainId: hre.network.config.chainId || 1337,
    deployer: deployer.address
  };

  fs.writeFileSync(
    path.join(__dirname, '../deployment.json'),
    JSON.stringify(deploymentInfo, null, 2)
  );

  // Create a copy for the frontend to access
  fs.writeFileSync(
    path.join(__dirname, '../../frontend/public/deployment.json'),
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log("Deployment information saved to deployment.json");
  console.log("Deployment complete!");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
