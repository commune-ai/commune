const hre = require('hardhat');

async function main() {
  console.log('Deploying to Base network...');

  const OpenomadToken = await hre.ethers.getContractFactory('OpenomadToken');
  const token = await OpenomadToken.deploy();
  await token.waitForDeployment();
  console.log('OpenomadToken deployed to:', await token.getAddress());

  const OpenomadNFT = await hre.ethers.getContractFactory('OpenomadNFT');
  const nft = await OpenomadNFT.deploy('https://api.openomad.io/metadata/');
  await nft.waitForDeployment();
  console.log('OpenomadNFT deployed to:', await nft.getAddress());
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});