const hre = require('hardhat');

async function main() {
  const OpenOmad = await hre.ethers.getContractFactory('OpenOmad');
  const openomad = await OpenOmad.deploy();
  await openomad.waitForDeployment();
  const address = await openomad.getAddress();
  console.log('OpenOmad deployed to:', address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});