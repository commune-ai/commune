// Script for deploying the CollectiveLoan contract

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying CollectiveLoan contract with the account:", deployer.address);

  // Parameters for contract deployment
  const minContribution = ethers.utils.parseEther("0.1"); // 0.1 ETH minimum contribution
  const votingThreshold = 5000; // 50% voting threshold (in basis points)

  // Deploy the contract
  const CollectiveLoan = await ethers.getContractFactory("CollectiveLoan");
  const collectiveLoan = await CollectiveLoan.deploy(minContribution, votingThreshold);
  await collectiveLoan.deployed();

  console.log("CollectiveLoan contract deployed to:", collectiveLoan.address);
  console.log("Minimum contribution:", ethers.utils.formatEther(minContribution), "ETH");
  console.log("Voting threshold:", votingThreshold / 100, "%");

  return collectiveLoan;
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
