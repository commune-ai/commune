// Script for testing the CollectiveLoan contract

async function main() {
  // Get signers for testing
  const [admin, member1, member2, member3, borrower] = await ethers.getSigners();
  console.log("Testing with admin account:", admin.address);

  // Parameters for contract deployment
  const minContribution = ethers.utils.parseEther("0.1"); // 0.1 ETH minimum contribution
  const votingThreshold = 5000; // 50% voting threshold (in basis points)

  // Deploy the contract
  const CollectiveLoan = await ethers.getContractFactory("CollectiveLoan");
  const collectiveLoan = await CollectiveLoan.deploy(minContribution, votingThreshold);
  await collectiveLoan.deployed();
  console.log("CollectiveLoan contract deployed to:", collectiveLoan.address);

  // Test joining the collective
  console.log("\n--- Testing joining the collective ---");
  await collectiveLoan.connect(member1).joinCollective({ value: ethers.utils.parseEther("1.0") });
  await collectiveLoan.connect(member2).joinCollective({ value: ethers.utils.parseEther("2.0") });
  await collectiveLoan.connect(member3).joinCollective({ value: ethers.utils.parseEther("3.0") });
  console.log("Three members joined with 1, 2, and 3 ETH respectively");

  // Check pool funds
  const totalPoolFunds = await collectiveLoan.totalPoolFunds();
  console.log("Total pool funds:", ethers.utils.formatEther(totalPoolFunds), "ETH");

  // Test loan request
  console.log("\n--- Testing loan request ---");
  await collectiveLoan.connect(member1).requestLoan(
    ethers.utils.parseEther("2.0"), // 2 ETH loan
    30, // 30 days duration
    500, // 5% interest rate
    "Home renovation"
  );
  console.log("Loan requested by member1 for 2 ETH");

  // Test voting on loan
  console.log("\n--- Testing loan voting ---");
  await collectiveLoan.connect(member2).voteOnLoan(0, true);
  await collectiveLoan.connect(member3).voteOnLoan(0, true);
  console.log("Member2 and Member3 voted in favor of the loan");

  // Check loan details
  const loanDetails = await collectiveLoan.getLoanDetails(0);
  console.log("Loan executed:", loanDetails.executed);
  console.log("Votes in favor:", loanDetails.votesInFavor.toString());

  // Test loan repayment
  console.log("\n--- Testing loan repayment ---");
  const totalDebt = await collectiveLoan.calculateTotalDebt(0);
  console.log("Total debt to repay:", ethers.utils.formatEther(totalDebt), "ETH");
  
  // Repay loan
  await collectiveLoan.connect(member1).repayLoan(0, { value: totalDebt });
  console.log("Loan repaid by member1");

  // Check loan status after repayment
  const updatedLoanDetails = await collectiveLoan.getLoanDetails(0);
  console.log("Loan repaid status:", updatedLoanDetails.repaid);

  // Test profit distribution and withdrawal
  console.log("\n--- Testing profit distribution ---");
  await collectiveLoan.distributeProfits();
  console.log("Profits distributed");

  // Check withdrawable amount for member2
  const withdrawableAmount = await collectiveLoan.calculateWithdrawableAmount(member2.address);
  console.log("Member2 can withdraw:", ethers.utils.formatEther(withdrawableAmount), "ETH");

  console.log("\nAll tests completed successfully!");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
