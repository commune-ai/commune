# Collective Loan Smart Contract

A decentralized solution for managing collective loans on the blockchain. This smart contract enables multiple participants to pool funds, collectively decide on loan disbursements, and share in the profits from loan repayments.

## Features

- **Collective Fund Pooling**: Members can join by contributing funds to a shared pool
- **Democratic Loan Approval**: Loan requests are approved through a voting mechanism
- **Proportional Profit Sharing**: Profits are distributed based on each member's contribution
- **Transparent Loan Management**: All loan requests, approvals, and repayments are tracked on-chain
- **Flexible Interest Rates**: Each loan can have a custom interest rate

## Smart Contract Functions

### Member Management

- `joinCollective()`: Join the collective by contributing funds
- `contributeFunds()`: Add more funds to the collective pool
- `withdrawFunds()`: Withdraw your share of the available funds
- `getMemberDetails()`: View details about a specific member

### Loan Management

- `requestLoan()`: Request a loan from the collective pool
- `voteOnLoan()`: Vote to approve or reject a loan request
- `repayLoan()`: Repay an active loan (partially or fully)
- `getLoanDetails()`: View details about a specific loan

### Administrative Functions

- `distributeProfits()`: Distribute profits to all members (admin only)

### View Functions

- `calculateTotalDebt()`: Calculate the total debt for a loan
- `calculateRemainingDebt()`: Calculate the remaining debt for a loan
- `calculateWithdrawableAmount()`: Calculate how much a member can withdraw
- `calculateTotalProfits()`: Calculate total profits in the pool

## Usage Example

1. Deploy the contract with minimum contribution and voting threshold parameters
2. Members join by sending ETH to the `joinCollective()` function
3. A member requests a loan by calling `requestLoan()`
4. Other members vote on the loan request
5. If approved, funds are automatically transferred to the borrower
6. Borrower repays the loan with interest
7. Profits can be distributed to all members proportionally

## Implementation Details

The contract uses a share-based system where each member's voting power and profit share are proportional to their contribution to the pool. Loans are approved when they receive votes exceeding the voting threshold (set at contract deployment).

## Security Considerations

- The contract includes access controls to ensure only authorized users can perform certain actions
- Funds are tracked carefully to prevent unauthorized withdrawals
- Loan execution only happens after sufficient votes are received

## Future Improvements

- Add collateral requirements for loans
- Implement a time-lock for large withdrawals
- Add support for different tokens/currencies
- Implement a reputation system for borrowers

## License

This smart contract is licensed under MIT.
