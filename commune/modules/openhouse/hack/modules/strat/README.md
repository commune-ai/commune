# USD Contribution Tracker Contract

This smart contract tracks and determines the USD value contributed to a liquidity pool by different users.

## Features

- Track contributions in various tokens and convert to USD value
- Support for multiple tokens with Chainlink price feeds
- Calculate user contribution percentages relative to the total pool
- Query historical contribution data

## Contract Overview

`USDContributionTracker.sol` is the main contract that implements the following functionality:

- Recording token contributions and their USD value
- Tracking individual user contributions
- Tracking total pool value in USD
- Calculating contribution percentages

## Usage

### Adding Supported Tokens

```solidity
function addSupportedToken(address token, address priceFeed) external onlyOwner
```

### Recording Contributions

```solidity
function recordContribution(address user, address token, uint256 amount) external onlyOwner
```

### Querying Contribution Data

```solidity
function getUserUsdContribution(address user) external view returns (uint256)
function getPoolUsdValue() external view returns (uint256)
function getUserTokenContribution(address user, address token) external view returns (uint256)
function getUserContributionPercentage(address user) external view returns (uint256)
```

## Dependencies

- OpenZeppelin Contracts
- Chainlink Price Feeds

## Security Considerations

- Only the contract owner can add new tokens and record contributions
- Price feeds must be reliable and up-to-date
- Decimal handling is important for accurate USD calculations
