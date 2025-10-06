# Stablecoin Yield Aggregator

A modular DeFi yield aggregator that optimizes stablecoin returns across multiple protocols while building a 21 million share pool from yield.

## Features

- **Multi-Strategy Yield Optimization**: Automatically distributes funds across Aave, Compound, and Curve
- **21M Share Pool**: 10% of all generated yield mints shares up to 21 million total
- **Modular Architecture**: Easy to add new yield strategies
- **Simple Web Interface**: User-friendly dashboard for deposits, withdrawals, and harvesting
- **ERC20 Yield Tokens**: Receive YAT tokens representing your share of the pool

## Smart Contracts

### Core Contract
- `StablecoinYieldAggregator.sol`: Main aggregator managing deposits, withdrawals, and yield distribution

### Strategy Contracts
- `AaveStrategy.sol`: Lends stablecoins on Aave V3
- `CompoundStrategy.sol`: Supplies to Compound V3
- `CurveStrategy.sol`: Provides liquidity to Curve stable pools

## Architecture

```
User Deposits USDC
       |
       v
Aggregator Contract
       |
       +---> 50% to Aave Strategy
       +---> 30% to Compound Strategy  
       +---> 20% to Curve Strategy
       |
       v
Yield Generated
       |
       +---> 90% to Users (compounds)
       +---> 10% to 21M Share Pool
```

## Deployment

1. Deploy the main aggregator contract
2. Deploy strategy contracts (Aave, Compound, Curve)
3. Add strategies to aggregator with allocations
4. Update contract addresses in `app/app.js`

## Web App

The app provides:
- Real-time TVL and yield statistics
- Deposit/withdraw interface
- Harvest yield from all strategies
- Pool share tracking and claiming

## Setup

1. Install dependencies:
```bash
npm install @openzeppelin/contracts
```

2. Deploy contracts:
```bash
npx hardhat run scripts/deploy.js --network mainnet
```

3. Update `app/app.js` with deployed addresses

4. Serve the web app:
```bash
cd app
python -m http.server 8000
```

## Security

- ReentrancyGuard on all state-changing functions
- Ownable for admin functions
- Strategy contracts isolated from main contract
- Approval-based fund transfers

## License

MIT