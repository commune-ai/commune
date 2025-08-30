# Plan B: Stake-Time Multiplier Vesting Curve

## Overview

This proposal introduces a vesting mechanism where 4.2 million tokens are distributed based on a stake-time multiplier curve. The longer participants lock their tokens, the higher their multiplier, creating incentives for long-term commitment.

## Core Mechanics

### Initial Parameters
- **Total Supply**: 4,200,000 tokens
- **Base Allocation**: Determined by initial stake amount
- **Multiplier Range**: 1x to 4x based on lock duration
- **Lock Periods**: 0 to 2 years

### Multiplier Formula

```
Multiplier = 1 + (3 × (Lock Days / 730))
```

Where:
- Lock Days = Number of days tokens are locked (0-730)
- 730 = Maximum lock period (2 years)
- Multiplier ranges from 1x (no lock) to 4x (2-year lock)

### Distribution Calculation

```
User Allocation = (User Stake × User Multiplier) / (Total Weighted Stakes) × 4,200,000
```

## Lock Period Examples

| Lock Duration | Multiplier | Tokens per 1000 Staked* |
|---------------|------------|------------------------|
| 0 days        | 1.0x       | 1,000                  |
| 6 months      | 1.75x      | 1,750                  |
| 1 year        | 2.5x       | 2,500                  |
| 18 months     | 3.25x      | 3,250                  |
| 2 years       | 4.0x       | 4,000                  |

*Assuming uniform participation

## Key Features

1. **Linear Multiplier Growth**: Predictable rewards that scale linearly with commitment
2. **Flexible Lock Periods**: Participants choose their own risk/reward balance
3. **Fair Distribution**: Tokens allocated proportionally based on stake × multiplier
4. **Anti-Whale Mechanism**: Large holders can't dominate without long-term commitment

## Implementation Details

### Smart Contract Functions
- `stake(amount, lockDays)`: Lock tokens and set vesting period
- `calculateMultiplier(lockDays)`: Returns multiplier for given lock period
- `claimTokens()`: Claim vested tokens after lock period
- `emergencyWithdraw()`: Withdraw with penalty before lock expires

### Vesting Schedule
- Tokens become claimable only after lock period expires
- No partial vesting during lock period
- Optional: Early withdrawal with 50% penalty

## Benefits

1. **Rewards Diamond Hands**: Higher rewards for longer commitments
2. **Reduces Sell Pressure**: Locked tokens can't be immediately dumped
3. **Price Stability**: Staggered unlock periods prevent mass selling
4. **Community Building**: Encourages long-term thinking and participation
5. **Predictable Supply**: Clear visibility on when tokens become liquid

## Risk Considerations

- **Liquidity Risk**: Participants can't access tokens during lock period
- **Opportunity Cost**: Locked tokens can't be used for other opportunities
- **Market Risk**: Token value may change during lock period

## Conclusion

Plan B creates a fair distribution mechanism that rewards long-term believers while ensuring gradual, predictable token releases. The stake-time multiplier curve aligns incentives between the protocol and its most committed supporters.