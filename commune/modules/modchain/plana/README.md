# Plan A: Progressive Vesting Curve

## Overview

This proposal introduces a novel vesting mechanism where the vesting rate dynamically adjusts based on token sales, creating a self-regulating system that protects long-term holders while allowing for gradual liquidity.

## Core Mechanics

### Initial State
- **Total Supply**: 100 tokens
- **Initial Vesting Rate**: 0% (fully locked)
- **Target Vesting Rate**: 100% (fully vested)

### Vesting Formula

The vesting rate increases multiplicatively with each token sold:

```
Vesting Rate = (Tokens Sold / Total Supply) × 100%
```

### Key Features

1. **Progressive Unlocking**: As participants sell tokens, the overall vesting rate increases for all holders
2. **Self-Balancing**: Early sellers unlock liquidity for patient holders
3. **Full Vesting at Completion**: When all 100 tokens have been sold, vesting reaches 100%

## Example Scenarios

| Tokens Sold | Vesting Rate | Available for Sale |
|-------------|--------------|--------------------|
| 0           | 0%           | 0 tokens           |
| 10          | 10%          | 10 tokens          |
| 25          | 25%          | 25 tokens          |
| 50          | 50%          | 50 tokens          |
| 75          | 75%          | 75 tokens          |
| 100         | 100%         | All tokens         |

## Benefits

1. **Prevents Dump**: Initial lock prevents immediate selling pressure
2. **Rewards Patience**: Long-term holders benefit from increased vesting as others exit
3. **Natural Price Discovery**: Market forces determine the pace of unlocking
4. **Simplicity**: Easy to understand and implement

## Implementation Considerations

- Smart contract should track cumulative sales
- Vesting rate updates in real-time with each sale
- Consider minimum sale amounts to prevent gaming
- Optional: Add time-based minimum vesting period

## Conclusion

This vesting curve creates an elegant solution where individual actions benefit the collective, encouraging thoughtful participation while ensuring eventual full liquidity.