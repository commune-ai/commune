# Token Generation Event (TGE) - StakeTime Vesting Schedule


![TGE Plot](./plots.png)

## Overview
This document outlines the vesting schedule using StakeTime with a linear vesting curve and exponential multiplier curve.

## StakeTime Configuration

### Anchor Points
- **Anchor 0**: Day 0 (TGE) - Multiplier: 0x
- **Anchor 1**: Year 1 (365 days) - Multiplier: 1x
- **Anchor 2**: Year 2 (730 days) - Multiplier: 4x

### Vesting Schedule
- **Type**: Linear vesting over 2 years
- **Start**: Day 0 (TGE)
- **End**: Day 730 (2 years)
- **Vesting Formula**: `vested_amount = total_allocation * (days_elapsed / 730)`

### Multiplier Curve
- **Type**: Exponential growth
- **Formula**: `multiplier = (4^(t/2))` where t is time in years
- **Key Points**:
  - t=0: multiplier = 0
  - t=1: multiplier = 1
  - t=2: multiplier = 4

## Implementation Details

### Linear Vesting Calculation
```
For any time t (in days):
vested_percentage = min(100%, (t / 730) * 100%)
```

### Exponential Multiplier Calculation
```
For time t (in years):
if t = 0: multiplier = 0
else: multiplier = 4^((t-1)/1) = 2^(2*(t-1))
```

## Vesting Timeline

| Time Period | Days | Vested % | Multiplier |
|------------|------|----------|------------|
| TGE (Day 0) | 0 | 0% | 0x |
| Month 3 | 90 | 12.3% | ~0.25x |
| Month 6 | 180 | 24.7% | ~0.5x |
| Year 1 | 365 | 50% | 1x |
| Month 18 | 540 | 74% | ~2x |
| Year 2 | 730 | 100% | 4x |

## StakeTime Benefits

1. **Early Stakers**: While vesting is linear, early stakers who hold through the full period benefit from the exponential multiplier growth
2. **Alignment**: The exponential multiplier incentivizes long-term holding beyond just the vesting period
3. **Flexibility**: Linear vesting ensures predictable token release while exponential multipliers reward patience

## Technical Parameters

```json
{
  "vesting": {
    "type": "linear",
    "duration_days": 730,
    "cliff_days": 0,
    "start_date": "TGE"
  },
  "staketime": {
    "anchors": [
      {"day": 0, "multiplier": 0},
      {"day": 365, "multiplier": 1},
      {"day": 730, "multiplier": 4}
    ],
    "multiplier_curve": "exponential",
    "vesting_curve": "linear"
  }
}
```

## Summary
This TGE structure uses StakeTime to create a balanced incentive system:
- **Linear vesting** ensures steady, predictable token distribution over 2 years
- **Exponential multiplier** (0x → 1x → 4x) strongly rewards long-term holders
- The combination encourages both liquidity (through vesting) and holding (through multipliers)

