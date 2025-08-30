

# 🎯 Distributed Token Pallet

> A sophisticated token distribution system using piecewise linear vesting curves

## 🌟 Overview

This pallet implements a flexible token distribution mechanism that:
- ✨ Distributes tokens based on time-weighted vesting
- 📈 Uses piecewise linear functions for vesting curves
- ⚡ Supports custom curve anchors for any monotonic distribution
- 🔗 Handles cross-chain bridging and redistribution

## 📊 Vesting Curve System

### Curve Anchors

The vesting system uses **curve anchors** - tuples that define points on a monotonically increasing curve:

```rust
// Curve anchor format: (time_months, multiplier)
type CurveAnchor = (u32, Perbill);

// Example: Linear vesting from 0% to 100% over 12 months
let anchors = vec![
    (0, Perbill::zero()),      // 0% at month 0
    (12, Perbill::one())       // 100% at month 12
];
```

### Piecewise Linear Function

The vesting multiplier at any time `t` is calculated by:
1. Finding the two anchors that bound time `t`
2. Linear interpolation between those anchors
3. Multiplying the base allocation by this multiplier

```
Multiplier
   ^
1.0|                    .-'
   |                 .-'
0.5|              .-'
   |           .-'
   |        .-'
0.0|-----.-'
   +----+----+----+----+
   0    3    6    9   12  Time (months)
```

## 🚀 How It Works

### 1. Initial Setup
```rust
// Configure the vesting curve
DistributionPallet::set_curve_anchors(
    origin,
    vec![
        (0, Perbill::from_percent(0)),
        (6, Perbill::from_percent(50)),
        (12, Perbill::from_percent(100))
    ]
);

// Set total supply for distribution
DistributionPallet::set_total_supply(origin, 1_000_000);
```

### 2. Bridge Period
During the bridge period, users can claim their allocations:
```rust
DistributionPallet::bridge_tokens(origin, amount);
```

### 3. Vesting Calculation
After the bridge period ends, the pallet:
1. Calculates each user's proportion of total bridged tokens
2. Applies the vesting curve based on elapsed time
3. Distributes tokens according to the vested amount

### 4. Token Claims
```rust
// Check vested amount at current time
let vested = DistributionPallet::calculate_vested_amount(account);

// Claim vested tokens
DistributionPallet::claim_vested(origin);
```

## 🎨 Example Curves

### Linear Vesting (Default)
```rust
vec![
    (0, Perbill::zero()),
    (12, Perbill::one())
]
```

### Cliff Vesting
```rust
vec![
    (0, Perbill::zero()),
    (6, Perbill::zero()),      // Cliff at 6 months
    (6, Perbill::from_percent(25)),
    (12, Perbill::one())
]
```

### Accelerated Vesting
```rust
vec![
    (0, Perbill::zero()),
    (3, Perbill::from_percent(50)),  // 50% in first 3 months
    (12, Perbill::one())
]
```

## 🔧 Storage

- `CurveAnchors`: Stores the vesting curve definition
- `TotalSupply`: Total tokens available for distribution
- `BridgedAmounts`: Tracks bridged tokens per account
- `VestingSchedules`: Individual vesting schedules
- `ClaimedAmounts`: Tracks claimed tokens

## 📝 Events

- `CurveAnchorsSet`: New vesting curve configured
- `TokensBridged`: User bridged tokens
- `VestingScheduleCreated`: New vesting schedule started
- `TokensClaimed`: Vested tokens claimed

## 🎯 Benefits

1. **Flexible Distribution**: Any monotonic curve possible
2. **Fair Allocation**: Time-weighted vesting ensures fairness
3. **Cross-chain Ready**: Built for multi-chain ecosystems
4. **Gas Efficient**: Optimized calculations and storage

---

*Built with ❤️ for decentralized token distribution*
