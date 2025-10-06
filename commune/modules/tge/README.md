# MOD Token Generation Event (TGE)

## Overview

MOD token distribution follows a three-phase approach designed to reward early holders, incentivize long-term commitment, and align with protocol usage.

---

## Phase 1: Initial Distribution (10/15/2025 → 01/01/2026)

**Total Supply: 1,000,000 MOD**

### Claimable Airdrop

- Initial distribution of **1 million tokens** on mainnet launch
- COMAI holders must prove holdings on Ethereum, Solana, or Mainnet
  - ⚠️ **Bridge off MEXC before October 15th, 2025**
- Holders manually confirm creation of $MOD from COMAI
- Distribution is **proportional** with **no dilution** for certain holders

---

## Phase 2: Stake-Time Rewards (12/01/2025 → 01/01/2026)

**Total Supply: 3,200,000 MOD**

### Unified Lock & Multiplier Curve System

**IMPORTANT:** The lock curve and multiplier curve are **identical** - they represent the same quadratic relationship.

Wallet holders choose a lock period (up to 2 years) to earn multiplied rewards:

| Lock Period | Multiplier | Example (1000 MOD) |
|-------------|------------|--------------------||
| 0 days (opt-out) | **0x** | 0 MOD |
| 1 year (365 days) | **1x** | 1,000 MOD |
| 2 years (730 days) | **4x** | 4,000 MOD |

**Unified Formula:** `Lock Curve = Multiplier Curve = (lock_days / 365)²`

![Lock Multiplier Curve](public/lock_multiplier.png)

### Vesting Curve

- **Linear vesting** over the chosen lock period
- Rewards distributed proportionally as time progresses
- Longer stakes = more total tokens + extended vesting

**Example:**
- Lock 1000 MOD for 2 years → Earn 4000 staketime tokens
- After 1 year: 2000 tokens vested
- After 2 years: 4000 tokens fully vested

![Vesting Schedule](public/vesting_curve.png)

### Tokenized Vesting

- **Vesting positions are tradeable** on secondary markets
- Fractionalize vesting into **staketime tokens**
- AMM module based on **Uniswap v4** for modular swaps
- Pair MODCHAIN with staketime tokens for liquidity

---

## Phase 3: Proof of Contribution (12/01/2025 → ∞)

### Minting Mechanism

New tokens generated based on **nominal USD value** contributed to the protocol:

#### Contribution Methods
1. **Locking liquidity** into the protocol
2. **Using verified modules** via telemetry (pay in USDC/USDT on Solana or Base)

#### Minting Rules

- **Base Rate:** 1 USD = 1 MOD minted
- **Dynamic Pricing:**
  - MOD < $1.00 → **10% surcharge** (encourages DEX buying)
  - MOD > $1.00 → **10% discount** (encourages protocol minting)

#### Supply Cap

- **Daily mint limit** follows Bitcoin's inflation curve (scaled to 42M max supply)
- **Max daily mints:** 14,400 MOD (2 × 7200 blocks)
- Halving schedule mirrors Bitcoin's scarcity model

---

## Technical Implementation

### Vesting & Multiplier Curves

See `tge/curves.py` for mathematical implementation:

```python
from tge.curves import VestingCurves

# Calculate multiplier (same as lock curve)
multiplier = VestingCurves.lock_multiplier(730)  # 4.0x for 2 years

# Calculate vested tokens
vested = VestingCurves.vested_tokens(
    days_elapsed=365,
    lock_period_days=730,
    base_tokens=1000
)  # 2000 tokens after 1 year of 2-year lock

# Get full distribution details
dist = VestingCurves.phase2_distribution(
    initial_allocation=1000,
    lock_days=730
)
```

### Visualization

Generate PNG images:

```python
import commune as c
tge = c.module('tge')
tge.save_images()  # Creates public/*.png files
```

**Available Charts:**
- Token distribution pie chart
- Phase 1 timeline
- Claim progress gauge
- **Unified lock/multiplier curve** (0x → 4x, same curve)
- **Vesting schedule** (linear distribution)

---

## Key Dates

| Date | Event |
|------|-------|
| **10/15/2025** | Phase 1 begins (airdrop claims open) |
| **12/01/2025** | Phase 2 begins (stake-time rewards) |
| **12/01/2025** | Phase 3 begins (proof of contribution) |
| **01/01/2026** | Phase 1 & 2 end |

---

## Summary

- **Phase 1:** Fair airdrop to COMAI holders (1M MOD)
- **Phase 2:** Quadratic rewards for long-term lockers (3.2M MOD) - **Lock Curve = Multiplier Curve**
- **Phase 3:** Sustainable minting tied to protocol usage (∞)

**Total Initial Supply:** 4.2M MOD  
**Max Supply:** 42M MOD (Bitcoin-style inflation)

---

## Resources

- **Code:** `tge/tge.py` (visualization) & `tge/curves.py` (math)
- **AMM:** Uniswap v4-based module for staketime token trading
- **Telemetry:** Verified module usage tracking for Phase 3 minting