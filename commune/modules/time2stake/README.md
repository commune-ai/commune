 # start of file
# Time2Stake

A Substrate pallet for time-based staking mechanisms.

## Overview

Time2Stake is a custom Substrate pallet that implements a time-based staking system where rewards are calculated based on both the amount staked and the duration of the stake.

## Features

- Time-based staking rewards
- Customizable staking periods
- Early unstaking penalties
- Reward multipliers based on staking duration

## Installation

Add this pallet to your runtime's `Cargo.toml`:

```toml
[dependencies]
time2stake = { version = "0.1.0", default-features = false, git = "https://github.com/yourusername/time2stake" }
```

## Usage

1. Import the pallet in your runtime's `lib.rs`:

```rust
pub use time2stake;
```

2. Implement the pallet's configuration trait for your runtime:

```rust
impl time2stake::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type Currency = Balances;
    // Add other required configuration
}
```

3. Add the pallet to your `construct_runtime!` macro:

```rust
construct_runtime!(
    pub enum Runtime where
        Block = Block,
        NodeBlock = opaque::Block,
        UncheckedExtrinsic = UncheckedExtrinsic
    {
        // ... other pallets
        Time2Stake: time2stake,
    }
);
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
