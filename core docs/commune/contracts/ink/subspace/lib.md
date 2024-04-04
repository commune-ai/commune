# Subspace Contract

The Subspace contract, written in Rust using `ink!`, manages the allocation and handling of a custom-made cryptocurrency, used for voting within a network. This custom coin represents stakes within a voting based network. Voting helps to allocate control levels given to nodes within the network. The contract holds a balances ledger of all coin allocations and dynamically adjusts coin minting rates based on votes cast by the network members.

## Features

- Initialize total supply of coins.
- Initialize founders' share and their initial mints.
- Keep track of votes done by users.
- Representation for users' to their last update.
- Mapping of users to their voting power.
- Mapping of users to the score each user has.
- Adjusts minting of coins based on voting results.

## Getting Started

### Prerequisites

- Rust programming language with Cargo package manager.
- `Ink!` smart contract language with its prerequisites.

### Functions

- `new(total_supply: u128, founders: Vec<AccountId>, founder_initial_mints: Vec<u64>)`: Constructs a new Subspace contract, initializing supply, balancing for founders, etc.
- `get()`: Returns the total supply of coins.

## Tests

- `it_works()`: Tests the initial supply of the coins based on the values given when the contract is created.
