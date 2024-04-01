# Flipper2 Contract

This contract, written in the Rust language using `ink!`, is a simple smart contract that stores a single boolean value. This contract provides methods to instantiate the contract with a specific boolean value, flip the value, and get the current state of the boolean.

## Features

- Initialize the `bool` value at the time of contract creation.
- Flip the `bool` value.
- Get the current value of the `bool`.

## Getting Started

### Prerequisites

- Rust programming language with Cargo package manager.
- `Ink!` smart contract language with its prerequisites.

### Functions

- `new(init_value: bool)`: Initializes the `bool` value. This function is called when the contract is created.
- `default()`: Initializes the `bool` value to `false`.
- `flip()`: Flips the status of the `bool` value. If the value is `false`, it is set to `true` and vice versa.
- `get()`: Returns the current `bool` value.

## Tests

The following test properly it's checked when the contract is compiled:

- `default_works()`: Tests that the `default()` function correctly initializes the `bool` value to `false`.
- `it_works()`: Test that verifies that the `bool` value is correctly initialized, flipped, and retrieved.
