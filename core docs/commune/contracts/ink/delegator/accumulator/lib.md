# Accumulator Smart Contract

The Accumulator smart contract is a simple demonstration of how a smart contract can hold and manipulate a value on-chain. 

Specifically, it holds an `i32` value which can get incremented or decremented. 

## Key Features

- The contract is initialized with a starting `i32` value.
- `inc()` function: It can increment the stored value by a certain amount.
- `get()` function: It can return the stored value at any given point in time.

## Usage

Ensure that you have Rust and its nightly version installed on your machine.

1. Compile the contract: `cargo +nightly build --release`
2. Run tests: `cargo +nightly test`
3. Deploy the contract to your local Substrate development node. Please refer to the [Substrate Contracts pallet](https://substrate.dev/rustdocs/latest/pallet_contracts/index.html) for more details on deploying contracts.

## Testing

Unit tests are available in the contract. They check if the `inc()` function is working correctly by creating an Accumulator contract instance and verifying if the value increments as expected.

_Note: Please make sure to run the tests with the `nightly` version of Rust, using the command `cargo +nightly test`._

## Understanding the code

This smart contract is written in Rust. If you're unfamiliar with Rust, consider reading the [Rust Book](https://doc.rust-lang.org/book) for an introduction to the language.

- `Accumulator` is the main structure that holds `i32` value.
- `new()` function: It acts as a constructor for the contract and initializes `Accumulator` with an `i32` value provided at the time of instance creation.
- `inc()` function: It allows the value in `Accumulator` to be incremented by a specified amount.
- `get()` function: It retrieves the current value of `Accumulator`.
