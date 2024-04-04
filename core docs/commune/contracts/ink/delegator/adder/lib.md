# Adder Smart Contract

The Adder smart contract provides an illustration of how a smart contract can interact with an external contract and manipulate its storage values on-chain. 

This contract increases the value stored in an instance of the Accumulator contract, which it references through an `AccumulatorRef`. 

## Key Features

- The contract is initialized with a reference to an Accumulator contract.
- `inc()` function: It increments the value held by the referenced Accumulator contract by a specified amount.

## Usage

Ensure that you have Rust and its nightly version installed on your machine. 

1. Compile the contract: `cargo +nightly build --release`
2. Run tests (if available): `cargo +nightly test`
3. Deploy the contract to your local Substrate development node. Refer to the [Substrate Contracts pallet](https://substrate.dev/rustdocs/latest/pallet_contracts/index.html) for more instructions on deploying contracts.

## Testing

In order to test the contract, unit tests should be written. Ideally, a test should create an instance of the Adder contract with a linked Accumulator contract and verify if the value in the Accumulator contract increments as expected when `inc()` is called.

_Note: Make sure to run the tests with the `nightly` version of Rust, using the command `cargo +nightly test`._

## Understanding the Code

This contract is written in Rust. To learn more about Rust, refer to the [Rust Book](https://doc.rust-lang.org/book).

Details of the code:

- `Adder` is the main structure that holds an instance of an `AccumulatorRef`.
- `new()` function: Serves as a constructor for the contract. It initializes `Adder` with an `AccumulatorRef`, which it uses to manipulate the value within the Accumulator contract.
- `inc()` function: Increases the value in the linked `Accumulator` by a specified amount.
