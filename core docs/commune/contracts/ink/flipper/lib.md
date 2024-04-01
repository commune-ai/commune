# Flipper Contract

This is a straightforward smart contract written in Rust using the `ink!` language.

The contract keeps a single boolean value in its storage and offers methods to change its status (from true to false or vice versa) and to retrieve its current value.

## Features

- Start the contract with a specified boolean value
- Toggle the boolean value
- Return the boolean's present state

## Getting Started

### Prerequisites

- Rust and the Cargo Package Manager
- `Ink!` language and its prerequisites

### Functions

- `new(init_value: bool)`: This function initializes the boolean value at contract creation
- `default()`: This function initializes a `false` boolean value
- `flip()`: This function toggles the boolean value
- `get()`: This function returns the current boolean value

## Tests

The contract comes with a test module that tests the basic functionality:

- `default_works()`: This checks if the default constructor rightly initializes the boolean value as `false`
- `it_works()`: This tests the ability to initialize the boolean value, toggle it, and retrieve its state
