# Fam Contract

This is a simple smart contract implemented in Rust using the `ink!` language.

The contract stores a single boolean value and provides functions to flip its status (from true to false or false to true) and to retrieve its current status.

## Features

- Initialize the boolean value at the time of contract creation
- Flip the boolean value
- Get the current status of the boolean value

## Getting Started

### Prerequisites

- Rust and Cargo package manager
- `Ink!` language and its prerequisites

### Functions

- `new(init_value: bool)`: Initializes the boolean value at the time of contract creation
- `default()`: Initializes the boolean value to `false`
- `flip()`: Flips the status of the boolean value
- `get()`: Returns the current status of the boolean value

## Tests

The contract also includes a test module which checks the basic functionality of the contract:

- `default_works()`: Tests if the default constructor initializes the boolean value to `false`
- `it_works()`: Tests if the value can be flipped correctly
