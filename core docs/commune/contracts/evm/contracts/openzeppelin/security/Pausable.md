# Pausable Contract

The `Pausable` contract is an abstract contract module that allows derived contracts to implement an emergency stop mechanism. The pause can be triggered by an authorized account, usually a contract owner or admin.

## Features

- **Pause and Unpause Events**: When a pause or unpause action occurs, respective `Paused` or `Unpaused` events are emitted, with the account that triggered the action passed as a parameter.

- **Pause State**: The contract maintains a boolean `_paused` state which indicates if the contract is currently paused or not.

- **Pausing Functions**: Two functions `_pause` and `_unpause` change the state of the contract. These are marked as `internal virtual`, which means they can be overridden in derived contracts.

- **Pause State Requiring Modifiers**: Two modifiers `whenNotPaused` and `whenPaused` are available to restrict invocation of certain functions based on the pause state of the contract.

## Usage

This contract must be used through inheritance and the derived contracts should include necessary modifiers on functions that need to be paused or unpaused. The `_pause` and `_unpause` methods, as well as the `paused()` view function, can be overridden in derived contracts to customize their behavior.

## Events

- The `Paused` event is emitted when the contract is paused.
- The `Unpaused` event is emitted when the contract is unpaused.

## Modifiers

- `whenNotPaused`: Functions using this modifier can be called only when the contract is not paused.
- `whenPaused`: Functions using this modifier can be called only when the contract is paused.

## Prerequisites

To use this contract effectively and safely, you should have a fundamental understanding of Solidity, Ethereum smart contracts, and the OpenZeppelin Contracts library.

## Available Since
This contract module is available since version 2.2.0 of OpenZeppelin Contracts library.