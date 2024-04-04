# ReentrancyGuard Contract

The `ReentrancyGuard` contract is a module that helps in preventing re-entrant calls to a function. 

This contract provides a `nonReentrant` modifier which can be applied to functions to ensure that there are no nested (reentrant) calls to them. The reentrancy guard ensures that a contract cannot call itself, directly or indirectly. 

Note: functions marked as `nonReentrant` cannot call each other. This can be overcome by making these functions `private`, and then adding `external` `nonReentrant` entry points to them.

## Features

- **NonReentrant Modifier**: The `nonReentrant` modifier is used to prevent a function from being re-entered during a call.

- **Reentrancy Status Tracking**: The contract maintains a `_status` state variable to track reentrancy. Two constants `_NOT_ENTERED` and `_ENTERED` represent the status.

- **Modifier Implementation**: The `nonReentrant` modifier is implemented with `_nonReentrantBefore` and `_nonReentrantAfter` private functions, which set the `_status` for controlling re-entrancy.

- **Reentrancy Status Check**: It provides `_reentrancyGuardEntered` function for checking if the guard is set to "entered".

## Usage

To make use of `ReentrancyGuard`, inherit from the contract and apply the `nonReentrant` modifier to functions that must not be reentrant.

## Example:

```solidity
contract MyContract is ReentrancyGuard {
    uint256 public counter;

    constructor() {
        counter = 0;
    }

    function incrementCounter() public nonReentrant {
        counter += 1;
    }
}
```
In the example above, the `incrementCounter` function cannot be re-entered while it is in the process of execution.

## Available Since:
The `ReentrancyGuard` contract is available in the OpenZeppelin Contracts since version 4.4.1. 

## Prerequisites:
The user should have basic understanding of Ethereum Smart Contracts and Solidity.