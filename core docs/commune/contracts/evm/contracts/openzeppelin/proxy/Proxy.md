# Proxy Contract

The `Proxy` contract is an abstract contract that provides functionality to delegate calls to another contract, referred to as the "implementation" contract. This is done using the EVM instruction `delegatecall`.

## Features

- **Delegate Calls:** The `_delegate` function uses assembly code to delegate calls from the proxy contract to the implementation contract. It does this by copying `msg.data` and calling the implementation. 

- **Fallback Function:** The contract provides a fallback function that delegates calls to the address returned by the `_implementation` function. This is triggered if no other function in the contract matches the call data.

- **Receive Function:** Like the Fallback function, this contract provides a receive function that delegates calls to the implementation contract if the call data is empty.

- **Before Fallback Hook:** The `_beforeFallback` hook is called before falling back to the implementation. This can be triggered as part of a manual fallback execution or as part of the Solidity fallback or receive functions. It can be overridden, but if overridden, should call `super._beforeFallback()`.

## Assembly Code

The contract makes use of Ethereum assembly code to manipulate data at low level, providing a complete control of memory. 

## Prerequisites

To use this contract, you should be familiar with Solidity, Ethereum's EVM, and the principles of proxy contracts. If you plan to override `_implementation` or `_beforeFallback` ensure that you understand the base behavior that you are changing.

## Contract Safety

Consider that the contract uses manual memory management and inline assembly which could potentially open up risks and complexities, ensure to handle all potential edge cases and thoroughly test your contracts to guarantee safety and correctness.

## Available Since
The `Proxy` contract was first introduced in OpenZeppelin v2.0.0.