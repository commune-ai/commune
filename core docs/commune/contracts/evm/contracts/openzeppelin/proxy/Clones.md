# Clones Library

The `Clones` library provides functions to create "clone" smart contracts that mimic the behavior of a given implementation contract, in a cost-efficient manner. It is used to implement the EIP-1167 standard which deploys minimal proxy contracts.

## Features

- **Clone:** The clone function deploys a clone contract that mimics the behaviour of an implementation contract and returns the address of this clone.

- **CloneDeterministic:** Deploys and returns the address of a deterministic clone that mimics the behavior of `implementation`. The deployed clone will be the same at the same address if the `implementation` and `salt` used are the same.

- **PredictDeterministicAddress:** This function can predict the address that a deterministic clone will have before its deployment. It has two variants, one with a deployer parameter (the address where the cloning transacation will originate from) and one without (defaults to the address of the current contract).

## Assembly Primitives

Given the nature of the functions, they are implemented using Ethereum assembly (EVM bytecodes). This requirement is due to the reason that deploying minimal proxy contracts involves careful creation and manipulation of raw bytecodes.

## Prerequisites

To fully understand and use these functions, you should be familiar with Solidity, Ethereum's EVM, and principles of proxy contracts. You should also be aware of the risks and complexities associated with the use of `assembly` code, especially in relation to issues of gas and memory safety.

## Notice

While these functions theoretically cannot fail under normal circumstances, always remember to handle the potential edge cases and always conduct rigorous testing to ensure the safety and correctness of your contracts. Note that cloning an implementation requires that the clone works correctly with the state layout declared by the implementation contract.

## Available Since
This library is available since OpenZeppelin v3.4.
