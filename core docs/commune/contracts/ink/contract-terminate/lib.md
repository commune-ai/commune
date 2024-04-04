# JustTerminate Solidity Contract

This repository contains a simple smart contract, `JustTerminate`, developed in Solidity and deployed on the Ethereum network. The primary function of this contract is to demonstrate the behavior of the `self.env().terminate()` function. It self-terminates once the `terminate_me()` function is called.

## Description

`JustTerminate` is a contract that does not require any storage. It provides a function, `terminate_me()`, which terminates the contract with the caller set as the beneficiary. 

The contract includes:

- `new()`: A constructor function that deploys a new instance of the contract.
- `terminate_me()`: A function that terminates the contract with the caller set as the beneficiary. 

## Tests

The contract includes a test case `terminating_works()` to check if the contract termination functionality works correctly. The function `assert_contract_termination` is used to check the termination of the contract.

## End-to-end Tests

This contract also includes end-to-end testing which checks if the contract correctly terminates. This e2e test checks for specific events that are expected to occur during the contract termination process. 

## Development setup

To work with this contract, you'll need to set up a local Ethereum environment. Given the contract is written in Solidity, you'll be needing the Solidity compiler `solc`.

Once you have your local environment set up and configured, you can clone the repository, and deploy the contracts to your local Ethereum environment.

## Deployment

For deployment, first compile the contract with `solc` then deploy the contract using a wallet capable of interacting with the Ethereum network like MetaMask.

## Authors 

This contract is a community-developed project.

## License

The contract and accompanying project files are licensed under the MIT License.