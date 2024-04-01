# PullPayment Contract

The `PullPayment` contract facilitates a pull-payment strategy, which is considered a best practice when transferring Ether in a secure way. This contract ensures that the receiving account must withdraw the payments itself, and does not interact directly with the paying contract. This strategy prevents recipients from blocking contract execution and eliminates reentrancy concerns.

## Features

- **Escrow Support**: The `PullPayment` contract interacts with an `Escrow` contract which is created when the contract is deployed.

- **Withdraw Payments Function**: This function allows to withdraw accumulated payments, with all gas being forwarded to the recipient. Note that any account can call this function, not just the payee.

- **View Payments**: A public function `payments` allows to view the payments owed to an address using the `depositsOf` function of the `Escrow` contract.

- **Asynchronous Transfer Function**: The internal function `_asyncTransfer` allows the payer to store the sent amount as credit to be pulled by the payee at a later time.

## Use

To use this contract, derive from the `PullPayment` contract and use the function `_asyncTransfer` instead of Solidity's `transfer` function. Payees can query their due payments with `payments`, and retrieve them with `withdrawPayments`.

## Security Recommendations

While using the `PullPayment` contract's `withdrawPayments` function forwards all gas to the recipient, be aware that this opens a door to reentrancy vulnerabilities. Ensure that you trust the recipient, or follow the checks-effects-interactions pattern or use a `ReentrancyGuard`.

## Events

The events `Withdrawn` and `Deposited` are emitted by the `Escrow` contract.

## Prerequisites

To use this contract effectively and safely, you should have a fundamental understanding of Solidity and Ethereum smart contracts.

## Contract Safety

Contracts unaware of the `PullPayment` strategy can still receive funds this way, by having a separate account call the `withdrawPayments`.

## Available Since

This `PullPayment` contract is available in the OpenZeppelin Contracts starting from version 4.4.0.