# Contract: GiveMe

This contract demonstrates how to transfer value using the `self.env().transfer()` function in the Ink! smart contract language for Substrate.

## Prerequisites

* Rust nightly version
* Canvas Node to execute contract

## Deploying the contract

This contract has two functions which transfer balances:

1. give_me function: It takes an amount as input. If the contract has sufficient balance, it transfers the requested amount to the caller.

2. was_it_ten function: It verifies if the payment sent with this call is exactly `10`. If not, the transaction will be reverted.

## Testing

The contract includes unit tests that verify the transfer functions. They check if the transfer works when the contract balance is higher or lower than the requested amount. 

A second function `was_it_ten` is also tested to ensure that it only accepts the amount of `10`.

Apart from unit tests, end-to-end tests are also included to interact with the contract in a real chain environment.
The E2E tests check if an error is returned when an amount, which is more than the contract's balance, is requested. They also verify if a transfer of value from contract to sender is successful when the contract has sufficient balance.

## Note

The `give_me` function can panic due to three reasons 
1. If the requested transfer exceeds the contract balance.
2. If the requested transfer would bring this contract's balance below the minimum balance (i.e. the chain's
   existential deposit).
3. If the transfer failed due to some other reason.

The `was_it_ten` function can panic if the amount of payment with the call is not exactly `10`.
