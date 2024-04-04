# ERC-20 smart contract

This crate contains an implementation of a simple ERC-20 smart contract with the most basic functionality:

- Create and track the total supply of tokens.
- Transfer tokens from one account to another.
- Allow one account to spend tokens on behalf of another account (with approval).
- Get the current token balance of an account.
- Get the current allowance of an account (how many tokens another account can spend on their behalf).

## Basic Usage

To use the contract:

1. Instantiate the contract by calling the `new` function with an initial token supply. This creates the total supply of tokens and assigns the entire supply to the creator's account.
2. Tokens can be transferred between accounts using the `transfer` function.
3. An account may also approve another account to spend tokens on its behalf using the `approve` function. The spender can then use the `transfer_from` function to transfer tokens from the approver's account.
4. The `balance_of` function can be used to check the token balance of any account.
5. The `allowance` function can be used to check how many tokens the owner has allowed a spender to use.

For example:

```rust
let mut contract = Erc20::new(1000);
contract.transfer(AccountId::from([0x02; 32]), 100);
contract.approve(AccountId::from([0x03; 32]), 50);
contract.transfer_from(AccountId::from([0x01; 32]), AccountId::from([0x03; 32]), 50);
println!("Account 1 balance: {}", contract.balance_of(AccountId::from([0x01; 32])));
println!("Account 2 balance: {}", contract.balance_of(AccountId::from([0x02; 32])));
println!("Account 3 balance: {}", contract.balance_of(AccountId::from([0x03; 32])));
```

## Error Handling

The contract uses standard [Rust error handling patterns](https://doc.rust-lang.org/book/ch09-00-error-handling.html). In case of failure, functions will return a `Result` containing an `Error` value that can be unwrapped or pattern matched to diagnose the issue.

Currently, two types of errors are defined:

- `InsufficientBalance`: Returned when an account does not have enough tokens to complete a transfer.
- `InsufficientAllowance`: Returned when an account attempts to use `transfer_from` to spend more tokens than it has been approved for by the token owner.

## Test

The contract includes a test module with unit tests for the core functionality (creating tokens, transferring tokens, approving allowances, etc.) and some error conditions. The tests use the `ink::test` environment to simulate a blockchain for testing purposes, with a set of predefined accounts for use in testing.

For example:

```rust
#[ink::test]
fn new_works() {
    // Constructor works.
    let _erc20 = Erc20::new(100);
    // Transfer event triggered during initial construction.
    let emitted_events = ink::env::test::recorded_events().collect::<Vec<_>>();
    assert_eq!(1, emitted_events.len());
    assert_transfer_event(
        &emitted_events[0],
        None,
        Some(AccountId::from([0x01; 32])),
        100,
    );
}
```

To run the tests, you can use `cargo test --package erc20`.

