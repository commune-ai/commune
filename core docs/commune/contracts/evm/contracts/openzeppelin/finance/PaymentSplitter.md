# PaymentSplitter

The `PaymentSplitter` smart contract allows for splitting Ether payments among a group of account addresses. Created by OpenZeppelin, the sender doesn't have to be aware of the division since it's implemented directly by the contract. 

The division can be equal or arbitrary, decided by assigning each account a number of shares. Of all the Ether received by the contract, each account can claim an amount proportional to the percentage they were assigned. The allocation of shares is determined at deployment and can't be updated afterwards.

## Features

- It supports a pull payment model. Payments aren't automatically forwarded but are kept in the contract, and the transfer requires triggering by calling the `release` function.
- The contract expects ERC20 tokens similar to native tokens (Ether).
- The constructor receives both the payees' addresses and shares upon contract creation.
- The contract keeps track of released payments to prevent double payments.
- The contract allows for querying the total shares held by payees, the total amount of Ether already released, the amount of shares held by an account, among other data.
- The contract emits events when payments are received, released, or when a new payee is added.

## Usage

To create a new instance of `PaymentSplitter`, the creator has to provide the list of payees (addresses) and their corresponding shares. These should be of the same size and there should be no duplicate payees.

The `release` function triggers a transfer to the chosen account of the amount of Ether they're owed, based on their percentage of total shares and their previous withdrawals.

## Warning

Please note that this contract should not be used with tokens that rebalance or apply fees during transfer due to potential malfunction.

## Events 

**PayeeAdded:** Emitted whenever a new payee is added, with their address and amount of shares they own.

**PaymentReceived:** Emitted when a payment is received.

**PaymentReleased:** Emitted when a payment has been released to a payee.

**ERC20PaymentReleased:** Emitted when a payment (in ERC20 tokens) has been released to a payee.