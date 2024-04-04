# VestingWallet

The `VestingWallet` smart contract manages the vesting of Ether and ERC20 tokens for a specified beneficiary. It can custody multiple tokens which will follow a pre-defined vesting schedule. This schedule is customizable through the `vestedAmount` function.

Any token sent to this contract will adhere to the vesting schedule as if they were locked from the start. If the vesting period has already begun, any tokens sent to this contract can be immediately or partly releasable.

## Features

- Define the beneficiary, start timestamp and vesting duration of the vesting wallet.
- Retrieve the beneficiary's address, start timestamp, the vesting duration, amount of ETH already released, and amount of certain ERC20 token already released.
- Check the amount of releasable ETH or ERC20 tokens.
- Release ETH or ERC20 tokens that have already vested.
- Calculate the amount of ETH or ERC20 tokens that has already vested.
- Customize the vesting formula to determine the amount vested given the total historical allocation.

## Events

- `EtherReleased`: Emitted when Ether has been released. 
- `ERC20Released`: Emitted when ERC20 tokens have been released.

## Usage

- Initialize the contract by providing the beneficiary's address, start timestamp for vesting, and the duration in seconds.
- To add assets to the vesting wallet, you can simply send it to the smart contract address. The contract should be engineered to receive ETH.
- Calling the `release()` function triggers the release of vested Ether to the beneficiary, emitting an `EtherReleased` event.
- Use the `release(token)` function to release the vested ERC20 token, where `token` is the address of the ERC20 token. This emits an `ERC20Released` event.

## Warning

The contract uses linear vesting by default but the vesting formula can be overridden to implement custom vesting schedules. The `_vestingSchedule` method, which calculates the vested amount based on timestamp and total allocation, can be adjusted as per needs.
  
## Interface and Methods 

The contract inherits from the `Context` contract and uses features from `SafeERC20`, `Address`.

Main Methods in the Contract: 
- `beneficiary()`: Returns the beneficiary address
- `start()`: Returns the start timestamp for vesting
- `duration()`: Returns the vesting duration in seconds
- `released()`: Returns total amount of Eth already released
- `released(token)`: Returns total amount of provided ERC20 tokens already released
- `releasable()`: Returns total amount of releasable Eth
- `releasable(token)`: Returns total amount of releasable ERC20 tokens
- `release()`: Releases the vested native token (ether)
- `release(token)`: Releases the vested ERC20 tokens
- `vestedAmount(timestamp)`: Calculates the amount of Ether that has already vested
- `vestedAmount(token, timestamp)`: Calculates the amount of ERC20 tokens that has already vested
- `_vestingSchedule(totalAllocation, timestamp)`: Calculates the amount vested as a function of time for an asset given its total historical allocation.