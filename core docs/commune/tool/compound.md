# Compound Finance Interaction Tool 

This tool provides functionalities to interact with Compound Finance protocol on Ethereum.

## Description

The tool contains two functions - `allowance` and `supply`. The `allowance` function fetches the allowance of a spender/account from a token smart contract. The `supply` function allows an account to supply an asset to Compound Finance in order to earn interest.

## Parameters

For `allowance` function:
- `owner_address`: The Ethereum address of the token owner.
- `spender_address`: The Ethereum address of the spender.
- `token_address`: The Ethereum address of the ERC-20 token contract.

For `supply` function:
- `dst_address`: The Ethereum address of the destination compound finance pool.
- `asset_address`: The Ethereum address of the ERC-20 token contract.
- `amount`: The amount of token to be supplied in wei.

## Returns

The `allowance` function returns an integer value representing the amount of token that the spender can spend from the owner's account.

The `supply` function sends a transaction to Ethereum network and returns the transaction hash string if the operation is successful.

## Usage

```python
token_address = '0x...' # The address of the ERC20 token
owner_address = '0x...' # The owner's Ethereum address
spender_address = '0x...' # The spender's Ethereum address

# Get the allowance of the spender
allowed_amount = allowance(owner_address, spender_address, token_address)

# Print the allowed amount
print(allowed_amount)

dst_address = '0x...' # The destination compound finance pool address
asset_address = '0x...' # The address of the ERC20 token
amount = 1000000000 # The amount of token to supply in wei (1 token)
# Supply token to Compound Finance
tx_hash = supply(dst_address, asset_address, amount)

# Print the transaction hash
print(tx_hash)
```

## Dependencies

- Web3: A Python library for interacting with Ethereum. If not installed, use pip to install it as `pip install web3`.
- os: A Python built-in module to interact with the operating system environment variables.

## Setup

This tool connects to Ethereum through a local Ethereum node. If you want to connect to a different Ethereum node or Infura, edit the following line:

```python
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
```
