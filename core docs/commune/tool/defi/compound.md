# Ethereum ERC20 Token and Compound Finance Interaction Script

This Python script uses the Web3 interface to the Ethereum blockchain and helps the user interact with ERC20 tokens and the Compound Finance protocol. 

The two primary capabilities of this script are:
1. Retrieving the quantity of ERC20 tokens an address (referred to as a "spender") is permitted to spend on behalf of another address (referred to as the "owner"). This is known as an 'allowance'.
2. Supplying tokens to the Compound Finance protocol through a transaction. 

For this script to function properly, it requires the following:
1. A local or remote Ethereum node to interact with.
2. The transaction sender's private key. 

## Requirements

1. Python version 3.7 or greater.
2. The Python Web3 package (`pip install web3`).
3. An operational Ethereum node (either locally run or remotely hosted).
4. The contract address and ABI of the ERC20 token that the script will be interacting with.
5. The contract address and ABI of the Compound Finance protocol.
6. An environment variable (PRIVATE_KEY) which represents the sender's Ethereum address' private key. 

## Usage

You'll first need to initialize the Web3 instance to connect to your Ethereum node:

```python
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
```

If you want to fetch allowance:

```python
allowed_amount = allowance(owner_address, spender_address, token_address)
print(f"The allowed amount is: {allowed_amount}")
```

When wanting to supply tokens to Compound:

```python
transaction_hash = supply(dst_address, asset_address, amount)
print(f"Transaction hash: {transaction_hash}")
```

Please note that "owner_address", "spender_address", "dst_address", "asset_address", and "amount", should all be actual Ethereum addresses or amounts supplied by the user. 

## Installation

1. Download or clone this repository onto your local machine.
2. Find the directory that contains the `erc20_compound_interaction.py` file.
3. Install the necessary packages:

    ```
    pip install web3 python-dotenv
    ```

You need to replace `'0xYourSenderAddress'` and `PRIVATE_KEY` with their real values.
