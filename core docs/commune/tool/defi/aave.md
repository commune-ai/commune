# Ethereum ERC20 Token and Aave Supply Interaction Script

This Python script uses the Web3 interface to the Ethereum blockchain to interact with ERC20 Tokens and the Aave protocol.

The script provides two main functionalities:
1. It fetches the allowed amount of ERC20 tokens that a specific address (spender) can spend on behalf of another address (owner). This is also known as 'allowance'.
2. It sends a transaction to supply tokens to Aave.

For this script to work, you must have:
1. A running Ethereum node (local or remote). In this code, it's assumed that a local Ethereum node is running at http://localhost:8545.
2. The private key of the Ethereum address making the transactions.

## Requirements

1. Python 3.7 or higher is needed.
2. The Web3 Python package should be installed. You can use pip for this (`pip install web3`).
3. An Ethereum node (local or remote) that the script can interact with.
4. The contract address and ABI of the ERC20 token you wish to interact with.
5. The contract address and ABI of the Aave protocol.
6. A valid PRIVATE_KEY environment variable of the sender's Ethereum address.

## Usage

First, initialize the Web3 instance with the Ethereum node HTTP provider:

```python
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
```

To fetch allowance:

```python
allowed_amount = allowance(owner_address, spender_address, token_address)
print(f"The allowed amount is: {allowed_amount}")
```

To supply tokens to Aave:

```python
transaction_hash = supply(dst_address, asset_address, amount)
print(f"Transaction hash: {transaction_hash}")
```

"owner_address", "spender_address", "dst_address", "asset_address", and "amount" are Ethereum addresses or amounts that should be provided by the user.

## Installation

1. Download or clone this repository to your local machine.
2. Navigate to the directory containing the `erc20_aave_interaction.py` file.
3. Install the required packages:

    ```
    pip install web3 python-dotenv
    ```

Remember to replace `'0xYourSenderAddress'` and `PRIVATE_KEY` with actual values.
