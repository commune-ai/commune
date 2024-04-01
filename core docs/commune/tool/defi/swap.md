# SwapTool

This is a Python module that facilitates swapping tokens on the Uniswap exchange using Ethereum. This module uses the Ethereum Web3 library and Uniswap's smart contracts to perform the exchange.

## Features

- Swap tokens on Uniswap.
- Connect to an Ethereum provider (e.g., Infura).
- Fetch current exchange rate between any two supported tokens.

## Requirements

- Python 3.6+
- web3 library
- Ethereum node access (e.g., Infura)

## Setup/Installation

Make sure you have Python 3.6+ and pip installed.

- Firstly, install web3 using pip:

  `pip install web3`

- Then, make sure you have an Ethereum node access (from Infura, for example). Replace the `infura_url` in the `__init__()` method with your Infura URL.
  
- In the Uniswap contract ABI json, replace the contract address ('YOUR_UNISWAP_CONTRACT_ADDRESS') with Uniswap's contract address.

## Usage

Here is a quick example:

```python
if __name__ == "__main__":
    swap_tool = SwapTool()
    result = swap_tool.call(tokenin='ETH', token_out='USDC', amount=1)
    print(result)
```
