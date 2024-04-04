# Uniswap Token Swap Tool

This tool allows you to swap tokens on Uniswap, a protocol on Ethereum for swapping tokens without the need for buyers and sellers to create demand.

## Description

This tool gets the input token amount and the output token amount for a swap on the Uniswap platform.

## Parameters

- `tokenin`: The token you are swapping from.
  - Defaults to 'ETH'.

- `tokenout`: The token you are swapping to.
  - Defaults to 'USDC'.

- `amount`: The amount of tokens you want to swap.
  - Defaults to 100.

## Returns

- A dictionary object containing the `input_token_amount` and the `output_token_amount`.

## Usage

```python
# Initialize the tool
swap_tool = SwapTool()

# Get the input and output token amount
result = swap_tool.call(tokenin='ETH', token_out='USDC', amount=1)

# Print the result
print(result)
```

## Dependencies

- Commune Module: A pre-built Python module.
- Json: A standard Python library for working with JSON data.
- Web3: A Python library for interacting with Ethereum. If not installed, use pip to install it as `pip install web3`.
- Autonomy Tool: A pre-built Python module.

## Setup

This tool connects to Ethereum through an Infura node. An Infura URL and the Uniswap contract address are already included in the code, but if you want to change them, edit the following lines:

```python
infura_url = "https://tame-tame-mound.ethereum-goerli.discover.quiknode.pro/60eccfb9952049ec1b6afe53922c5c006fff17cd/"
address = '0x2a1530C4C41db0B0b2bB646CB5Eb1A67b7158667'
```
