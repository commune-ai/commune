# Token Price Comparison Tool

This tool allows you to compare the prices of two cryptocurrency tokens.

## Description

The Token Price Comparison Tool takes in the names and prices of two tokens, and returns the name of the token that is either cheaper or more expensive, depending on your preference.

## Parameters

- `token1_name`: A string that represents the name of the first token.
- `token1_price`: A float that represents the current price of the first token.
- `token2_name`: A string that represents the name of the second token.
- `token2_price`: A float that represents the current price of the second token.
- `find_cheaper`: A boolean that indicates whether you want to find the cheaper token (if set to True) or the more expensive token (if set to False).

## Returns

A string that represents the name of the token that is either cheaper or more expensive, depending on the `find_cheaper` boolean input.

If both tokens have the same price, the function will return a string stating "Both tokens have the same price."

## Usage

The usage of the tool is as simple as calling the `call` function on a `CompareTokenPrice` object with the appropriate parameters. Here is an example:

```python
token1_name = "ETH"
token1_price = 3000.0
token2_name = "BTC"
token2_price = 45000.0

cheaper_token = CompareTokenPrice().call(token1_name, token1_price, token2_name, token2_price, find_cheaper=True)
print(f"The cheaper token is {cheaper_token}")

more_expensive_token = CompareTokenPrice().call(token1_name, token1_price, token2_name, token2_price, find_cheaper=False)
print(f"The more expensive token is {more_expensive_token}")
```

## Dependencies

This tool is built using the `commune` module. The external dependencies are:

- `commune`: A pre-built Python module. If `commune` is not already installed in your environment, you can install it by executing `pip install commune` in your terminal.

## Setup

There is no additional setup necessary for this tool. All you need is the names and prices of the two tokens you want to compare. 
