# 1Inch Token Price Fetcher

This Python module uses the 1Inch's Price API to fetch prices of tokens provided in a list or for a specific token from a list of whitelisted tokens.

1Inch offers several DeFi services and is based on the Ethereum blockchain.

For the module to work, you need a 1Inch API key.

## Requirements

1. Python 3.7 or later needs to be installed on your computer.
2. Python packages `requests`, `json`, `os`, `web3`, `dotenv`, and `commune` are required.
3. A valid API key from 1Inch is required.
4. For fetching data from the 1Inch API, you need an internet connection.

## Installation

1. Download or clone this repository onto your local computer.
2. Open a terminal or command prompt and navigate to the directory containing the `Inch.py` file.
3. Install the required Python packages with pip.

```bash
pip install requests json os web3 dotenv commune
```

## Usage

First you have to create an instance of the `Inch` class. You can then call methods of this instance.

Remember to use your actual account key in the place of `ONEINCH_API_KEY`.

```python
inch_instance = Inch(api_key='ONEINCH_API_KEY')

# Fetch prices for whitelisted tokens
inch_instance.get_whitelisted_token_prices()

# Fetch prices for specific tokens
tokens = ['TOKEN1', 'TOKEN2', 'TOKEN3']
inch_instance.get_requested_token_prices(tokens)

# Fetch prices using specific addresses
addresses = ['ADDRESS1', 'ADDRESS2', 'ADDRESS3']
inch_instance.get_prices_for_addresses(addresses)
```

These methods return the current prices of the specified tokens or an error message.
