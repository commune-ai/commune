# 1Inch Token Prices Fetcher

This Python script uses the 1Inch's Price API to fetch the prices of a list of given tokens.

1Inch is a decentralized finance protocol running on the Ethereum blockchain.

An API key from 1Inch is required to use this script.

## Requirements

1. Python version 3.7 or higher needs to be installed on your computer.
2. The required Python libraries: `requests`, `json`, `os`, `web3`, `dotenv` and `commune`.
3. A valid 1Inch API key.
4. Access to the internet to fetch the data from the API.

## Installation

1. Download or clone this repository to your local computer.
2. Navigate to the directory containing the `InchPrices.py` file using a terminal or command prompt.
3. Install the required Python libraries.

```bash
pip install requests json os web3 dotenv commune
```

## Usage

Firstly, you need to create an instance of the `InchPrices` class. You can then call methods of this instance.

Ensure that you replace `'INCH_API_KEY'` with your actual account key.

```python
instance = InchPrices(api_key='INCH_API_KEY')

# Get prices for specific tokens
result = instance.call(tokens=['weth', 'usdc'])
print(result)
```

The `call()` method returns the current prices of the specified tokens or an error message.
