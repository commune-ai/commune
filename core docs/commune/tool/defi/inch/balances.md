# 1Inch Token Balance Fetcher

This Python module uses the 1Inch Balance API to fetch token balances for a specified wallet address. 

1Inch is a decentralized finance protocol on the Ethereum blockchain that provides aggregated price feeds from various decentralized exchanges. 

You will need an API key from 1Inch to use this module.  

## Requirements

1. You need to have Python 3.7 or later installed on your computer.
2. Required Python packages: `requests`, `json`, and `commune`.
3. A valid 1Inch API key.
4. An Internet connection is required to fetch the data from API.

## Installation

1. Download or clone this repository to your local computer.
2. Navigate to the directory containing the `Inch.py` file in a terminal or command prompt.
3. Install the required Python packages.

```bash
pip install requests json commune
```

## Usage

Create an instance of Inch class, then call this instance with specific parameters to get token balance data.

```python
inch_instance = Inch(api_key=YOUR_1INCH_API_KEY)

# Fetch token balances for a specific wallet address
wallet_address = 'YOUR_WALLET_ADDRESS'
data = inch_instance.call(wallet_address)
print(data)
```

The returned data is a JSON formatted string with token balances, which can be converted to a Python dictionary using `json.loads()`.

The module also includes a class method `test()` for testing purposes. You can use an Ethereum wallet address to call this method and check if your program is working properly.
