# DefiLlama DeFi Pool Data Fetcher

This Python module uses the DefiLlama API to fetch most recent data of pools in DeFi protocols on specified Ethereum blockchain parameters.

You can use this module to filter data by chain, project and symbol. The results are presented in a dictionary that includes data for APY, market, asset, chain and timestamp.

## Requirements

1. You need to have Python 3.7 or later installed on your computer.
2. Required Python packages: `requests`, `json`, `dotenv`, `web3` and `commune`.
3. An Internet connection is required to fetch the data from API.

## Installation

1. Download or clone this repository to your local computer.
2. Navigate to the directory containing the `DefiLlama.py` file in a terminal or command prompt.
3. Install the required Python packages.

```bash
pip install requests json dotenv web3 commune
```

## Usage

Create an instance of DefiLlama and call it with specific parameters to get filtered pool data.

```python

dl_instance = DefiLlama()

# Fetch data for a specific chain and project
params = {
    "chain": "Ethereum",
    "project": "compound",
    "symbol": "ETH"
}

data = dl_instance.call(**params)
print(data)

```

The returned data is a list of dictionaries containing APY, market, asset, chain and timestamp information.
