# Lido APY Fetcher

This Python module uses the Defillama API to fetch the most recent Annual Percentage Yield (APY) of Lido, a popular decentralized finance protocol on the Ethereum blockchain. 

You can use this module to filter data by chain and project. The results are presented in a dictionary that includes data for APY, market, chain, and timestamp.

## Requirements

1. You need to have Python 3.7 or later installed on your computer.
2. Required Python packages: `requests`, `json`, `dotenv`, `web3` and `commune`.
3. An Internet connection is required to fetch the data from API.

## Installation

1. Download or clone this repository to your local computer.
2. Navigate to the directory containing the `Lido.py` file in a terminal or command prompt.
3. Install the required Python packages.

```bash
pip install requests json dotenv web3 commune
```

## Usage

Create an instance of Lido and call it with specific parameters to get filtered APY data.

```python

lido_instance = Lido()

# Fetch data for a specific chain
params = {
    "chain": "Ethereum",
    "project": "lido",
}

data = lido_instance.call(**params)
print(data)
```

The returned data is a list of dictionaries containing APY, market, chain and timestamp information.
