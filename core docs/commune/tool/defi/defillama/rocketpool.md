# Rocket Pool APY Fetcher

This Python module connects to the DefiLlama API and allows the user to select which chain, project, or symbol they want in terms of fetching Rocket Pool data.

Rocket Pool is a decentralized Ethereum Proof of Stake pool. This module fetches the latest Annual Percentage Yield (APY) of Rocket Pool.

## Requirements

1. You need to have Python 3.7 or later installed on your computer.
2. Required Python packages: `requests`, `json`, `dotenv`, `web3` and `commune`.
3. An Internet connection is required to fetch the data from API.

## Installation

1. Download or clone this repository to your local computer.
2. Navigate to the directory containing the `RocketPool.py` file in a terminal or command prompt.
3. Install the required Python packages.

```bash
pip install requests json dotenv web3 commune
```

## Usage

Create an instance of RocketPool and call it with specific parameters to get filtered APY data.

```python

rocket_pool_instance = RocketPool()

# Fetch data for a specific chain and project
params = {
    "chain": "Ethereum",
    "project": "rocket-pool",
    "symbol": "RETH"
}

data = rocket_pool_instance.call(**params)
print(data)

```

The returned data is a dictionary containing the highest APY, market, asset, chain, and timestamp obtained among the filtered results.

The module's main function, `call()`, can fetch APY data based on the chain, project, and symbol specified. By default, it fetches data for the 'rocket-pool' project.
If no data is found for the given parameters, the function will return an error message.
