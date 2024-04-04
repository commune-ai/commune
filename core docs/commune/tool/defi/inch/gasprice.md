# 1Inch Gas Price Fetcher

This Python module uses the 1Inch Gas Price API to fetch the current base gas price.

1Inch is a decentralized finance protocol on the Ethereum blockchain that provides various financial services. 

You will need an API key from 1Inch to use this module.  

## Requirements

1. You need to have Python 3.7 or later installed on your computer.
2. Required Python packages: `requests`, `json`, `os`, and `commune`.
3. A valid 1Inch API key.
4. An Internet connection is required to fetch the data from API.

## Installation

1. Download or clone this repository to your local computer.
2. Navigate to the directory containing the `SwaggerInch.py` file in a terminal or command prompt.
3. Install the required Python packages.

```bash
pip install requests json os commune
```

## Usage

Create an instance of SwaggerInch class, then call this instance to get the current base gas price.

Please ensure that environment variable 'INCH_API_KEY' has been correctly set with your valid 1Inch API key.

```python
inch_instance = SwaggerInch()

gas_price = inch_instance.call()
print(gas_price)
```

The returned data is the current base gas price.
