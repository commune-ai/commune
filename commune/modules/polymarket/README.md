# Polymarket Module

A Commune module for interacting with Polymarket prediction markets. This module provides a simple interface to access market data, prices, and historical information from Polymarket.

## Features

- Get detailed information about specific prediction markets
- List available markets with filtering and pagination
- Retrieve current market prices
- Access historical market data
- Simple, unified interface following Commune module patterns

## Installation

1. Make sure you have the required dependencies:
```bash
pip install py-clob-client python-dotenv
```

2. Set up your environment variables in a `.env` file:
```env
CLOB_HOST=https://clob.polymarket.com
KEY=your_private_key_from_polymarket
FUNDER=your_funder_address_from_polymarket
```

## Usage

### Basic Usage

```python
import commune as c

# Create a Polymarket module instance
pm = c.module('polymarket')

# List available markets
markets = pm.list_markets(limit=5)
for market in markets:
    print(f"Market: {market['description']}")
    print(f"Volume: {market['volume']}")
    print("---")

# Get specific market information
market_id = "some_market_id"
info = pm.get_market_info(market_id)
print(f"Title: {info['title']}")
print(f"Status: {info['status']}")

# Get market prices
prices = pm.get_market_prices(market_id)
print(f"Current Price: {prices['current_price']}")

# Get historical data
history = pm.get_market_history(market_id, timeframe="7d")
print(f"Historical data for {history['title']}")
```

### Using the Forward Method

The module also supports a unified `forward` method:

```python
# List markets
result = pm.forward(action='list_markets', limit=10, status='active')

# Get market info
result = pm.forward(action='get_market_info', market_id='some_id')

# Get prices
result = pm.forward(action='get_market_prices', market_id='some_id')

# Get history
result = pm.forward(action='get_market_history', market_id='some_id', timeframe='30d')
```

### Testing

Run the built-in test method to verify everything is working:

```python
pm = c.module('polymarket')
pm.test()
```

## API Methods

### `list_markets(status=None, limit=10, offset=0)`
List available prediction markets.

**Parameters:**
- `status` (str, optional): Filter by market status ('active', 'resolved')
- `limit` (int): Number of markets to return (default: 10)
- `offset` (int): Number of markets to skip for pagination (default: 0)

**Returns:** List of market dictionaries

### `get_market_info(market_id)`
Get detailed information about a specific market.

**Parameters:**
- `market_id` (str): Market ID or slug

**Returns:** Dictionary with market information

### `get_market_prices(market_id)`
Get current prices for a market.

**Parameters:**
- `market_id` (str): Market ID or slug

**Returns:** Dictionary with price information

### `get_market_history(market_id, timeframe='7d')`
Get historical data for a market.

**Parameters:**
- `market_id` (str): Market ID or slug
- `timeframe` (str): Time period ('1d', '7d', '30d', 'all')

**Returns:** Dictionary with historical data

## Error Handling

All methods return dictionaries. If an error occurs, the dictionary will contain an 'error' key with a description of the problem:

```python
result = pm.get_market_info('invalid_id')
if 'error' in result:
    print(f"Error: {result['error']}")
```

## Environment Variables

- `CLOB_HOST`: The Polymarket CLOB API host (default: https://clob.polymarket.com)
- `KEY`: Your private key from Polymarket
- `FUNDER`: Your funder address from Polymarket

## License

This module follows the same license as the Commune project.