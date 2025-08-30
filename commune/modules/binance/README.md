# Binance API Module

This module provides a simple interface to interact with the Binance cryptocurrency exchange API.

## Installation

To use this module, you need to install the required dependencies:

```bash
python -m commune.modules.binance.binance install
```

Or directly:

```bash
pip install python-binance
```

## Usage

### Basic Usage

```python
import commune as c

# Initialize without API credentials (limited functionality)
binance = c.module('binance')()

# Get current BTC/USDT price
price = binance.get_ticker_price('BTCUSDT')
print(price)

# Get all prices
all_prices = binance.get_all_prices()
print(all_prices)
```

### With API Credentials

```python
import commune as c

# Initialize with API credentials
binance = c.module('binance')(
    api_key='your_api_key',
    api_secret='your_api_secret'
)

# Get account information
account_info = binance.get_account_info()
print(account_info)

# Get balances
balances = binance.get_balances()
print(balances)

# Place a limit order
order = binance.place_order(
    symbol='BTCUSDT',
    side='BUY',
    order_type='LIMIT',
    quantity=0.001,
    price=20000,
    time_in_force='GTC'
)
print(order)
```

## Available Methods

- `get_account_info()`: Get account information
- `get_balances()`: Get all asset balances
- `get_ticker_price(symbol)`: Get latest price for a symbol
- `get_all_prices()`: Get all ticker prices
- `get_order_book(symbol, limit)`: Get order book for a symbol
- `place_order(symbol, side, order_type, quantity, price, time_in_force)`: Place a new order
- `get_order(symbol, order_id)`: Get order information
- `cancel_order(symbol, order_id)`: Cancel an order
- `get_historical_klines(symbol, interval, start_str, end_str, limit)`: Get historical klines (candlestick data)
- `get_exchange_info()`: Get exchange information

## Security Note

Never hardcode your API keys in your code. Use environment variables or secure configuration files.