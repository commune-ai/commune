import commune as c
import os

# Load API credentials from environment variables (for security)
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

def main():
    # Initialize Binance module
    binance = c.module('binance')(
        api_key=API_KEY,
        api_secret=API_SECRET
    )
    
    # If no API credentials, we can still get public data
    if not API_KEY or not API_SECRET:
        c.print("[yellow]Running in public data mode (no API credentials provided)[/yellow]")
    
    # Get BTC/USDT price
    btc_price = binance.get_ticker_price('BTCUSDT')
    c.print(f"[green]Current BTC/USDT price:[/green] {btc_price}")
    
    # Get ETH/USDT price
    eth_price = binance.get_ticker_price('ETHUSDT')
    c.print(f"[green]Current ETH/USDT price:[/green] {eth_price}")
    
    # Get order book for BTC/USDT
    order_book = binance.get_order_book('BTCUSDT', limit=5)
    c.print(f"[green]BTC/USDT Order Book (top 5):[/green]")
    c.print(f"Bids: {order_book.get('bids', [])}")
    c.print(f"Asks: {order_book.get('asks', [])}")
    
    # Get historical klines (candlestick data)
    klines = binance.get_historical_klines(
        symbol='BTCUSDT',
        interval='1h',
        start_str='1 day ago UTC',
        limit=24
    )
    c.print(f"[green]BTC/USDT Historical Data (24 hours, hourly):[/green]")
    c.print(f"Number of candles: {len(klines)}")
    
    # If API credentials are provided, get account information
    if API_KEY and API_SECRET:
        c.print("\n[green]Account Information:[/green]")
        account_info = binance.get_account_info()
        
        if 'error' in account_info:
            c.print(f"[red]Error getting account info: {account_info['error']}[/red]")
        else:
            # Get and print balances with non-zero amounts
            balances = [b for b in account_info.get('balances', []) 
                        if float(b.get('free', 0)) > 0 or float(b.get('locked', 0)) > 0]
            
            c.print(f"Number of assets: {len(balances)}")
            for balance in balances:
                c.print(f"Asset: {balance['asset']}, Free: {balance['free']}, Locked: {balance['locked']}")

if __name__ == "__main__":
    main()