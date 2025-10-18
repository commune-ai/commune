import commune as c
import os
import json
from typing import Dict, List, Any, Optional, Union

class Binance:
    """
    A module for interacting with the Binance API.
    """
    def __init__(self, 
                 api_key: str = None, 
                 api_secret: str = None,
                 testnet: bool = False,
                 **kwargs):
        """
        Initialize the Binance API module.
        
        Args:
            api_key (str, optional): Binance API key. Defaults to None.
            api_secret (str, optional): Binance API secret. Defaults to None.
            testnet (bool, optional): Whether to use the testnet. Defaults to False.
            **kwargs: Additional arguments to pass to the parent class.
        """
        
        # Set API credentials
        self.api_key = api_key 
        self.api_secret = api_secret 
        self.testnet = testnet 
        
        # Initialize the client
        self.client = None
        if self.api_key and self.api_secret:
            self.connect()
    
    def connect(self):
        """
        Connect to the Binance API by initializing the client.
        
        Returns:
            The initialized client object.
        """
        try:
            # Import here to avoid dependency issues if not installed
            from binance.client import Client
            
            # Initialize the client with credentials
            self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
            return self.client
        except ImportError:
            c.print("Binance package not installed. Please run 'pip install python-binance'")
            return None
        except Exception as e:
            c.print(f"Error connecting to Binance API: {str(e)}")
            return None
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information including balances.
        
        Returns:
            Dict containing account information.
        """
        if not self.client:
            self.connect()
        
        try:
            return self.client.get_account()
        except Exception as e:
            return {'error': str(e)}
    
    def get_balances(self) -> List[Dict[str, Any]]:
        """
        Get all asset balances.
        
        Returns:
            List of dictionaries containing asset balances.
        """
        account_info = self.get_account_info()
        if 'error' in account_info:
            return [{'error': account_info['error']}]
        
        return account_info.get('balances', [])
    
    def get_ticker_price(self, symbol: str = 'BTCUSDT') -> Dict[str, Any]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol (str, optional): Trading pair symbol. Defaults to 'BTCUSDT'.
            
        Returns:
            Dict containing price information.
        """
        if not self.client:
            self.connect()
        
        try:
            return self.client.get_ticker(symbol=symbol)
        except Exception as e:
            return {'error': str(e)}
    
    def get_all_prices(self) -> List[Dict[str, Any]]:
        """
        Get all ticker prices.
        
        Returns:
            List of dictionaries containing all ticker prices.
        """
        if not self.client:
            self.connect()
        
        try:
            return self.client.get_all_tickers()
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_order_book(self, symbol: str = 'BTCUSDT', limit: int = 10) -> Dict[str, Any]:
        """
        Get the order book for a symbol.
        
        Args:
            symbol (str, optional): Trading pair symbol. Defaults to 'BTCUSDT'.
            limit (int, optional): Limit of results. Defaults to 10.
            
        Returns:
            Dict containing order book information.
        """
        if not self.client:
            self.connect()
        
        try:
            return self.client.get_order_book(symbol=symbol, limit=limit)
        except Exception as e:
            return {'error': str(e)}
    
    def place_order(self, 
                   symbol: str, 
                   side: str, 
                   order_type: str, 
                   quantity: float,
                   price: float = None,
                   time_in_force: str = 'GTC') -> Dict[str, Any]:
        """
        Place a new order on Binance.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            side (str): Order side ('BUY' or 'SELL').
            order_type (str): Order type ('LIMIT', 'MARKET', etc.).
            quantity (float): Order quantity.
            price (float, optional): Order price (required for LIMIT orders).
            time_in_force (str, optional): Time in force. Defaults to 'GTC' (Good Till Cancelled).
            
        Returns:
            Dict containing order information.
        """
        if not self.client:
            self.connect()
        
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'timeInForce': time_in_force
            }
            
            if price and order_type != 'MARKET':
                params['price'] = price
            
            return self.client.create_order(**params)
        except Exception as e:
            return {'error': str(e)}
    
    def get_order(self, symbol: str, order_id: int = None) -> Dict[str, Any]:
        """
        Get order information.
        
        Args:
            symbol (str): Trading pair symbol.
            order_id (int, optional): Order ID. Defaults to None.
            
        Returns:
            Dict containing order information.
        """
        if not self.client:
            self.connect()
        
        try:
            return self.client.get_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            return {'error': str(e)}
    
    def cancel_order(self, symbol: str, order_id: int = None) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            symbol (str): Trading pair symbol.
            order_id (int, optional): Order ID. Defaults to None.
            
        Returns:
            Dict containing cancellation information.
        """
        if not self.client:
            self.connect()
        
        try:
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            return {'error': str(e)}
    
    def get_historical_klines(self, 
                             symbol: str = 'BTCUSDT', 
                             interval: str = '1d',
                             start_str: str = '1 day ago UTC',
                             end_str: str = None,
                             limit: int = 500) -> List[List]:
        """
        Get historical klines (candlestick data).
        
        Args:
            symbol (str, optional): Trading pair symbol. Defaults to 'BTCUSDT'.
            interval (str, optional): Kline interval. Defaults to '1d'.
            start_str (str, optional): Start time. Defaults to '1 day ago UTC'.
            end_str (str, optional): End time. Defaults to None.
            limit (int, optional): Limit of results. Defaults to 500.
            
        Returns:
            List of klines data.
        """
        if not self.client:
            self.connect()
        
        try:
            return self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )
        except Exception as e:
            return [['error', str(e)]]
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Returns:
            Dict containing exchange information.
        """
        if not self.client:
            self.connect()
        
        try:
            return self.client.get_exchange_info()
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def install():
        """
        Install required dependencies for the Binance module.
        
        Returns:
            Dict containing installation results.
        """
        try:
            import subprocess
            result = subprocess.run(
                ['pip', 'install', 'python-binance'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'message': 'Successfully installed python-binance',
                    'details': result.stdout
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to install python-binance',
                    'error': result.stderr
                }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error during installation: {str(e)}'
            }
    
    def forward(self, fn_name: str = 'get_ticker_price', *args, **kwargs):
        """
        Dynamically call a method of the class.
        
        Args:
            fn_name (str): Name of the method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
            
        Returns:
            Result of the called method
        """
        if hasattr(self, fn_name):
            method = getattr(self, fn_name)
            if callable(method):
                return method(*args, **kwargs)
            return method
        else:
            return {'error': f'Method {fn_name} not found'}
