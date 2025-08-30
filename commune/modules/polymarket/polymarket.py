import commune as c
import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.constants import POLYGON

# Load environment variables
load_dotenv()

class Polymarket(c.Module):
    """
    A commune module for interacting with Polymarket prediction markets.
    Provides methods to get market info, list markets, get prices, and historical data.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Polymarket module with CLOB client configuration.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        # Initialize CLOB client configuration
        self.host = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
        self.key = os.getenv("KEY")  # Private key exported from polymarket UI
        self.funder = os.getenv("FUNDER")  # Funder address from polymarket UI
        self.chain_id = POLYGON
        
        # Initialize the client
        self._client = None
        
    def get_client(self) -> ClobClient:
        """Get or create the CLOB client instance."""
        if self._client is None:
            self._client = ClobClient(
                self.host,
                key=self.key,
                chain_id=self.chain_id,
                funder=self.funder,
                signature_type=1,
            )
            self._client.set_api_creds(self._client.create_or_derive_api_creds())
        return self._client
    
    def get_market_info(self, market_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific prediction market.
        
        Args:
            market_id: Market ID or slug
            
        Returns:
            Dictionary containing market information
        """
        try:
            client = self.get_client()
            market_data = client.get_market(market_id)
            
            if not market_data or not isinstance(market_data, dict):
                return {"error": "No market information available"}
                
            return {
                "condition_id": market_data.get('condition_id', 'N/A'),
                "title": market_data.get('title', 'N/A'),
                "status": market_data.get('status', 'N/A'),
                "resolution_date": market_data.get('resolution_date', 'N/A'),
                "raw_data": market_data
            }
        except Exception as e:
            return {"error": f"Failed to get market info: {str(e)}"}
    
    def list_markets(self, status: Optional[str] = None, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get a list of prediction markets with optional filters.
        
        Args:
            status: Filter by market status (e.g., 'active', 'resolved')
            limit: Number of markets to return (default: 10)
            offset: Number of markets to skip (for pagination)
            
        Returns:
            List of market dictionaries
        """
        try:
            client = self.get_client()
            markets_data = client.get_markets()
            
            # Handle string response (if the response is a JSON string)
            if isinstance(markets_data, str):
                try:
                    markets_data = json.loads(markets_data)
                except json.JSONDecodeError:
                    return [{"error": "Invalid response format from API"}]
            
            # Ensure we have a list of markets
            if not isinstance(markets_data, list):
                if isinstance(markets_data, dict) and 'data' in markets_data:
                    markets_data = markets_data['data']
                else:
                    return [{"error": "Unexpected response format from API"}]
            
            # Filter by status if specified
            if status:
                markets_data = [
                    market for market in markets_data 
                    if isinstance(market, dict) and market.get('status', '').lower() == status.lower()
                ]
            
            # Apply pagination
            markets_data = markets_data[offset:offset + limit]
            
            # Format the results
            formatted_markets = []
            for market in markets_data:
                try:
                    volume = float(market.get('volume', 0))
                    volume_str = f"${volume:,.2f}"
                except (ValueError, TypeError):
                    volume_str = f"${market.get('volume', 0)}"
                    
                formatted_markets.append({
                    "condition_id": market.get('condition_id', 'N/A'),
                    "description": market.get('description', 'N/A'),
                    "category": market.get('category', 'N/A'),
                    "question": market.get('question', 'N/A'),
                    "active": market.get('active', 'N/A'),
                    "rewards": market.get('rewards', 'N/A'),
                    "closed": market.get('closed', 'N/A'),
                    "slug": market.get('market_slug', 'N/A'),
                    "min_incentive_size": market.get('min_incentive_size', 'N/A'),
                    "max_incentive_spread": market.get('max_incentive_spread', 'N/A'),
                    "end_date": market.get('end_date_iso', 'N/A'),
                    "start_time": market.get('game_start_time', 'N/A'),
                    "min_order_size": market.get('minimum_order_size', 'N/A'),
                    "min_tick_size": market.get('minimum_tick_size', 'N/A'),
                    "volume": volume_str
                })
            
            return formatted_markets
            
        except Exception as e:
            return [{"error": f"Failed to list markets: {str(e)}"}]
    
    def get_market_prices(self, market_id: str) -> Dict[str, Any]:
        """
        Get current prices and trading information for a market.
        
        Args:
            market_id: Market ID or slug
            
        Returns:
            Dictionary containing price information
        """
        try:
            client = self.get_client()
            market_data = client.get_market(market_id)
            
            if not market_data or not isinstance(market_data, dict):
                return {"error": "No market data available"}
                
            return {
                "title": market_data.get('title', 'Unknown Market'),
                "current_price": market_data.get('current_price', 'N/A'),
                "raw_data": market_data
            }
        except Exception as e:
            return {"error": f"Failed to get market prices: {str(e)}"}
    
    def get_market_history(self, market_id: str, timeframe: str = "7d") -> Dict[str, Any]:
        """
        Get historical price and volume data for a market.
        
        Args:
            market_id: Market ID or slug
            timeframe: Time period for historical data ('1d', '7d', '30d', 'all')
            
        Returns:
            Dictionary containing historical data
        """
        try:
            client = self.get_client()
            # Note: This is a simplified version. The actual CLOB client might have
            # different methods for historical data
            market_data = client.get_market(market_id)
            
            if not market_data or not isinstance(market_data, dict):
                return {"error": "No historical data available"}
                
            return {
                "title": market_data.get('title', 'Unknown Market'),
                "timeframe": timeframe,
                "history": market_data.get('history', []),
                "raw_data": market_data
            }
        except Exception as e:
            return {"error": f"Failed to get market history: {str(e)}"}
    
    def forward(self, action: str = 'list_markets', **kwargs):
        """
        Main entry point for the module. Routes to specific methods based on action.
        
        Args:
            action: The action to perform ('get_market_info', 'list_markets', 
                   'get_market_prices', 'get_market_history')
            **kwargs: Arguments to pass to the specific method
            
        Returns:
            Result from the called method
        """
        actions = {
            'get_market_info': self.get_market_info,
            'list_markets': self.list_markets,
            'get_market_prices': self.get_market_prices,
            'get_market_history': self.get_market_history
        }
        
        if action not in actions:
            return {"error": f"Unknown action: {action}. Available actions: {list(actions.keys())}"}
        
        method = actions[action]
        
        # Extract method-specific parameters
        if action == 'get_market_info':
            return method(market_id=kwargs.get('market_id'))
        elif action == 'list_markets':
            return method(
                status=kwargs.get('status'),
                limit=kwargs.get('limit', 10),
                offset=kwargs.get('offset', 0)
            )
        elif action == 'get_market_prices':
            return method(market_id=kwargs.get('market_id'))
        elif action == 'get_market_history':
            return method(
                market_id=kwargs.get('market_id'),
                timeframe=kwargs.get('timeframe', '7d')
            )
    
    def test(self):
        """
        Test the module functionality.
        """
        c.print("Testing Polymarket module...", color="cyan")
        
        # Test listing markets
        c.print("\n1. Testing list_markets:", color="blue")
        markets = self.list_markets(limit=3)
        c.print(f"Found {len(markets)} markets", color="green")
        if markets and not markets[0].get('error'):
            c.print(f"First market: {markets[0].get('description', 'N/A')}", color="green")
        
        # Test getting market info if we have markets
        if markets and len(markets) > 0 and not markets[0].get('error'):
            market_id = markets[0].get('condition_id')
            if market_id and market_id != 'N/A':
                c.print(f"\n2. Testing get_market_info for market {market_id}:", color="blue")
                info = self.get_market_info(market_id)
                c.print(f"Market title: {info.get('title', 'N/A')}", color="green")
                
                c.print(f"\n3. Testing get_market_prices for market {market_id}:", color="blue")
                prices = self.get_market_prices(market_id)
                c.print(f"Current price: {prices.get('current_price', 'N/A')}", color="green")
        
        c.print("\nTesting complete!", color="cyan")
        return {"status": "tests_completed"}