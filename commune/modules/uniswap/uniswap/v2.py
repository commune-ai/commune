import requests
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from decimal import Decimal
import hashlib


class UniswapV2Python:
    """
    Python implementation of Uniswap V2-style AMM with similar features to RaydiumScraper.
    
    Features:
    - Token resolution by symbol or address
    - Pool discovery between token pairs
    - Price quotes and best price routing
    - Liquidity pool statistics
    - Simple caching mechanism
    - Error handling and retries
    """
    
    # Mock endpoints - in production these would be actual Uniswap/DEX APIs
    TOKEN_LIST_URL = "https://tokens.coingecko.com/uniswap/all.json"
    POOL_API_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"  # Example GraphQL endpoint
    
    def __init__(self, timeout: int = 15, max_retries: int = 3, backoff: float = 0.8):
        self.sess = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        
        # Caches
        self._token_cache: Dict[str, Any] = {}
        self._pool_cache: Dict[str, Any] = {}
        self._price_cache: Dict[str, Any] = {}
        
        # Token mappings
        self._symbol_to_address: Dict[str, str] = {}
        self._address_to_token: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with common tokens
        self._init_common_tokens()
    
    def _init_common_tokens(self):
        """Initialize with common ERC20 tokens"""
        common_tokens = {
            "ETH": {
                "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                "symbol": "WETH",
                "decimals": 18,
                "name": "Wrapped Ether"
            },
            "USDC": {
                "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "symbol": "USDC",
                "decimals": 6,
                "name": "USD Coin"
            },
            "USDT": {
                "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "symbol": "USDT",
                "decimals": 6,
                "name": "Tether USD"
            },
            "DAI": {
                "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                "symbol": "DAI",
                "decimals": 18,
                "name": "Dai Stablecoin"
            }
        }
        
        for symbol, data in common_tokens.items():
            self._symbol_to_address[symbol] = data["address"]
            self._address_to_token[data["address"]] = data
    
    def _get(self, url: str, headers: Optional[Dict] = None) -> Any:
        """Fetch data with retry logic"""
        last_err = None
        for i in range(self.max_retries):
            try:
                r = self.sess.get(url, timeout=self.timeout, headers=headers)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if i < self.max_retries - 1:
                    time.sleep(self.backoff * (2 ** i))
        raise RuntimeError(f"GET {url} failed after {self.max_retries} retries: {last_err}")
    
    def load_tokens(self, update: bool = False) -> List[Dict[str, Any]]:
        """Load token list from external source or cache"""
        cache_key = "token_list"
        
        if not update and cache_key in self._token_cache:
            return self._token_cache[cache_key]
        
        try:
            # Try to load from CoinGecko's Uniswap token list
            data = self._get(self.TOKEN_LIST_URL)
            tokens = data.get("tokens", [])
            
            for token in tokens:
                address = token.get("address")
                symbol = token.get("symbol", "").upper()
                
                if address and symbol:
                    self._symbol_to_address[symbol] = address
                    self._address_to_token[address] = {
                        "address": address,
                        "symbol": symbol,
                        "decimals": token.get("decimals", 18),
                        "name": token.get("name", ""),
                        "logoURI": token.get("logoURI", "")
                    }
            
            self._token_cache[cache_key] = list(self._address_to_token.values())
            return self._token_cache[cache_key]
            
        except Exception as e:
            print(f"Warning: Failed to load external token list: {e}")
            # Return cached common tokens
            return list(self._address_to_token.values())
    
    def resolve_token(self, token: str) -> str:
        """Convert symbol to address or validate address"""
        if not token:
            raise ValueError("Empty token")
        
        token = token.strip()
        
        # Check if it's already an address (0x + 40 hex chars)
        if token.startswith("0x") and len(token) == 42:
            return token
        
        # Try to resolve as symbol
        symbol = token.upper()
        if symbol in self._symbol_to_address:
            return self._symbol_to_address[symbol]
        
        # If not found, assume it's an address
        return token
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token metadata by symbol or address"""
        address = self.resolve_token(token)
        return self._address_to_token.get(address)
    
    def calculate_price(self, reserve_in: float, reserve_out: float) -> float:
        """Calculate price from reserves using constant product formula"""
        if reserve_in <= 0:
            return 0
        return reserve_out / reserve_in
    
    def calculate_output_amount(
        self,
        amount_in: float,
        reserve_in: float,
        reserve_out: float,
        fee: float = 0.003
    ) -> float:
        """Calculate output amount using Uniswap V2 formula"""
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return 0
        
        amount_in_with_fee = amount_in * (1 - fee)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in + amount_in_with_fee
        
        return numerator / denominator
    
    def create_mock_pool(self, token_a: str, token_b: str, pool_type: str = "V2") -> Dict[str, Any]:
        """Create a mock pool for demonstration (in production, fetch from blockchain)"""
        # Generate deterministic pool ID
        pool_id = hashlib.md5(f"{token_a}_{token_b}_{pool_type}".encode()).hexdigest()[:16]
        
        # Mock reserves (in production, fetch from smart contract)
        base_liquidity = 1000000  # Base liquidity in USD
        price_ratio = 1.0  # Mock price ratio
        
        if "USD" in token_a or "USD" in token_b:
            price_ratio = 1.0
        elif "ETH" in token_a or "ETH" in token_b:
            price_ratio = 2000.0  # Mock ETH price
        
        return {
            "type": pool_type,
            "id": pool_id,
            "token0": token_a,
            "token1": token_b,
            "reserve0": base_liquidity,
            "reserve1": base_liquidity / price_ratio,
            "fee": 0.003,  # 0.3% fee
            "liquidity": base_liquidity * 2,
            "volume24h": base_liquidity * 0.1,  # Mock 10% daily volume
        }
    
    def find_pools(
        self,
        token_a: str,
        token_b: str,
        min_liquidity: float = 0
    ) -> List[Dict[str, Any]]:
        """Find all pools for a token pair"""
        addr_a = self.resolve_token(token_a)
        addr_b = self.resolve_token(token_b)
        
        # In production, query The Graph or blockchain for actual pools
        # For demo, create mock pools
        pools = []
        
        # V2 pool
        v2_pool = self.create_mock_pool(addr_a, addr_b, "V2")
        if v2_pool["liquidity"] >= min_liquidity:
            pools.append(v2_pool)
        
        # V3 pool (with different fee tiers)
        for fee_tier in [0.0005, 0.003, 0.01]:  # 0.05%, 0.3%, 1%
            v3_pool = self.create_mock_pool(addr_a, addr_b, "V3")
            v3_pool["fee"] = fee_tier
            v3_pool["id"] += f"_fee{int(fee_tier * 10000)}"
            if v3_pool["liquidity"] >= min_liquidity:
                pools.append(v3_pool)
        
        # Sort by liquidity
        pools.sort(key=lambda p: p["liquidity"], reverse=True)
        return pools
    
    def quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: float = 1.0,
        slippage: float = 0.01
    ) -> List[Dict[str, Any]]:
        """Get price quotes across all pools"""
        addr_in = self.resolve_token(token_in)
        addr_out = self.resolve_token(token_out)
        
        pools = self.find_pools(addr_in, addr_out)
        quotes = []
        
        for pool in pools:
            # Determine which token is token0/token1
            if pool["token0"] == addr_in:
                reserve_in = pool["reserve0"]
                reserve_out = pool["reserve1"]
            else:
                reserve_in = pool["reserve1"]
                reserve_out = pool["reserve0"]
            
            # Calculate price and output
            price = self.calculate_price(reserve_in, reserve_out)
            amount_out = self.calculate_output_amount(
                amount_in, reserve_in, reserve_out, pool["fee"]
            )
            
            # Estimate price impact
            price_after = self.calculate_price(
                reserve_in + amount_in,
                reserve_out - amount_out
            )
            price_impact = abs(price_after - price) / price * 100
            
            quotes.append({
                "poolType": pool["type"],
                "poolId": pool["id"],
                "tokenIn": self._address_to_token.get(addr_in, {}).get("symbol", addr_in[:6]),
                "tokenOut": self._address_to_token.get(addr_out, {}).get("symbol", addr_out[:6]),
                "addressIn": addr_in,
                "addressOut": addr_out,
                "price": price,
                "amountIn": amount_in,
                "amountOut": amount_out,
                "amountOutMin": amount_out * (1 - slippage),
                "fee": pool["fee"],
                "liquidity": pool["liquidity"],
                "volume24h": pool["volume24h"],
                "priceImpact": price_impact,
                "route": [addr_in, addr_out]
            })
        
        # Sort by best output amount
        quotes.sort(key=lambda q: q["amountOut"], reverse=True)
        return quotes
    
    def best_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Get the best quote for a token swap"""
        quotes = self.quote(token_in, token_out, amount_in)
        return quotes[0] if quotes else None
    
    def search_tokens(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tokens by symbol or name"""
        query = query.upper()
        results = []
        
        for address, token in self._address_to_token.items():
            symbol = token.get("symbol", "").upper()
            name = token.get("name", "").upper()
            
            if query in symbol or query in name:
                results.append(token)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_pool_stats(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a specific pool"""
        # In production, fetch from blockchain
        # For demo, return mock stats
        return {
            "poolId": pool_id,
            "tvl": 2000000,  # $2M TVL
            "volume24h": 200000,  # $200k daily volume
            "volume7d": 1400000,  # $1.4M weekly volume
            "fees24h": 600,  # $600 daily fees
            "apy": 12.5,  # 12.5% APY
            "priceChange24h": 2.3,  # 2.3% price change
        }
    
    def simulate_swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        max_slippage: float = 0.01
    ) -> Dict[str, Any]:
        """Simulate a token swap with detailed execution info"""
        best = self.best_quote(token_in, token_out, amount_in)
        
        if not best:
            return {
                "success": False,
                "error": "No liquidity pools found for this pair"
            }
        
        # Simulate execution
        execution_price = best["amountOut"] / amount_in
        
        return {
            "success": True,
            "poolId": best["poolId"],
            "route": best["route"],
            "amountIn": amount_in,
            "amountOut": best["amountOut"],
            "executionPrice": execution_price,
            "priceImpact": best["priceImpact"],
            "fee": best["fee"],
            "minimumReceived": best["amountOutMin"],
            "gasEstimate": 150000,  # Mock gas estimate
        }
    
    def run_demo(self):
        """Demo the Uniswap implementation"""
        print("\n=== Python Uniswap V2 Demo ===")
        
        # Load tokens
        print("\n1. Loading tokens...")
        tokens = self.load_tokens()
        print(f"   Loaded {len(self._address_to_token)} tokens")
        
        # Search tokens
        print("\n2. Searching for 'USD' tokens:")
        usd_tokens = self.search_tokens("USD", limit=5)
        for token in usd_tokens:
            print(f"   {token['symbol']} - {token['name']}")
        
        # Get quotes
        print("\n3. Getting quotes for ETH -> USDC:")
        quotes = self.quote("ETH", "USDC", amount_in=1.0)
        for i, quote in enumerate(quotes[:3]):
            print(f"   Pool {i+1}: {quote['poolType']} - Price: ${quote['price']:.2f}, Output: {quote['amountOut']:.2f} USDC")
        
        # Best quote
        print("\n4. Best quote for ETH -> USDC:")
        best = self.best_quote("ETH", "USDC", amount_in=1.0)
        if best:
            print(f"   Best pool: {best['poolType']} (fee: {best['fee']*100}%)")
            print(f"   1 ETH = {best['amountOut']:.2f} USDC")
            print(f"   Price impact: {best['priceImpact']:.2f}%")
        
        # Simulate swap
        print("\n5. Simulating swap of 10 ETH -> USDC:")
        swap = self.simulate_swap("ETH", "USDC", 10.0)
        if swap["success"]:
            print(f"   Expected output: {swap['amountOut']:.2f} USDC")
            print(f"   Execution price: ${swap['executionPrice']:.2f}")
            print(f"   Price impact: {swap['priceImpact']:.2f}%")
            print(f"   Minimum received: {swap['minimumReceived']:.2f} USDC")


if __name__ == "__main__":
    # Create instance and run demo
    uniswap = UniswapV2Python()
    uniswap.run_demo()
