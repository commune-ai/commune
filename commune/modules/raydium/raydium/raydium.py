import time
import math
import requests
from typing import Dict, List, Optional, Tuple, Any
import json
import commune as c
class RaydiumScraper:
    """
    Enhanced Raydium price scraper for AMM + CLMM pools.

    Features
    --------
    - Symbol or mint inputs (e.g., 'SOL' or mint address)
    - Lists all Raydium pools between two tokens (AMM + CLMM when available)
    - Returns per-pool price, reserve data, fee info, and a best-price helper
    - Simple in-memory caching + retry logic
    - Enhanced error handling and validation
    - Better pool sorting and filtering

    Notes
    -----
    - Public endpoints used:
        * Token list: https://token.jup.ag/all
        * AMM pools:  https://api.raydium.io/v2/main/pairs
        * CLMM pools: https://api.raydium.io/v2/ammV3/ammPools (if available)
    - Endpoints can evolve; this class tries to be defensive and won't break if CLMM is unavailable.
    """

    JUP_TOKENLIST = "https://token.jup.ag/all"
    RAY_AMM_POOLS = "https://api.raydium.io/v2/main/pairs"
    RAY_CLMM_POOLS = "https://api.raydium.io/v2/ammV3/ammPools"

    def __init__(self, timeout: int = 15, max_retries: int = 3, backoff: float = 0.8):
        self.sess = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

        self._tokenlist_cache: Dict[str, Any] = {}
        self._amm_cache: Dict[str, Any] = {}
        self._clmm_cache: Dict[str, Any] = {}

        # symbol -> mint and mintokenlistt -> token meta
        self._symbol_to_mint: Dict[str, str] = {}
        self._mint_meta: Dict[str, Dict[str, Any]] = {}

        self.store = c.mod('store')('~/.commune/raydium')


    # -----------------------------
    # Low-level helpers
    # -----------------------------

    def _get(self, url: str) -> Any:
        """Fetch data from URL with retry logic."""
        last_err = None
        for i in range(self.max_retries):
            try:
                r = self.sess.get(url, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if i < self.max_retries - 1:
                    time.sleep(self.backoff * (2 ** i))
        raise RuntimeError(f"GET {url} failed after {self.max_retries} retries: {last_err}")

    def tokens(self, search=None, n=None, update=False) -> List[Dict[str, Any]]:
        """Load token list from Jupiter API."""
        data = self.store.get('raydium_tokenlist.json', None, update=update)
        if data == None:
            data = self._get(self.JUP_TOKENLIST)
            self.store.put(path, data)
        try:
            # Jupiter returns a list of tokens with fields like 'address', 'symbol', 'decimals'
            for t in data:
                mint = t.get("address")
                sym = (t.get("symbol") or "").upper()
                if not mint or not sym:
                    continue
                # Handle duplicate symbols by keeping the first one
                if sym not in self._symbol_to_mint:
                    self._symbol_to_mint[sym] = mint

                if search and search.lower() not in sym.lower():
                    continue
                self._mint_meta[mint] = {
                    'address': mint,
                    "symbol": sym,
                    "decimals": t.get("decimals", 0),
                    "name": t.get("name", ""),
                    "logoURI": t.get("logoURI", "")
                }
            self._tokenlist_cache = {"loaded": True, "count": len(self._mint_meta), "timestamp": time.time()}
        except Exception as e:
            print(f"Warning: Failed to load token list: {e}")
            self._tokenlist_cache = {"loaded": False, "error": str(e)}

        result = list(self._mint_meta.values())
        if search:
            result = [t for t in result if search.lower() in t["symbol"].lower() or search.lower() in t["name"].lower()]
        if n is not None:
            result = result[:n]
        return result

    def num_tokens(self) -> int:
        """Return number of tokens in the token list."""
        if not self._tokenlist_cache:
            self.tokens()
        return len(self._mint_meta)

    def _ensure_mint(self, token: str) -> str:
        """Accepts 'SOL' or a mint address, returns the mint address."""
        if not token:
            raise ValueError("Empty token")
        token = token.strip()
        # Check if it looks like a mint address (base58, 32-44 chars)
        if len(token) > 20 and all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in token):
            return token
        sym = token.upper()
        if sym in self._symbol_to_mint:
            return self._symbol_to_mint[sym]
        # If unknown, assume it's a mint already
        return token

    def _load_amm_pools(self):
        path = 'raydium_amm_pools.json'
        pools =self.store.get(path, None)
        if pools == None:

            """Load AMM pools from Raydium API."""
            if self._amm_cache and time.time() - self._amm_cache.get("timestamp", 0) < 300:  # 5 min cache
                return
            try:
                pools = self._get(self.RAY_AMM_POOLS)
                self.store.put('raydium_amm_pools.json', self._amm_cache)
            except Exception as e:
                print(f"Warning: Failed to load AMM pools: {e}")
                pools = []

        # Normalize a bit
        norm = []
        for p in pools or []:
            # Common AMM fields seen historically:
            # 'ammId','lpMint','baseMint','quoteMint','price','market','baseReserve','quoteReserve','feeRate'
            norm.append({
                "type": "AMM",
                "id": p.get("ammId") or p.get("id") or p.get("lpMint"),
                "lpMint": p.get("lpMint"),
                "baseMint": p.get("baseMint"),
                "quoteMint": p.get("quoteMint"),
                "price_hint": p.get("price"),
                "baseReserve": p.get("baseReserve"),
                "quoteReserve": p.get("quoteReserve"),
                "feeRate": p.get("feeRate"),
                "liquidity": p.get("liquidity", 0),
                "volume24h": p.get("volume24h", 0),
                "raw": p,
            })
        self._amm_cache = {"pools": norm, "count": len(norm), "timestamp": time.time()}

    def _load_clmm_pools(self):
        """Load CLMM pools from Raydium API."""
        path = 'raydium_clmm_pools.json'
        pools = self.store.get(path, None)
        if pools is None:
            if self._clmm_cache and time.time() - self._clmm_cache.get("timestamp", 0) < 300:  # 5 min cache
                return
            print("Loading CLMM pools from Raydium API...")
            pools = self._get(self.RAY_CLMM_POOLS)
            self.store.put(path, pools)
        
        norm = []
        for p in pools or []:
            # Typical CLMM fields (can vary): 'id','mintA','mintB','price','feeRate','sqrtPriceX64','liquidity'
            try:
                norm.append({
                    "type": "CLMM",
                    "id": p.get("id"),
                    "lpMint": p.get("lpMint"),
                    "baseMint": p.get("mintA") or p.get("baseMint"),
                    "quoteMint": p.get("mintB") or p.get("quoteMint"),
                    "price_hint": p.get("price"),
                    "feeRate": p.get("feeRate"),
                    "sqrtPriceX64": p.get("sqrtPriceX64"),
                    "liquidity": p.get("liquidity", 0),
                    "volume24h": p.get("volume24h", 0),
                    "raw": p,
                })
            except Exception as e:
                continue
        self._clmm_cache = {"pools": norm, "count": len(norm), "timestamp": time.time()}

    def list_pools(
        self,
        token_a: str,
        token_b: str,
        include_amm: bool = True,
        include_clmm: bool = True,
        min_liquidity: float = 0,
    ) -> List[Dict[str, Any]]:
        """Return all Raydium pools that match token pair (in either order)."""
        a = self._ensure_mint(token_a)
        b = self._ensure_mint(token_b)

        if include_amm:
            self._load_amm_pools()
        if include_clmm:
            self._load_clmm_pools()

        pools: List[Dict[str, Any]] = []
        universe = []
        if include_amm and self._amm_cache:
            universe += self._amm_cache["pools"]
        if include_clmm and self._clmm_cache:
            universe += self._clmm_cache["pools"]

        for p in universe:
            m0 = p.get("baseMint")
            m1 = p.get("quoteMint")
            if not m0 or not m1:
                continue
            if ({m0, m1} == {a, b}):
                # Filter by minimum liquidity
                try:
                    liq = float(p.get("liquidity", 0))
                    if liq >= min_liquidity:
                        pools.append(p)
                except (ValueError, TypeError):
                    pools.append(p)  # Include if liquidity can't be parsed

        # Sort by type preference (CLMM first often has tighter pricing), then by liquidity if available
        def sort_key(p):
            t = 0 if p["type"] == "CLMM" else 1
            try:
                liq = float(p.get("liquidity", 0))
            except (ValueError, TypeError):
                liq = 0
            try:
                vol = float(p.get("volume24h", 0))
            except (ValueError, TypeError):
                vol = 0
            return (t, -liq, -vol)

        pools.sort(key=sort_key)
        return pools

    def _mint_to_symbol(self, mint: str) -> str:
        """Convert mint address to symbol."""
        return self._mint_meta.get(mint, {}).get("symbol", mint[:6] + "…")

    def _price_from_amm(self, pool: Dict[str, Any], want_mint_out: str) -> Optional[float]:
        """
        AMM price estimate: try 'price' hint, else ratio of reserves.
        Raydium's 'price' is typically base→quote. We adapt to desired direction.
        """
        base = pool.get("baseMint")
        quote = pool.get("quoteMint")
        ph = pool.get("price_hint")

        def as_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if ph is not None:
            ph = as_float(ph)

        if ph is None:
            br = as_float(pool.get("baseReserve"))
            qr = as_float(pool.get("quoteReserve"))
            if br and qr and br > 0:
                ph = qr / br
            else:
                return None

        if ph <= 0:
            return None

        if want_mint_out == quote:
            # price of 1 base in quote
            return ph
        elif want_mint_out == base:
            # price of 1 quote in base
            return 1.0 / ph
        return None

    def _price_from_clmm(self, pool: Dict[str, Any], want_mint_out: str) -> Optional[float]:
        """
        CLMM price estimate:
        - Prefer 'price' if present (assume mintA -> mintB).
        - Else approximate from sqrtPriceX64 if available: price ~ (sqrtP^2 / 2^128)
          where sqrtP is in Q64.64 fixed-point for mintA/mintB.
        """
        a = pool.get("baseMint")
        b = pool.get("quoteMint")
        ph = pool.get("price_hint")

        def as_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if ph is not None:
            ph = as_float(ph)
        else:
            sp = pool.get("sqrtPriceX64")
            if sp is not None:
                try:
                    sp = float(sp)
                    ph = (sp * sp) / (2 ** 128)
                except Exception:
                    ph = None

        if ph is None or ph <= 0:
            return None

        if want_mint_out == b:
            # price of 1 A in B
            return ph
        elif want_mint_out == a:
            return 1.0 / ph
        return None

    def quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: float = 1.0,
        pool_id: Optional[str] = None,
        include_amm: bool = True,
        include_clmm: bool = True,
        slippage: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """
        Return price quotes for token_in -> token_out across Raydium pools.
        If pool_id is given, only quote that pool (if found).
        Output fields:
            - poolType
            - poolId
            - tokenIn / tokenOut (symbol)
            - mintIn / mintOut
            - price (units of token_out per 1 token_in)
            - amountOut (expected output for given input)
            - feeRate (if available)
            - priceImpact (estimated)
        """
        mint_in = self._ensure_mint(token_in)
        mint_out = self._ensure_mint(token_out)

        pools = self.list_pools(mint_in, mint_out, include_amm, include_clmm)
        if pool_id:
            pools = [p for p in pools if p.get("id") == pool_id]

        quotes = []
        for p in pools:
            if p["type"] == "AMM":
                price = self._price_from_amm(p, mint_out)
            else:
                price = self._price_from_clmm(p, mint_out)

            if price is None or price <= 0:
                continue

            # Calculate output amount
            amount_out = amount_in * price
            
            # Estimate price impact (simplified)
            liquidity = float(p.get("liquidity", 0) or 0)
            price_impact = 0.0
            if liquidity > 0:
                # Very rough estimate: impact ~ amount / liquidity
                price_impact = min((amount_in * price) / liquidity * 100, 100.0)

            quotes.append({
                "poolType": p["type"],
                "poolId": p.get("id"),
                "tokenIn": self._mint_to_symbol(mint_in),
                "tokenOut": self._mint_to_symbol(mint_out),
                "mintIn": mint_in,
                "mintOut": mint_out,
                "price": price,
                "amountIn": amount_in,
                "amountOut": amount_out,
                "amountOutMin": amount_out * (1 - slippage),
                "feeRate": p.get("feeRate"),
                "liquidity": liquidity,
                "volume24h": float(p.get("volume24h", 0) or 0),
                "priceImpact": price_impact,
            })

        # Deduplicate by poolId (some APIs can return shadows)
        seen = set()
        uniq = []
        for q in quotes:
            k = (q["poolId"], q["poolType"])
            if k not in seen:
                seen.add(k)
                uniq.append(q)
        # Best price first
        uniq.sort(key=lambda x: x["price"], reverse=True)
        return uniq

    def best_price(
        self,
        token_in: str,
        token_out: str,
        amount_in: float = 1.0,
        include_amm: bool = True,
        include_clmm: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Convenience: return the best single pool quote for token_in -> token_out."""
        qs = self.quote(token_in, token_out, amount_in, None, include_amm, include_clmm)
        return qs[0] if qs else None

    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token metadata by symbol or mint address."""
        mint = self._ensure_mint(token)
        return self._mint_meta.get(mint)

    def search_tokens(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tokens by symbol or name."""
        query = query.upper()
        results = []
        
        for mint, meta in self._mint_meta.items():
            symbol = meta.get("symbol", "").upper()
            name = meta.get("name", "").upper()
            
            if query in symbol or query in name:
                results.append({
                    "mint": mint,
                    "symbol": meta.get("symbol"),
                    "name": meta.get("name"),
                    "decimals": meta.get("decimals"),
                    "logoURI": meta.get("logoURI"),
                })
                
                if len(results) >= limit:
                    break
        
        return results

    def get_pool_stats(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a specific pool."""
        self._load_amm_pools()
        self._load_clmm_pools()
        
        all_pools = []
        if self._amm_cache:
            all_pools.extend(self._amm_cache["pools"])
        if self._clmm_cache:
            all_pools.extend(self._clmm_cache["pools"])
        
        for p in all_pools:
            if p.get("id") == pool_id:
                return {
                    "poolId": p.get("id"),
                    "poolType": p["type"],
                    "baseMint": p.get("baseMint"),
                    "quoteMint": p.get("quoteMint"),
                    "baseSymbol": self._mint_to_symbol(p.get("baseMint", "")),
                    "quoteSymbol": self._mint_to_symbol(p.get("quoteMint", "")),
                    "liquidity": float(p.get("liquidity", 0) or 0),
                    "volume24h": float(p.get("volume24h", 0) or 0),
                    "feeRate": p.get("feeRate"),
                    "baseReserve": p.get("baseReserve"),
                    "quoteReserve": p.get("quoteReserve"),
                    "price": p.get("price_hint"),
                }
        return None

    def run(self):
        """Example usage of the scraper."""
        scraper = RaydiumScraper()

        print("\n=== Raydium Scraper Demo ===")
        
        # 1) List pools and get per-pool quotes (example: SOL -> USDC)
        print("\n1. Getting quotes for SOL -> USDC:")
        quotes = scraper.quote("SOL", "USDC", amount_in=10)
        for i, q in enumerate(quotes[:5]):
            print(f"   Pool {i+1}: {q['poolType']} - Price: ${q['price']:.4f}, Output: {q['amountOut']:.2f} USDC")

        # 2) Best single-pool price
        print("\n2. Best price for SOL -> USDC:")
        best = scraper.best_price("SOL", "USDC", amount_in=10)
        if best:
            print(f"   Best pool: {best['poolType']} - Price: ${best['price']:.4f}")
            print(f"   10 SOL = {best['amountOut']:.2f} USDC")
            print(f"   Pool ID: {best['poolId']}")

        # 3) Search for tokens
        print("\n3. Searching for 'RAY' tokens:")
        ray_tokens = scraper.search_tokens("RAY", limit=3)
        for t in ray_tokens:
            print(f"   {t['symbol']} - {t['name']} ({t['mint'][:8]}...)")

        # 4) Get token info
        print("\n4. Getting SOL token info:")
        sol_info = scraper.get_token_info("SOL")
        if sol_info:
            print(f"   Symbol: {sol_info['symbol']}")
            print(f"   Name: {sol_info['name']}")
            print(f"   Decimals: {sol_info['decimals']}")

        # 5) Pool statistics
        if best:
            print(f"\n5. Getting stats for best pool:")
            stats = scraper.get_pool_stats(best['poolId'])
            if stats:
                print(f"   Liquidity: ${stats['liquidity']:,.2f}")
                print(f"   24h Volume: ${stats['volume24h']:,.2f}")
                print(f"   Fee Rate: {stats['feeRate']}%")

    @staticmethod
    def test():
        """Comprehensive test function for RaydiumScraper."""
        print("\n=== Testing RaydiumScraper ===")
        
        try:
            # Initialize scraper
            scraper = RaydiumScraper(timeout=10, max_retries=2)
            print("✓ Scraper initialized successfully")
            
            # Test 1: Token resolution
            print("\n--- Test 1: Token Resolution ---")
            test_tokens = ["SOL", "USDC", "RAY"]
            for token in test_tokens:
                mint = scraper._ensure_mint(token)
                print(f"✓ {token} -> {mint[:8]}...")
            
            # Test 2: Pool listing
            print("\n--- Test 2: Pool Listing ---")
            pools = scraper.list_pools("SOL", "USDC")
            print(f"✓ Found {len(pools)} pools for SOL/USDC")
            if pools:
                print(f"  - AMM pools: {sum(1 for p in pools if p['type'] == 'AMM')}")
                print(f"  - CLMM pools: {sum(1 for p in pools if p['type'] == 'CLMM')}")
            
            # Test 3: Price quotes
            print("\n--- Test 3: Price Quotes ---")
            quotes = scraper.quote("SOL", "USDC", amount_in=1.0)
            print(f"✓ Got {len(quotes)} price quotes")
            if quotes:
                best_quote = quotes[0]
                print(f"  - Best price: ${best_quote['price']:.4f}")
                print(f"  - Pool type: {best_quote['poolType']}")
                print(f"  - Expected output: {best_quote['amountOut']:.4f} USDC")
            
            # Test 4: Best price helper
            print("\n--- Test 4: Best Price Helper ---")
            best = scraper.best_price("RAY", "USDC")
            if best:
                print(f"✓ Best RAY->USDC price: ${best['price']:.4f}")
            else:
                print("✗ No RAY/USDC pools found")
            
            # Test 5: Token search
            print("\n--- Test 5: Token Search ---")
            search_results = scraper.search_tokens("USD", limit=5)
            print(f"✓ Found {len(search_results)} tokens matching 'USD'")
            for token in search_results[:3]:
                print(f"  - {token['symbol']}: {token['name']}")
            
            # Test 6: Token info
            print("\n--- Test 6: Token Info ---")
            info = scraper.get_token_info("SOL")
            if info:
                print(f"✓ SOL info: {info['name']} (decimals: {info['decimals']})")
            
            # Test 7: Pool statistics
            print("\n--- Test 7: Pool Statistics ---")
            if quotes:
                pool_id = quotes[0]['poolId']
                stats = scraper.get_pool_stats(pool_id)
                if stats:
                    print(f"✓ Pool stats for {pool_id[:8]}...")
                    print(f"  - Liquidity: ${stats['liquidity']:,.2f}")
                    print(f"  - 24h Volume: ${stats['volume24h']:,.2f}")
            
            # Test 8: Error handling
            print("\n--- Test 8: Error Handling ---")
            try:
                scraper._ensure_mint("")  # Empty token
            except ValueError:
                print("✓ Correctly handled empty token error")
            
            try:
                quotes = scraper.quote("INVALID_TOKEN_XYZ", "USDC")
                print(f"✓ Handled invalid token gracefully (returned {len(quotes)} quotes)")
            except Exception as e:
                print(f"✗ Unexpected error with invalid token: {e}")
            
            # Test 9: Cache refresh
            print("\n--- Test 9: Cache Refresh ---")
            scraper.refresh("tokens")
            print("✓ Token cache refreshed")
            
            # Test 10: Different token pairs
            print("\n--- Test 10: Various Token Pairs ---")
            test_pairs = [("SOL", "RAY"), ("USDC", "USDT"), ("RAY", "SOL")]
            for token_in, token_out in test_pairs:
                best = scraper.best_price(token_in, token_out)
                if best:
                    print(f"✓ {token_in}->{token_out}: ${best['price']:.6f}")
                else:
                    print(f"✗ No pools found for {token_in}/{token_out}")
            
            print("\n=== All tests completed! ===")
            return True
            
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Run the demo
    scraper = RaydiumScraper()
    scraper.run()
    
    # Run comprehensive tests
    print("\n" + "="*50)
    test_raydium_scraper()
