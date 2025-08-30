from web3 import Web3
from eth_abi import encode, decode
import json
import math
from decimal import Decimal
from typing import Dict, List, Tuple, Optional
import requests


class UniswapV3Python:
    """
    Python implementation of Uniswap V3 with concentrated liquidity features.
    
    Key V3 Features:
    - Concentrated liquidity positions
    - Multiple fee tiers (0.05%, 0.3%, 1%)
    - Price ranges and ticks
    - Non-fungible liquidity positions
    - Advanced price oracles
    """
    
    # Uniswap V3 Factory and Router addresses on Ethereum mainnet
    FACTORY_ADDRESS = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
    ROUTER_ADDRESS = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
    POSITION_MANAGER_ADDRESS = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
    
    # Fee tiers in basis points (hundredths of a percent)
    FEE_TIERS = {
        "0.05%": 500,    # 0.05% = 500/10000
        "0.3%": 3000,    # 0.3% = 3000/10000
        "1%": 10000      # 1% = 10000/10000
    }
    
    # Tick spacing for each fee tier
    TICK_SPACINGS = {
        500: 10,
        3000: 60,
        10000: 200
    }
    
    def __init__(self, provider_url: str = "https://mainnet.infura.io/v3/YOUR_KEY"):
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        self.chain_id = 1  # Ethereum mainnet
        
        # Load ABIs (simplified versions)
        self.factory_abi = self._get_factory_abi()
        self.pool_abi = self._get_pool_abi()
        self.router_abi = self._get_router_abi()
        
        # Contract instances
        self.factory = self.w3.eth.contract(
            address=self.FACTORY_ADDRESS,
            abi=self.factory_abi
        )
        
    def _get_factory_abi(self) -> List[Dict]:
        """Minimal Factory ABI for pool queries"""
        return [
            {
                "inputs": [
                    {"name": "tokenA", "type": "address"},
                    {"name": "tokenB", "type": "address"},
                    {"name": "fee", "type": "uint24"}
                ],
                "name": "getPool",
                "outputs": [{"name": "pool", "type": "address"}],
                "type": "function"
            }
        ]
    
    def _get_pool_abi(self) -> List[Dict]:
        """Minimal Pool ABI for price and liquidity queries"""
        return [
            {
                "inputs": [],
                "name": "slot0",
                "outputs": [
                    {"name": "sqrtPriceX96", "type": "uint160"},
                    {"name": "tick", "type": "int24"},
                    {"name": "observationIndex", "type": "uint16"},
                    {"name": "observationCardinality", "type": "uint16"},
                    {"name": "observationCardinalityNext", "type": "uint16"},
                    {"name": "feeProtocol", "type": "uint8"},
                    {"name": "unlocked", "type": "bool"}
                ],
                "type": "function"
            },
            {
                "inputs": [],
                "name": "liquidity",
                "outputs": [{"name": "", "type": "uint128"}],
                "type": "function"
            },
            {
                "inputs": [],
                "name": "token0",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function"
            },
            {
                "inputs": [],
                "name": "token1",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function"
            },
            {
                "inputs": [],
                "name": "fee",
                "outputs": [{"name": "", "type": "uint24"}],
                "type": "function"
            },
            {
                "inputs": [{"name": "tick", "type": "int24"}],
                "name": "ticks",
                "outputs": [
                    {"name": "liquidityGross", "type": "uint128"},
                    {"name": "liquidityNet", "type": "int128"},
                    {"name": "feeGrowthOutside0X128", "type": "uint256"},
                    {"name": "feeGrowthOutside1X128", "type": "uint256"},
                    {"name": "tickCumulativeOutside", "type": "int56"},
                    {"name": "secondsPerLiquidityOutsideX128", "type": "uint160"},
                    {"name": "secondsOutside", "type": "uint32"},
                    {"name": "initialized", "type": "bool"}
                ],
                "type": "function"
            }
        ]
    
    def _get_router_abi(self) -> List[Dict]:
        """Minimal Router ABI for swaps"""
        return [
            {
                "inputs": [
                    {
                        "components": [
                            {"name": "tokenIn", "type": "address"},
                            {"name": "tokenOut", "type": "address"},
                            {"name": "fee", "type": "uint24"},
                            {"name": "recipient", "type": "address"},
                            {"name": "deadline", "type": "uint256"},
                            {"name": "amountIn", "type": "uint256"},
                            {"name": "amountOutMinimum", "type": "uint256"},
                            {"name": "sqrtPriceLimitX96", "type": "uint160"}
                        ],
                        "name": "params",
                        "type": "tuple"
                    }
                ],
                "name": "exactInputSingle",
                "outputs": [{"name": "amountOut", "type": "uint256"}],
                "type": "function"
            }
        ]
    
    def get_pool_address(self, token0: str, token1: str, fee: int) -> str:
        """Get the pool address for a token pair and fee tier"""
        # Order tokens
        if token0.lower() > token1.lower():
            token0, token1 = token1, token0
        
        pool_address = self.factory.functions.getPool(token0, token1, fee).call()
        return pool_address
    
    def get_pool_info(self, pool_address: str) -> Dict:
        """Get comprehensive pool information"""
        pool = self.w3.eth.contract(address=pool_address, abi=self.pool_abi)
        
        # Get slot0 data
        slot0 = pool.functions.slot0().call()
        sqrt_price_x96 = slot0[0]
        tick = slot0[1]
        
        # Get other pool data
        liquidity = pool.functions.liquidity().call()
        token0 = pool.functions.token0().call()
        token1 = pool.functions.token1().call()
        fee = pool.functions.fee().call()
        
        # Calculate price from sqrtPriceX96
        price = self.sqrt_price_to_price(sqrt_price_x96)
        
        return {
            "address": pool_address,
            "token0": token0,
            "token1": token1,
            "fee": fee,
            "sqrtPriceX96": sqrt_price_x96,
            "tick": tick,
            "liquidity": liquidity,
            "price": price,
            "price_inverted": 1 / price if price > 0 else 0
        }
    
    def sqrt_price_to_price(self, sqrt_price_x96: int, decimals0: int = 18, decimals1: int = 18) -> float:
        """Convert sqrtPriceX96 to human-readable price"""
        # sqrtPriceX96 = sqrt(price) * 2^96
        # price = (sqrtPriceX96 / 2^96)^2
        price = (sqrt_price_x96 / (2 ** 96)) ** 2
        
        # Adjust for decimals
        decimal_adjustment = 10 ** (decimals1 - decimals0)
        return price * decimal_adjustment
    
    def price_to_tick(self, price: float) -> int:
        """Convert price to tick"""
        return math.floor(math.log(price) / math.log(1.0001))
    
    def tick_to_price(self, tick: int) -> float:
        """Convert tick to price"""
        return 1.0001 ** tick
    
    def get_tick_range(self, current_tick: int, fee_tier: int, range_percent: float = 10) -> Tuple[int, int]:
        """Calculate tick range for a position based on percentage range"""
        tick_spacing = self.TICK_SPACINGS[fee_tier]
        
        # Calculate price range
        current_price = self.tick_to_price(current_tick)
        lower_price = current_price * (1 - range_percent / 100)
        upper_price = current_price * (1 + range_percent / 100)
        
        # Convert to ticks and round to tick spacing
        lower_tick = self.price_to_tick(lower_price)
        upper_tick = self.price_to_tick(upper_price)
        
        # Round to tick spacing
        lower_tick = (lower_tick // tick_spacing) * tick_spacing
        upper_tick = (upper_tick // tick_spacing) * tick_spacing
        
        return lower_tick, upper_tick
    
    def calculate_liquidity_amounts(
        self,
        amount0: float,
        amount1: float,
        sqrt_price_x96: int,
        lower_tick: int,
        upper_tick: int,
        decimals0: int = 18,
        decimals1: int = 18
    ) -> Dict:
        """Calculate liquidity amounts for a position"""
        # Convert ticks to sqrt prices
        sqrt_price_lower = math.sqrt(self.tick_to_price(lower_tick)) * (2 ** 96)
        sqrt_price_upper = math.sqrt(self.tick_to_price(upper_tick)) * (2 ** 96)
        
        # Current sqrt price
        sqrt_price_current = sqrt_price_x96
        
        # Calculate liquidity based on current price position
        if sqrt_price_current <= sqrt_price_lower:
            # Current price is below range, all in token0
            liquidity = amount0 * (sqrt_price_upper * sqrt_price_lower) / (sqrt_price_upper - sqrt_price_lower)
        elif sqrt_price_current >= sqrt_price_upper:
            # Current price is above range, all in token1
            liquidity = amount1 / (sqrt_price_upper - sqrt_price_lower)
        else:
            # Current price is in range
            liquidity0 = amount0 * (sqrt_price_upper * sqrt_price_current) / (sqrt_price_upper - sqrt_price_current)
            liquidity1 = amount1 / (sqrt_price_current - sqrt_price_lower)
            liquidity = min(liquidity0, liquidity1)
        
        return {
            "liquidity": int(liquidity),
            "amount0": amount0,
            "amount1": amount1,
            "lower_tick": lower_tick,
            "upper_tick": upper_tick
        }
    
    def get_position_value(
        self,
        liquidity: int,
        sqrt_price_x96: int,
        lower_tick: int,
        upper_tick: int,
        decimals0: int = 18,
        decimals1: int = 18
    ) -> Dict:
        """Calculate current value of a liquidity position"""
        # Convert ticks to sqrt prices
        sqrt_price_lower = math.sqrt(self.tick_to_price(lower_tick)) * (2 ** 96)
        sqrt_price_upper = math.sqrt(self.tick_to_price(upper_tick)) * (2 ** 96)
        sqrt_price_current = sqrt_price_x96
        
        # Calculate amounts based on current price
        if sqrt_price_current <= sqrt_price_lower:
            # All in token0
            amount0 = liquidity * (sqrt_price_upper - sqrt_price_lower) / (sqrt_price_upper * sqrt_price_lower)
            amount1 = 0
        elif sqrt_price_current >= sqrt_price_upper:
            # All in token1
            amount0 = 0
            amount1 = liquidity * (sqrt_price_upper - sqrt_price_lower)
        else:
            # In range
            amount0 = liquidity * (sqrt_price_upper - sqrt_price_current) / (sqrt_price_upper * sqrt_price_current)
            amount1 = liquidity * (sqrt_price_current - sqrt_price_lower)
        
        # Convert to human-readable amounts
        amount0_human = amount0 / (10 ** decimals0)
        amount1_human = amount1 / (10 ** decimals1)
        
        return {
            "amount0": amount0_human,
            "amount1": amount1_human,
            "in_range": sqrt_price_lower <= sqrt_price_current <= sqrt_price_upper
        }
    
    def simulate_swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        fee_tier: int = 3000,
        decimals_in: int = 18,
        decimals_out: int = 18
    ) -> Dict:
        """Simulate a swap and calculate expected output"""
        # Get pool
        pool_address = self.get_pool_address(token_in, token_out, fee_tier)
        if pool_address == "0x0000000000000000000000000000000000000000":
            return {"error": "Pool does not exist"}
        
        # Get pool info
        pool_info = self.get_pool_info(pool_address)
        
        # Determine if we're swapping token0 to token1 or vice versa
        zero_for_one = token_in.lower() == pool_info["token0"].lower()
        
        # Calculate price impact and output amount (simplified)
        sqrt_price_x96 = pool_info["sqrtPriceX96"]
        liquidity = pool_info["liquidity"]
        
        # Convert amount to wei
        amount_in_wei = int(amount_in * (10 ** decimals_in))
        
        # Simplified calculation (actual calculation is more complex)
        if zero_for_one:
            # Selling token0 for token1
            price = self.sqrt_price_to_price(sqrt_price_x96, decimals_in, decimals_out)
            amount_out = amount_in * price * (1 - fee_tier / 1000000)  # Apply fee
        else:
            # Selling token1 for token0
            price = 1 / self.sqrt_price_to_price(sqrt_price_x96, decimals_out, decimals_in)
            amount_out = amount_in * price * (1 - fee_tier / 1000000)  # Apply fee
        
        # Calculate price impact (simplified)
        price_impact = (amount_in_wei / liquidity) * 100 if liquidity > 0 else 0
        
        return {
            "pool_address": pool_address,
            "amount_in": amount_in,
            "amount_out": amount_out,
            "price": price,
            "price_impact_percent": price_impact,
            "fee_tier": fee_tier,
            "zero_for_one": zero_for_one
        }
    
    def get_all_pools_for_pair(self, token0: str, token1: str) -> List[Dict]:
        """Get all pools (different fee tiers) for a token pair"""
        pools = []
        
        for fee_name, fee_value in self.FEE_TIERS.items():
            pool_address = self.get_pool_address(token0, token1, fee_value)
            
            if pool_address != "0x0000000000000000000000000000000000000000":
                try:
                    pool_info = self.get_pool_info(pool_address)
                    pool_info["fee_tier_name"] = fee_name
                    pools.append(pool_info)
                except:
                    pass
        
        return pools
    
    def find_best_pool(self, token0: str, token1: str, amount_in: float) -> Optional[Dict]:
        """Find the best pool (fee tier) for a swap based on expected output"""
        pools = self.get_all_pools_for_pair(token0, token1)
        
        if not pools:
            return None
        
        best_pool = None
        best_output = 0
        
        for pool in pools:
            swap_result = self.simulate_swap(
                token0, token1, amount_in, pool["fee"]
            )
            
            if "amount_out" in swap_result and swap_result["amount_out"] > best_output:
                best_output = swap_result["amount_out"]
                best_pool = pool
                best_pool["expected_output"] = best_output
        
        return best_pool
    
    def calculate_impermanent_loss(
        self,
        price_initial: float,
        price_current: float,
        lower_price: float,
        upper_price: float
    ) -> Dict:
        """Calculate impermanent loss for a concentrated liquidity position"""
        # For positions that are always in range
        if price_initial >= lower_price and price_initial <= upper_price and \
           price_current >= lower_price and price_current <= upper_price:
            # Standard IL calculation
            price_ratio = price_current / price_initial
            il_percent = 2 * math.sqrt(price_ratio) / (1 + price_ratio) - 1
            il_percent = abs(il_percent) * 100
        else:
            # Position went out of range - more complex calculation
            # Simplified version
            if price_current < lower_price or price_current > upper_price:
                # Position is out of range, IL can be significant
                il_percent = 50  # Placeholder - actual calculation is complex
            else:
                il_percent = 0
        
        return {
            "impermanent_loss_percent": il_percent,
            "price_initial": price_initial,
            "price_current": price_current,
            "in_range": lower_price <= price_current <= upper_price
        }
    
    def demo(self):
        """Demo the Uniswap V3 implementation"""
        print("\n=== Uniswap V3 Python Demo ==={}".format(""))
        
        # Example tokens
        WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
        
        print("\n1. Finding all WETH/USDC pools:")
        pools = self.get_all_pools_for_pair(WETH, USDC)
        for pool in pools:
            print(f"   Fee tier: {pool.get('fee_tier_name', pool['fee']/10000)}%")
            print(f"   Price: 1 WETH = {pool['price']:.2f} USDC")
            print(f"   Liquidity: {pool['liquidity']}")
            print(f"   Current tick: {pool['tick']}")
            print()
        
        print("\n2. Simulating swap of 1 WETH to USDC:")
        for fee_tier in [500, 3000, 10000]:
            result = self.simulate_swap(WETH, USDC, 1.0, fee_tier, 18, 6)
            if "error" not in result:
                print(f"   Fee {fee_tier/10000}%: {result['amount_out']:.2f} USDC")
                print(f"   Price impact: {result['price_impact_percent']:.4f}%")
        
        print("\n3. Finding best pool for 10 WETH swap:")
        best = self.find_best_pool(WETH, USDC, 10.0)
        if best:
            print(f"   Best fee tier: {best['fee']/10000}%")
            print(f"   Expected output: {best.get('expected_output', 0):.2f} USDC")
        
        print("\n4. Concentrated liquidity position example:")
        # Get current pool state
        pool_address = self.get_pool_address(WETH, USDC, 3000)
        pool_info = self.get_pool_info(pool_address)
        current_tick = pool_info['tick']
        
        # Calculate position range (±5% from current price)
        lower_tick, upper_tick = self.get_tick_range(current_tick, 3000, 5)
        print(f"   Current tick: {current_tick}")
        print(f"   Position range: [{lower_tick}, {upper_tick}]")
        print(f"   Price range: [{self.tick_to_price(lower_tick):.4f}, {self.tick_to_price(upper_tick):.4f}]")
        
        # Calculate liquidity for 1 WETH + equivalent USDC
        liquidity_info = self.calculate_liquidity_amounts(
            1.0,  # 1 WETH
            pool_info['price'],  # Equivalent USDC
            pool_info['sqrtPriceX96'],
            lower_tick,
            upper_tick,
            18, 6
        )
        print(f"   Liquidity: {liquidity_info['liquidity']}")
        
        print("\n5. Impermanent loss calculation:")
        il_result = self.calculate_impermanent_loss(
            price_initial=2000,
            price_current=2500,
            lower_price=1800,
            upper_price=2200
        )
        print(f"   IL: {il_result['impermanent_loss_percent']:.2f}%")
        print(f"   Position in range: {il_result['in_range']}")


if __name__ == "__main__":
    # Note: This requires a valid Ethereum node connection
    # For demo purposes, we'll create the instance but won't run actual blockchain calls
    try:
        v3 = UniswapV3Python()
        print("Uniswap V3 implementation created successfully!")
        print("\nKey V3 features implemented:")
        print("- Concentrated liquidity positions")
        print("- Multiple fee tiers (0.05%, 0.3%, 1%)")
        print("- Tick-based pricing")
        print("- Position range calculations")
        print("- Impermanent loss for concentrated positions")
        print("\nTo run the demo with real data, provide a valid Ethereum node URL.")
    except Exception as e:
        print(f"Note: {e}")
        print("This is a demonstration of the V3 implementation structure.")
