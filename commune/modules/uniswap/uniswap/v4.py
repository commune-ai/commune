# pip install web3==6.* eth-abi==5.* requests
from __future__ import annotations

from decimal import Decimal, getcontext
from typing import Dict, Optional, Tuple, Any
from web3 import Web3
from web3.contract.contract import Contract

# High precision for price math
getcontext().prec = 80

class UniswapV4PriceScraper:
    """
    Uniswap v4 on-chain price reader using the official StateView contract.
    - Computes PoolId = keccak256(abi.encode(PoolKey)) and calls StateView.getSlot0(poolId)
    - Derives mid-price from sqrtPriceX96 and token decimals.
    - Can compare prices between any two pools.

    References:
      • StateView getSlot0 (sqrtPriceX96, tick, protocolFee, lpFee) – Uniswap v4 docs
      • PoolId = keccak256(abi.encode(PoolKey)) – Uniswap v4 docs
      • sqrtPriceX96 is sqrt(price token1/token0) in Q64.96 – Uniswap v4 docs
    """

    # Minimal ABIs
    _STATEVIEW_ABI = [
        {
            "type": "function",
            "name": "getSlot0",
            "stateMutability": "view",
            "inputs": [{"name": "poolId", "type": "bytes32"}],
            "outputs": [
                {"name": "sqrtPriceX96", "type": "uint160"},
                {"name": "tick", "type": "int24"},
                {"name": "protocolFee", "type": "uint24"},
                {"name": "lpFee", "type": "uint24"},
            ],
        }
    ]

    _ERC20_ABI = [
        {"type":"function","name":"decimals","stateMutability":"view","inputs":[],"outputs":[{"type":"uint8"}]},
        {"type":"function","name":"symbol","stateMutability":"view","inputs":[],"outputs":[{"type":"string"}]},
        {"type":"function","name":"name","stateMutability":"view","inputs":[],"outputs":[{"type":"string"}]},
    ]

    # Canonical deployed StateView addresses from Uniswap docs (Aug 2025)
    # Add more networks as needed.
    _STATEVIEW_BY_CHAIN = {
        "ethereum": Web3.to_checksum_address("0x7FfE42C4A5dEEa5B0fEC41C94C136cF115597227"),
        "base":     Web3.to_checksum_address("0xA3C0C9b65bAd0B08107aA264B0f3Db444B867A71"),
        "arbitrum": Web3.to_checksum_address("0x76fD297E2D437Cd7F76D50F01afE6160F86E9990"),
        "optimism": Web3.to_checksum_address("0xC18a3169788F4F75A170290584EcA6395C75eCDB"),
        "polygon":  Web3.to_checksum_address("0x5Ea1BD7974c8a611cBAB0bDcAfcB1D9cC9B3bA5a"),
    }

    ZERO = "0x0000000000000000000000000000000000000000"

    def __init__(self,
                 rpc_url: str,
                 chain: str = "ethereum",
                 stateview_address: Optional[str] = None):
        """
        rpc_url: HTTPS RPC endpoint for the target chain.
        chain:   One of keys in _STATEVIEW_BY_CHAIN (default 'ethereum').
        stateview_address: Optional override of StateView address.
        """
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
        if not self.w3.is_connected():
            raise RuntimeError("Web3 provider not connected. Check your RPC URL.")

        if stateview_address:
            self.stateview_addr = Web3.to_checksum_address(stateview_address)
        else:
            if chain not in self._STATEVIEW_BY_CHAIN:
                raise ValueError(f"Unknown chain '{chain}'. Provide stateview_address.")
            self.stateview_addr = self._STATEVIEW_BY_CHAIN[chain]

        self.stateview: Contract = self.w3.eth.contract(
            address=self.stateview_addr, abi=self._STATEVIEW_ABI
        )

    # ------------------------ public API ------------------------

    def mid_price(self,
                  tokenA: str,
                  tokenB: str,
                  fee: int,
                  tick_spacing: int,
                  hooks: str = ZERO) -> Dict[str, Any]:
        """
        Get the current mid-price for a specific v4 pool.

        tokenA, tokenB: ERC-20 addresses (any order).
        fee:             uint24, in hundredths of a bip (e.g. 3000 = 0.30%).
        tick_spacing:    int24. (In v4 this is independent and must match the pool.)
        hooks:           Hook contract address, or ZERO if none.

        Returns dict with:
          - poolId
          - currency0, currency1 (canonical order)
          - sqrtPriceX96, tick, lpFee, protocolFee
          - price_1_per_0 (normalized by decimals)
          - price_0_per_1 (inverse)
          - decimals0, decimals1, symbols
        """
        c0, c1 = self._canonical_pair(tokenA, tokenB)
        pool_id = self._compute_pool_id(c0, c1, fee, tick_spacing, hooks)

        # read slot0 from StateView
        sqrtP, tick, protocol_fee, lp_fee = self.stateview.functions.getSlot0(pool_id).call()

        # token metadata
        d0, s0 = self._decimals_and_symbol(c0)
        d1, s1 = self._decimals_and_symbol(c1)

        price_1_per_0 = self._price_from_sqrtP(sqrtP, d0, d1)
        price_0_per_1 = (Decimal(1) / price_1_per_0) if price_1_per_0 != 0 else Decimal(0)

        return {
            "poolId": pool_id.hex(),
            "currency0": c0,
            "currency1": c1,
            "sqrtPriceX96": int(sqrtP),
            "tick": int(tick),
            "lpFee": int(lp_fee),
            "protocolFee": int(protocol_fee),
            "decimals0": d0,
            "decimals1": d1,
            "symbol0": s0,
            "symbol1": s1,
            "price_1_per_0": str(price_1_per_0),  # token1 per 1 token0
            "price_0_per_1": str(price_0_per_1),  # token0 per 1 token1
        }

    def compare_pools(self,
                      poolA: Dict[str, Any],
                      poolB: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare mid-prices of any two pools (each dict must have tokenA, tokenB, fee, tick_spacing, hooks).
        Returns both prices and their ratio (A / B in token1-per-token0 terms for each pool's own canonical pair).
        """
        a = self.mid_price(**poolA)
        b = self.mid_price(**poolB)

        # Note: Each price is token1 per token0 under its own canonical order.
        # For a clean ratio, compare as decimals and label clearly.
        pa = Decimal(a["price_1_per_0"])
        pb = Decimal(b["price_1_per_0"])
        ratio = (pa / pb) if pb != 0 else Decimal(0)

        return {
            "poolA": a,
            "poolB": b,
            "ratio_A_over_B_token1per0": str(ratio)
        }

    # ------------------------ internals ------------------------

    def _erc20(self, addr: str) -> Contract:
        return self.w3.eth.contract(address=Web3.to_checksum_address(addr), abi=self._ERC20_ABI)

    def _decimals_and_symbol(self, addr: str) -> Tuple[int, str]:
        c = self._erc20(addr)
        # Some tokens are notorious; be defensive
        try:
            decimals = int(c.functions.decimals().call())
        except Exception:
            decimals = 18
        try:
            symbol = c.functions.symbol().call()
        except Exception:
            symbol = addr[:6] + "…" + addr[-4:]
        return decimals, symbol

    @staticmethod
    def _canonical_pair(a: str, b: str) -> Tuple[str, str]:
        a = Web3.to_checksum_address(a)
        b = Web3.to_checksum_address(b)
        return (a, b) if a.lower() < b.lower() else (b, a)

    def _compute_pool_id(self,
                         currency0: str,
                         currency1: str,
                         fee: int,
                         tick_spacing: int,
                         hooks: str) -> bytes:
        """
        PoolId = keccak256(abi.encode(PoolKey{currency0, currency1, fee(uint24), tickSpacing(int24), hooks}))
        """
        hooks = Web3.to_checksum_address(hooks)
        return Web3.solidity_keccak(
            ["address", "address", "uint24", "int24", "address"],
            [currency0, currency1, fee, tick_spacing, hooks]
        )

    @staticmethod
    def _price_from_sqrtP(sqrt_price_x96: int, decimals0: int, decimals1: int) -> Decimal:
        """
        Convert sqrtPriceX96 to mid price (token1 per 1 token0), normalized by decimals.
        price = (sqrtP / 2^96)^2 * 10^(decimals1 - decimals0)
        """
        if sqrt_price_x96 == 0:
            return Decimal(0)
        Q96 = Decimal(2) ** 96
        sp = Decimal(sqrt_price_x96) / Q96
        raw = sp * sp
        scale = Decimal(10) ** (decimals1 - decimals0)
        return raw * scale
