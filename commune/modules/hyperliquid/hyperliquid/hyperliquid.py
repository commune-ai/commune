import json
import time
from typing import Any, Dict, List, Optional, Union

from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants


class HyperliquidClient:
    """Thin, *all‑in* Python wrapper around the **hyperliquid‑python‑sdk**.

    ▸ **assets()**          – list every perp + spot market in one call (JSON blobs)
    ▸ **orderbook()**       – L2 book snapshot
    ▸ **trades()**          – recent public trades (fills)
    ▸ **balances() / positions()**
    ▸ **place_order() / cancel_order() / market_order()** – full trading via SDK signing

    The class stays *read‑only* if you skip *private_key*; sign‑in once and all trading
    helpers light up.
    """

    # -----------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------
    def __init__(
        self,
        base_url: str | None = None,
        private_key: str | None = None,
        address: str | None = None,
        timeout: int = 10,
        skip_ws: bool = True,
    ) -> None:
        self.base_url = base_url or constants.MAINNET_API_URL
        self.timeout = timeout

        # ---- public‑data handle (never needs a key) ---- #
        self.info = Info(self.base_url, skip_ws=skip_ws)

        # ---- trading handle (needs a key) ---- #
        self.wallet: Optional[LocalAccount] = None
        self.exchange: Optional[Exchange] = None
        if private_key is not None:
            self.wallet = Account.from_key(private_key)
            self.exchange = Exchange(wallet=self.wallet, base_url=self.base_url)
            # allow overriding address (ex: signing with API‑wallet for vault)
            if address:
                self.exchange.account_address = address

    # -----------------------------------------------------------
    # Metadata / discovery
    # -----------------------------------------------------------
    def assets(self) -> List[Dict[str, Any]]:
        """Return every tradable *perp* and *spot* market as JSON blobs."""
        perp_meta = self.info.meta()  # perps
        spot_meta = self.info.spot_meta()  # spot pairs

        blobs: List[Dict[str, Any]] = []

        # Perp universe
        for idx, a in enumerate(perp_meta["universe"]):
            blobs.append({"type": "perp", "asset_id": idx, **a})

        # Spot universe (spot assets start at 10k)
        for s in spot_meta["universe"]:
            asset_id = s["index"] + 10_000
            base_idx, quote_idx = s["tokens"]
            pair = f"{spot_meta['tokens'][base_idx]['name']}/{spot_meta['tokens'][quote_idx]['name']}"
            blobs.append({"type": "spot", "asset_id": asset_id, "pair": pair, **s})

        return blobs

    # -----------------------------------------------------------
    # Market data
    # -----------------------------------------------------------
    def orderbook(self, symbol: str, depth: int = 50) -> Dict[str, Any]:
        """Return a depth *levels* snapshot (bids/asks) for *symbol*."""
        snapshot = self.info.l2_snapshot(symbol.upper())
        bids, asks = snapshot["levels"]  # [[bid‑levels], [ask‑levels]]
        return {"bids": bids[:depth], "asks": asks[:depth], "time": snapshot["time"]}

    def trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Public recent trades (fills). Limit 500 by Hyperliquid API."""
        asset = self.info.name_to_coin[symbol.upper()]
        now = int(time.time() * 1000)
        # fetch last ~24h and slice – HL API has no direct "trades" endpoint, we use fillsByTime
        day_ms = 86_400_000
        raw = self.info.post("/info", {"type": "tradesByTime", "asset": asset, "startTime": now - day_ms})
        return raw[-limit:]

    # -----------------------------------------------------------
    # Account state helpers
    # -----------------------------------------------------------
    def balances(self, address: str | None = None) -> Dict[str, Any]:
        who = address or self._default_address()
        return self.info.user_state(who)

    def positions(self, address: str | None = None) -> List[Dict[str, Any]]:
        who = address or self._default_address()
        return self.info.user_state(who)["assetPositions"]

    # -----------------------------------------------------------
    # Trading helpers (require private key)
    # -----------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        is_buy: bool,
        size: float,
        price: float,
        tif: str = "Gtc",
        reduce_only: bool = False,
        client_id: Optional[str] = None,
        builder: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit a *limit* order via the signed SDK."""
        self._require_trading()
        order_type = {"limit": {"tif": tif}}
        res = self.exchange.order(
            symbol.upper(),
            is_buy,
            size,
            price,
            order_type=order_type,
            reduce_only=reduce_only,
            cloid=client_id,
            builder=builder,
        )
        return res

    def market_order(
        self,
        symbol: str,
        is_buy: bool,
        size: float,
        slippage: float = Exchange.DEFAULT_SLIPPAGE,
        client_id: Optional[str] = None,
        builder: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fire a synthetic *market* order (aggressive IoC limit) via SDK."""
        self._require_trading()
        return self.exchange.market_open(
            symbol.upper(),
            is_buy,
            size,
            slippage=slippage,
            cloid=client_id,
            builder=builder,
        )

    def cancel_order(self, symbol: str, oid: int) -> Dict[str, Any]:
        """Cancel an order by numeric *oid*."""
        self._require_trading()
        return self.exchange.cancel(symbol.upper(), oid)

    # -----------------------------------------------------------
    # Utility
    # -----------------------------------------------------------
    def _default_address(self) -> str:
        if self.exchange and self.exchange.account_address:
            return self.exchange.account_address
        if self.wallet:
            return self.wallet.address
        raise ValueError("No address configured. Pass one to balances()/positions().")

    def _require_trading(self):
        if not self.exchange:
            raise RuntimeError("Trading functions need a private_key during init.")

    # Quick smoke test
    def self_test(self) -> Dict[str, Any]:
        meta_ok = len(self.assets())
        ping = list(self.info.all_mids().items())[:3]  # grab a couple mids
        return {"meta_assets": meta_ok, "sample_mids": ping, "ok": True}


if __name__ == "__main__":
    # Local demo (read‑only). Supply PRIVATE_KEY to enable trading.
    client = HyperliquidClient()
    print(json.dumps(client.self_test(), indent=2))
