"""Core client for interacting with the module registry."""

from substrateinterface import SubstrateInterface

from scripts.config import get_config


class ModNetClient:
    """Client for interacting with the Mod-Net module registry."""

    def __init__(
        self,
        substrate_url: str,
        ipfs_api_url: str | None = None,
        ipfs_gateway_url: str | None = None,
    ):
        """
        Initialize the ModNet client.

        Args:
            substrate_url: Substrate WebSocket URL
            ipfs_api_url: Optional IPFS API URL (default from config)
            ipfs_gateway_url: Optional IPFS gateway URL (default from config)
        """
        config = get_config()
        self.substrate = SubstrateInterface(url=substrate_url)
        self.ipfs_api_url = ipfs_api_url or config.ipfs.api_url
        self.ipfs_gateway_url = ipfs_gateway_url or config.ipfs.gateway_url

    def health_check(self) -> bool:
        """Check if the client can connect to the substrate node.

        Returns:
            bool: True if connection is healthy
        """
        try:
            self.substrate.get_chain_head()
            return True
        except Exception:
            return False
