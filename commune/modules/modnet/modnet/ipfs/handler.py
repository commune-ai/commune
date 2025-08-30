"""IPFS handler for module registry metadata storage."""

from typing import Any

import ipfshttpclient

from scripts.config import get_config


class IPFSHandler:
    """Handler for IPFS operations."""

    config = get_config()

    def __init__(
        self,
        api_url: str = config.ipfs.api_url,
        gateway_url: str = config.ipfs.gateway_url,
    ) -> None:
        """Initialize the IPFS handler.

        Args:
            api_url: IPFS API URL
            gateway_url: IPFS gateway URL
        """
        self.api_url = api_url
        self.gateway_url = gateway_url
        self.client: ipfshttpclient.Client | None = None

    def connect(self) -> None:
        """Connect to IPFS node."""
        if not self.client:
            self.client = ipfshttpclient.connect(self.api_url)

    def disconnect(self) -> None:
        """Disconnect from IPFS node."""
        if self.client:
            self.client.close()
            self.client = None

    def add_json(self, data: dict[str, Any]) -> str:
        """Add JSON data to IPFS.

        Args:
            data: JSON data to store

        Returns:
            str: IPFS CID of stored data

        Raises:
            ConnectionError: If not connected to IPFS
        """
        if not self.client:
            raise ConnectionError("Not connected to IPFS")

        res = self.client.add_json(data)
        return str(res)
