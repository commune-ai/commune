"""Tests for the IPFS handler."""

from unittest.mock import MagicMock, patch

import pytest

from modnet.ipfs.handler import IPFSHandler


def test_ipfs_handler_initialization() -> None:
    """Test IPFS handler initialization."""
    handler = IPFSHandler()
    assert handler.api_url == "http://localhost:5001"
    assert handler.gateway_url == "http://localhost:8080"
    assert handler.client is None


def test_ipfs_handler_custom_urls() -> None:
    """Test IPFS handler with custom URLs."""
    handler = IPFSHandler(
        api_url="http://custom:5001",
        gateway_url="http://custom:8080",
    )
    assert handler.api_url == "http://custom:5001"
    assert handler.gateway_url == "http://custom:8080"


@patch("ipfshttpclient.connect")
def test_ipfs_connect(mock_connect: MagicMock) -> None:
    """Test IPFS connection."""
    mock_client = MagicMock()
    mock_connect.return_value = mock_client

    handler = IPFSHandler()
    handler.connect()

    mock_connect.assert_called_once_with("http://localhost:5001")
    assert handler.client == mock_client


def test_ipfs_disconnect() -> None:
    """Test IPFS disconnection."""
    handler = IPFSHandler()
    mock_client = MagicMock()
    handler.client = mock_client

    handler.disconnect()

    mock_client.close.assert_called_once()
    assert handler.client is None


@patch("ipfshttpclient.connect")
def test_add_json(mock_connect: MagicMock) -> None:
    """Test adding JSON to IPFS."""
    mock_client = MagicMock()
    mock_client.add_json.return_value = "QmTest123"
    mock_connect.return_value = mock_client

    handler = IPFSHandler()
    handler.connect()

    data = {"test": "data"}
    cid = handler.add_json(data)

    mock_client.add_json.assert_called_once_with(data)
    assert cid == "QmTest123"


def test_add_json_not_connected() -> None:
    """Test adding JSON when not connected."""
    handler = IPFSHandler()

    with pytest.raises(ConnectionError, match="Not connected to IPFS"):
        handler.add_json({"test": "data"})
