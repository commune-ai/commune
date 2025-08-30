"""Tests for the ModNet client."""

from unittest.mock import MagicMock, patch

from modnet.core.client import ModNetClient


@patch("modnet.core.client.SubstrateInterface")
def test_client_initialization(mock_substrate: MagicMock) -> None:
    """Test client initialization with default values."""
    mock_substrate.return_value = MagicMock()
    client = ModNetClient("ws://localhost:9944")
    assert client.ipfs_api_url == "http://localhost:5001"
    assert client.ipfs_gateway_url == "http://localhost:8080"
    mock_substrate.assert_called_once_with(url="ws://localhost:9944")


@patch("modnet.core.client.SubstrateInterface")
def test_client_custom_ipfs_urls(mock_substrate: MagicMock) -> None:
    """Test client initialization with custom IPFS URLs."""
    mock_substrate.return_value = MagicMock()
    client = ModNetClient(
        "ws://localhost:9944",
        ipfs_api_url="http://custom:5001",
        ipfs_gateway_url="http://custom:8080",
    )
    assert client.ipfs_api_url == "http://custom:5001"
    assert client.ipfs_gateway_url == "http://custom:8080"
    mock_substrate.assert_called_once_with(url="ws://localhost:9944")


@patch("modnet.core.client.SubstrateInterface")
def test_health_check_success(mock_substrate: MagicMock) -> None:
    """Test successful health check."""
    mock_instance = MagicMock()
    mock_instance.get_chain_head.return_value = "0x1234"
    mock_substrate.return_value = mock_instance

    client = ModNetClient("ws://localhost:9944")
    assert client.health_check() is True
    mock_instance.get_chain_head.assert_called_once()


@patch("modnet.core.client.SubstrateInterface")
def test_health_check_failure(mock_substrate: MagicMock) -> None:
    """Test failed health check."""
    mock_instance = MagicMock()
    mock_instance.get_chain_head.side_effect = Exception("Connection failed")
    mock_substrate.return_value = mock_instance

    client = ModNetClient("ws://localhost:9944")
    assert client.health_check() is False
    mock_instance.get_chain_head.assert_called_once()
