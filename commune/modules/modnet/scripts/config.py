#!/usr/bin/env python3
"""
Centralized Configuration Management

This module provides centralized configuration for all services and components
in the mod-net modules project, replacing hard-coded values with environment
variables and sensible defaults.
"""

import os
from dataclasses import dataclass, field


def _get_default_ws_url() -> str:
    """Get default WebSocket URL with security considerations."""
    # Use secure WebSocket (wss://) for production environments
    if os.getenv("ENVIRONMENT") == "production":
        return "wss://substrate-node:9944"
    # Use insecure WebSocket (ws://) only for local development
    return "ws://127.0.0.1:9944"


@dataclass
class SubstrateConfig:
    """Configuration for Substrate blockchain connection."""

    # Connection URLs
    http_url: str = field(
        default_factory=lambda: os.getenv("SUBSTRATE_HTTP_URL", "http://127.0.0.1:9933")
    )
    ws_url: str = field(
        default_factory=lambda: os.getenv("SUBSTRATE_WS_URL", _get_default_ws_url())
    )

    # Ports
    http_port: int = field(
        default_factory=lambda: int(os.getenv("SUBSTRATE_HTTP_PORT", "9933"))
    )
    ws_port: int = field(
        default_factory=lambda: int(os.getenv("SUBSTRATE_WS_PORT", "9944"))
    )

    # Authentication
    keypair_seed: str = field(
        default_factory=lambda: os.getenv("SUBSTRATE_KEYPAIR_SEED", "//Alice")
    )

    # Testing
    test_public_key: str = field(
        default_factory=lambda: os.getenv(
            "SUBSTRATE_TEST_PUBLIC_KEY", "0x1234567890abcdef1234567890abcdef12345678"
        )
    )

    # Node configuration
    node_binary: str = field(
        default_factory=lambda: os.getenv(
            "SUBSTRATE_NODE_BINARY", "solochain-template-node"
        )
    )
    auto_detect: bool = field(
        default_factory=lambda: os.getenv("SUBSTRATE_AUTO_DETECT", "true").lower()
        == "true"
    )


@dataclass
class IPFSConfig:
    """Configuration for IPFS services."""

    # API and Gateway URLs
    api_url: str = field(
        default_factory=lambda: os.getenv("IPFS_API_URL", "http://localhost:5001")
    )
    gateway_url: str = field(
        default_factory=lambda: os.getenv("IPFS_GATEWAY_URL", "http://localhost:8080")
    )

    # Ports
    api_port: int = field(
        default_factory=lambda: int(os.getenv("IPFS_API_PORT", "5001"))
    )
    gateway_port: int = field(
        default_factory=lambda: int(os.getenv("IPFS_GATEWAY_PORT", "8080"))
    )

    # Host
    host: str = field(default_factory=lambda: os.getenv("IPFS_HOST", "localhost"))


@dataclass
class CommuneIPFSConfig:
    """Configuration for commune-ipfs backend service."""

    # Server configuration
    host: str = field(
        default_factory=lambda: os.getenv("COMMUNE_IPFS_HOST", "localhost")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("COMMUNE_IPFS_PORT", "8000"))
    )

    # URLs
    base_url: str = field(
        default_factory=lambda: f"http://{os.getenv('COMMUNE_IPFS_HOST', 'localhost')}:{os.getenv('COMMUNE_IPFS_PORT', '8000')}"
    )

    # IPFS backend configuration
    ipfs_api_url: str = field(
        default_factory=lambda: os.getenv(
            "COMMUNE_IPFS_API_URL", "http://localhost:5001"
        )
    )
    ipfs_gateway_url: str = field(
        default_factory=lambda: os.getenv(
            "COMMUNE_IPFS_GATEWAY_URL", "http://localhost:8080"
        )
    )


@dataclass
class TestConfig:
    """Configuration for test environments."""

    # Test module configuration
    test_module_registry_url: str = field(
        default_factory=lambda: os.getenv(
            "TEST_MODULE_REGISTRY_URL", "http://localhost:8004"
        )
    )

    # RPC test URLs
    rpc_test_urls: list[str] = field(
        default_factory=lambda: [
            os.getenv("TEST_RPC_URL_1", "http://127.0.0.1:9933"),
            os.getenv("TEST_RPC_URL_2", "http://localhost:9933"),
            os.getenv("TEST_RPC_URL_3", "http://0.0.0.0:9933"),
        ]
    )

    # Integration test configuration
    substrate_url: str = field(
        default_factory=lambda: os.getenv("TEST_SUBSTRATE_URL", "ws://127.0.0.1:9944")
    )
    ipfs_backend_url: str = field(
        default_factory=lambda: os.getenv(
            "TEST_IPFS_BACKEND_URL", "http://localhost:8000"
        )
    )


@dataclass
class SecurityConfig:
    """Configuration for security-related settings."""

    # Default keypair seeds (should be overridden in production)
    default_keypair_seed: str = field(
        default_factory=lambda: os.getenv("DEFAULT_KEYPAIR_SEED", "//Alice")
    )

    # Key generation
    use_secure_random: bool = field(
        default_factory=lambda: os.getenv("USE_SECURE_RANDOM", "true").lower() == "true"
    )

    # Environment detection
    is_development: bool = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development").lower()
        == "development"
    )
    is_production: bool = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development").lower()
        == "production"
    )


@dataclass
class ModNetConfig:
    """Main configuration container for the entire mod-net project."""

    substrate: SubstrateConfig = field(default_factory=SubstrateConfig)
    ipfs: IPFSConfig = field(default_factory=IPFSConfig)
    commune_ipfs: CommuneIPFSConfig = field(default_factory=CommuneIPFSConfig)
    test: TestConfig = field(default_factory=TestConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __post_init__(self):
        """Post-initialization to handle dynamic URL construction."""
        # Update commune-ipfs base_url with actual host and port
        self.commune_ipfs.base_url = (
            f"http://{self.commune_ipfs.host}:{self.commune_ipfs.port}"
        )

        # Update IPFS URLs with actual host and ports
        self.ipfs.api_url = f"http://{self.ipfs.host}:{self.ipfs.api_port}"
        self.ipfs.gateway_url = f"http://{self.ipfs.host}:{self.ipfs.gateway_port}"

    def print_config(self, show_sensitive: bool = False):
        """Print current configuration (optionally hiding sensitive values)."""
        print("🔧 Mod-Net Configuration:")
        print("\n📡 Substrate:")
        print(f"   HTTP URL: {self.substrate.http_url}")
        print(f"   WebSocket URL: {self.substrate.ws_url}")
        print(f"   Auto-detect: {self.substrate.auto_detect}")
        if show_sensitive:
            print(f"   Keypair Seed: {self.substrate.keypair_seed}")
        else:
            print(f"   Keypair Seed: {'*' * len(self.substrate.keypair_seed)}")

        print("\n📦 IPFS:")
        print(f"   API URL: {self.ipfs.api_url}")
        print(f"   Gateway URL: {self.ipfs.gateway_url}")

        print("\n🌐 Commune-IPFS:")
        print(f"   Base URL: {self.commune_ipfs.base_url}")
        print(f"   Host: {self.commune_ipfs.host}")
        print(f"   Port: {self.commune_ipfs.port}")

        print("\n🧪 Test Configuration:")
        print(f"   Module Registry URL: {self.test.test_module_registry_url}")
        print(f"   Substrate URL: {self.test.substrate_url}")
        print(f"   IPFS Backend URL: {self.test.ipfs_backend_url}")

        print("\n🔒 Security:")
        print(
            f"   Environment: {'Development' if self.security.is_development else 'Production'}"
        )
        print(f"   Secure Random: {self.security.use_secure_random}")

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check for development seeds in production
        if self.security.is_production and self.substrate.keypair_seed in [
            "//Alice",
            "//Bob",
            "//Charlie",
        ]:
            issues.append("Production environment using development keypair seed")

        # Check for localhost in production
        if self.security.is_production:
            if (
                "localhost" in self.substrate.http_url
                or "127.0.0.1" in self.substrate.http_url
            ):
                issues.append(
                    "Production environment using localhost for Substrate connection"
                )
            if "localhost" in self.ipfs.api_url or "127.0.0.1" in self.ipfs.api_url:
                issues.append(
                    "Production environment using localhost for IPFS connection"
                )

        return issues


# Global configuration instance
config = ModNetConfig()


def get_config() -> ModNetConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> ModNetConfig:
    """Reload configuration from environment variables."""
    global config
    config = ModNetConfig()
    return config


if __name__ == "__main__":
    # Test configuration
    config.print_config(show_sensitive=True)

    print("\n🔍 Configuration Validation:")
    issues = config.validate_config()
    if issues:
        for issue in issues:
            print(f"   ⚠️ {issue}")
    else:
        print("   ✅ Configuration is valid")
