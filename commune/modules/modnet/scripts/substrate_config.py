#!/usr/bin/env python3
"""
Substrate Configuration Management

Centralized configuration for Substrate node RPC connections,
supporting environment variables, auto-detection, and manual overrides.
"""

import os
import re
import subprocess
from dataclasses import dataclass


@dataclass
class SubstrateConfig:
    """Configuration for Substrate node connection."""

    http_url: str
    ws_url: str
    http_port: int
    ws_port: int
    node_binary: str = "solochain-template-node"
    auto_detect: bool = True


class SubstrateConfigManager:
    """Manages Substrate node configuration and auto-detection."""

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> SubstrateConfig:
        """Load configuration from environment variables and auto-detection."""
        # Environment variable overrides
        http_port = self._get_env_port("SUBSTRATE_HTTP_PORT", 9933)
        ws_port = self._get_env_port("SUBSTRATE_WS_PORT", 9944)

        # Auto-detect if enabled and no explicit ports set
        auto_detect = os.environ.get("SUBSTRATE_AUTO_DETECT", "true").lower() == "true"
        if auto_detect and not (
            os.environ.get("SUBSTRATE_HTTP_PORT")
            and os.environ.get("SUBSTRATE_WS_PORT")
        ):
            detected_ports = self._auto_detect_ports()
            if detected_ports:
                http_port = detected_ports.get("http", http_port)
                ws_port = detected_ports.get("ws", ws_port)

        # Build URLs
        http_url = f"http://127.0.0.1:{http_port}"
        ws_url = f"ws://127.0.0.1:{ws_port}"

        return SubstrateConfig(
            http_url=http_url,
            ws_url=ws_url,
            http_port=http_port,
            ws_port=ws_port,
            auto_detect=auto_detect,
        )

    def _get_env_port(self, env_var: str, default: int) -> int:
        """Get port from environment variable with fallback."""
        try:
            return int(os.environ.get(env_var, default))
        except ValueError:
            return default

    def _auto_detect_ports(self) -> dict[str, int] | None:
        """Auto-detect RPC ports from running Substrate node."""
        try:
            # Method 1: Check process command line arguments
            ports = self._detect_from_process_args()
            if ports:
                return ports

            # Method 2: Parse recent log output for RPC server info
            ports = self._detect_from_process_output()
            if ports:
                return ports

            return None

        except Exception as e:
            print(f"⚠️ Auto-detection failed: {e}")
            return None

    def _detect_from_process_args(self) -> dict[str, int] | None:
        """Detect ports from process command line arguments."""
        try:
            # Find substrate node processes
            result = subprocess.run(
                ["pgrep", "-f", "solochain-template-node"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            for pid in result.stdout.strip().split("\n"):
                if not pid:
                    continue

                # Get command line arguments
                cmd_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "args", "--no-headers"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if cmd_result.returncode == 0:
                    cmd_line = cmd_result.stdout.strip()

                    # Look for --rpc-port argument
                    rpc_port_match = re.search(r"--rpc-port[=\s]+(\d+)", cmd_line)
                    if rpc_port_match:
                        http_port = int(rpc_port_match.group(1))
                        return {"http": http_port, "ws": 9944}

            return None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            return None

    def _detect_from_process_output(self) -> dict[str, int] | None:
        """Detect ports by examining process output or logs."""
        try:
            # Method 1: Try to find substrate node process and get its file descriptors
            result = subprocess.run(
                ["pgrep", "-f", "solochain-template-node"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            # Get the first PID
            pids = [
                pid.strip() for pid in result.stdout.strip().split("\n") if pid.strip()
            ]
            if not pids:
                return None

            pid = pids[0]

            # Method 2: Check network connections for this process
            netstat_result = subprocess.run(
                ["netstat", "-tlnp"], capture_output=True, text=True, timeout=5
            )

            if netstat_result.returncode == 0:
                lines = netstat_result.stdout.split("\n")
                for line in lines:
                    if pid in line and ("127.0.0.1:" in line or "0.0.0.0:" in line):
                        # Parse line like: tcp 0 0 127.0.0.1:46749 0.0.0.0:* LISTEN 12345/solochain
                        parts = line.split()
                        if len(parts) >= 4:
                            addr = parts[3]
                            if ":" in addr:
                                port_str = addr.split(":")[-1]
                                try:
                                    port = int(port_str)
                                    # Assume this is the HTTP RPC port
                                    return {"http": port, "ws": 9944}
                                except ValueError:
                                    continue

            # Method 3: Try lsof to find open ports for the process
            lsof_result = subprocess.run(
                ["lsof", "-p", pid, "-i", "TCP", "-P", "-n"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if lsof_result.returncode == 0:
                lines = lsof_result.stdout.split("\n")
                for line in lines:
                    if "LISTEN" in line and ("127.0.0.1:" in line or "*:" in line):
                        # Parse line like: solochain 12345 user 10u IPv4 123456 0t0 TCP 127.0.0.1:46749 (LISTEN)
                        match = re.search(r":(\d+)\s+\(LISTEN\)", line)
                        if match:
                            port = int(match.group(1))
                            # Skip common ports that aren't RPC (like 30333 for p2p)
                            if (
                                port != 30333 and port != 9615
                            ):  # Skip p2p and prometheus ports
                                return {"http": port, "ws": 9944}

            # Method 4: Check recent journalctl logs for this specific process
            journal_result = subprocess.run(
                [
                    "journalctl",
                    "_PID=" + pid,
                    "--since",
                    "5 minutes ago",
                    "--no-pager",
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if journal_result.returncode == 0:
                log_content = journal_result.stdout
                # Look for "Running JSON-RPC server: addr=" pattern
                rpc_match = re.search(
                    r"Running JSON-RPC server: addr=([^,\s]+)", log_content
                )
                if rpc_match:
                    addr = rpc_match.group(1)
                    # Extract port from address like "127.0.0.1:46749" or "0.0.0.0:46749"
                    port_match = re.search(r":(\d+)$", addr)
                    if port_match:
                        http_port = int(port_match.group(1))
                        return {"http": http_port, "ws": 9944}

            return None

        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            ValueError,
            FileNotFoundError,
        ):
            return None

    def get_connection_urls(self) -> list[str]:
        """Get list of connection URLs to try."""
        urls = []

        # Primary URLs from config
        urls.append(self.config.http_url)
        urls.append(self.config.ws_url)

        # Fallback URLs
        if self.config.http_port != 9933:
            urls.append("http://127.0.0.1:9933")
        if self.config.ws_port != 9944:
            urls.append("ws://127.0.0.1:9944")

        return urls

    def print_config(self):
        """Print current configuration."""
        print("🔧 Substrate Configuration:")
        print(f"   HTTP URL: {self.config.http_url}")
        print(f"   WebSocket URL: {self.config.ws_url}")
        print(f"   Auto-detect: {self.config.auto_detect}")
        print(f"   Connection URLs: {self.get_connection_urls()}")


# Global config instance
substrate_config = SubstrateConfigManager()


if __name__ == "__main__":
    # Test configuration
    substrate_config.print_config()

    print("\n🔍 Testing connection URLs:")
    for url in substrate_config.get_connection_urls():
        print(f"   {url}")
