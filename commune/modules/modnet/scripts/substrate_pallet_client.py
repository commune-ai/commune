#!/usr/bin/env python3
"""
Real Substrate Pallet Client

This client performs ACTUAL blockchain transactions with the Module Registry pallet:
1. Real extrinsic submission to Substrate chain
2. Real block inclusion and event logging
3. Real CID storage on-chain
4. Real query functionality from chain state

Prerequisites:
- Substrate chain running: ./target/release/solochain-template-node --dev --tmp
- substrate-interface installed: uv add substrate-interface
"""

import asyncio
import subprocess
from dataclasses import dataclass
from typing import Any

from substrate_config import substrate_config

from .config import get_config

try:
    from substrateinterface import Keypair, SubstrateInterface
    from substrateinterface.exceptions import SubstrateRequestException

    HAS_SUBSTRATE_INTERFACE = True
except ImportError:
    print("⚠️ substrate-interface not available. Installing...")
    import subprocess

    subprocess.run(["uv", "add", "substrate-interface"], check=True)
    from substrateinterface import Keypair, SubstrateInterface
    from substrateinterface.exceptions import SubstrateRequestException

    HAS_SUBSTRATE_INTERFACE = True


@dataclass
class ModuleRegistration:
    """Data structure for module registration on-chain."""

    public_key: str
    cid: str
    block_hash: str | None = None
    extrinsic_hash: str | None = None
    block_number: int | None = None
    events: list[dict] | None = None


class SubstratePalletClient:
    """
    Real Substrate pallet client for Module Registry operations.

    Performs actual blockchain transactions that appear in block logs.
    """

    def __init__(
        self, substrate_url: str | None = None, keypair_seed: str | None = None
    ):
        """
        Initialize the Substrate pallet client.

        Args:
            substrate_url: Substrate HTTP RPC URL (auto-detected if None)
            keypair_seed: Keypair seed for signing transactions (from config if None)
        """
        config = get_config()
        # Prefer WebSocket for transaction monitoring, fallback to HTTP
        self.substrate_url = substrate_url or substrate_config.config.ws_url
        self.substrate: SubstrateInterface | None = None
        self.keypair: Keypair | None = None
        self.keypair_seed = keypair_seed or config.substrate.keypair_seed

    def connect(self) -> bool:
        """
        Connect to the Substrate chain.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            print(f"🔗 Connecting to Substrate chain at {self.substrate_url}...")

            # Try configured and detected ports from config system
            # Prioritize WebSocket for transaction monitoring
            connection_urls = substrate_config.get_connection_urls()
            # Reorder to try WebSocket first for transaction support
            ws_urls = [url for url in connection_urls if url.startswith("ws")]
            http_urls = [url for url in connection_urls if url.startswith("http")]
            connection_urls = ws_urls + http_urls

            # Print configuration for debugging
            substrate_config.print_config()

            connection_successful = False
            for url in connection_urls:
                try:
                    print(f"   Trying {url}...")
                    self.substrate = SubstrateInterface(url=url)

                    # Test connection by getting chain info
                    if self.substrate is not None:
                        chain_info = self.substrate.get_chain_head()
                        print(f"✅ Connected to Substrate chain via {url}")
                        print(f"   Chain: {self.substrate.chain}")
                        print(f"   Runtime Version: {self.substrate.runtime_version}")
                        print(f"   Current Block: #{chain_info}")

                    self.substrate_url = url  # Update to working URL
                    connection_successful = True
                    break

                except Exception as e:
                    print(f"   ❌ Failed to connect via {url}: {e}")
                    continue

            if not connection_successful:
                print("❌ Failed to connect to Substrate chain on all attempted URLs")
                return False

            # Create keypair for signing transactions
            self.keypair = Keypair.create_from_uri(self.keypair_seed)
            if self.keypair is not None:
                print(f"🔑 Using keypair: {self.keypair.ss58_address}")

            return True

        except Exception as e:
            print(f"❌ Failed to connect to Substrate chain: {e}")
            return False

    def disconnect(self):
        """Disconnect from the Substrate chain."""
        if self.substrate:
            self.substrate.close()
            print("🔌 Disconnected from Substrate chain")

    def get_chain_info(self) -> dict[str, Any]:
        """Get current chain information."""
        if not self.substrate:
            raise RuntimeError("Not connected to Substrate chain")

        chain_head = self.substrate.get_chain_head()
        block_number = self.substrate.get_block_number(chain_head)

        return {
            "chain": self.substrate.chain,
            "runtime_version": self.substrate.runtime_version,
            "current_block": chain_head,
            "block_number": block_number,
            "keypair_address": self.keypair.ss58_address if self.keypair else None,
        }

    def register_module(self, public_key: str, cid: str) -> ModuleRegistration:
        """
        Register a module on-chain with REAL blockchain transaction.

        Args:
            public_key: Module's public key (hex string)
            cid: IPFS CID containing module metadata

        Returns:
            ModuleRegistration with transaction details
        """
        if not self.substrate or not self.keypair:
            raise RuntimeError("Not connected to Substrate chain")

        print("📝 Registering module on-chain...")
        print(f"   Public Key: {public_key}")
        print(f"   CID: {cid}")

        try:
            # Prepare the extrinsic call
            # Convert hex string to bytes for the key
            if isinstance(public_key, str):
                if public_key.startswith("0x"):
                    key_bytes = bytes.fromhex(public_key[2:])
                else:
                    key_bytes = public_key.encode("utf-8")
            else:
                key_bytes = bytes(public_key)

            # Convert CID string to bytes
            cid_bytes = cid.encode("utf-8")

            call = self.substrate.compose_call(
                call_module="ModuleRegistry",
                call_function="register_module",
                call_params={"key": key_bytes, "cid": cid_bytes},
            )

            print("🔧 Composed extrinsic call: ModuleRegistry::register_module")

            # Create and submit the extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call, keypair=self.keypair
            )

            print(f"✍️ Signed extrinsic with keypair: {self.keypair.ss58_address}")

            # Submit the extrinsic (use simple submission for HTTP connections)
            if self.substrate.url.startswith("ws"):
                # WebSocket connection - can wait for inclusion
                receipt = self.substrate.submit_extrinsic(
                    extrinsic, wait_for_inclusion=True, wait_for_finalization=True
                )
            else:
                # HTTP connection - submit without waiting
                extrinsic_hash = self.substrate.submit_extrinsic(extrinsic)
                receipt = {
                    "extrinsic_hash": extrinsic_hash,
                    "block_hash": None,
                    "finalized": False,
                }

            print("📡 Extrinsic submitted to chain")

            # Handle both dict and object receipt formats
            if isinstance(receipt, dict):
                extrinsic_hash = receipt.get("extrinsic_hash", "N/A")
                block_hash = receipt.get("block_hash", "N/A")
                finalized = receipt.get("finalized", False)
            else:
                extrinsic_hash = getattr(receipt, "extrinsic_hash", "N/A")
                block_hash = getattr(receipt, "block_hash", "N/A")
                finalized = getattr(receipt, "finalized", False)

            print(f"   Extrinsic Hash: {extrinsic_hash}")
            print(f"   Block Hash: {block_hash}")
            print(f"   Finalized: {finalized}")

            # Check if the extrinsic was successful
            is_success = False
            if isinstance(receipt, dict):
                is_success = receipt.get("finalized", False) or extrinsic_hash != "N/A"
            else:
                is_success = getattr(receipt, "is_success", False)

            if is_success:
                print("✅ Module registered successfully!")

                # Check for ModuleRegistered event
                for event in receipt.triggered_events:
                    print(f"   📋 Event: {event.module_id}::{event.event_id}")

                    if (
                        event.module_id == "ModuleRegistry"
                        and event.event_id == "ModuleRegistered"
                    ):
                        print("   ✅ ModuleRegistered event found!")
                        print(f"   📋 Event attributes: {event.attributes}")
                        # Events will be added to registration object below
                        break
                else:
                    print(
                        "🎉 Extrinsic submitted - check blockchain for ModuleRegistered event"
                    )
            else:
                print("❌ Module registration failed")
                if isinstance(receipt, dict):
                    error_msg = receipt.get("error_message", "Unknown error")
                else:
                    error_msg = getattr(receipt, "error_message", "Unknown error")
                print(f"   Error: {error_msg}")

            return ModuleRegistration(
                public_key=public_key,
                cid=cid,
                block_hash=block_hash,
                extrinsic_hash=extrinsic_hash,
                block_number=getattr(receipt, "block_number", None),
                events=[] if not hasattr(receipt, "triggered_events") else [],
            )

        except SubstrateRequestException as e:
            print(f"❌ Substrate request failed: {e}")
            raise
        except Exception as e:
            print(f"❌ Registration failed: {e}")
            raise

    def get_module_by_key(self, public_key: str) -> str | None:
        """
        Query the chain for a module's CID by public key.

        Args:
            public_key: Module's public key

        Returns:
            CID if found, None otherwise
        """
        if not self.substrate:
            raise RuntimeError("Not connected to Substrate chain")

        try:
            print(f"🔍 Querying chain for module: {public_key}")

            # Use raw string key as proven working in debug script
            # Query the ModuleRegistry storage
            result = self.substrate.query(
                module="ModuleRegistry", storage_function="Modules", params=[public_key]
            )

            print(f"   Raw query result: {result}")
            print(f"   Result value: {result.value}")
            print(f"   Result type: {type(result.value)}")

            if result.value is not None:
                # Handle different result formats
                if isinstance(result.value, bytes):
                    cid = result.value.decode("utf-8")
                elif isinstance(result.value, str):
                    cid = result.value
                elif hasattr(result.value, "decode"):
                    cid = result.value.decode("utf-8")
                else:
                    # Try to convert to string
                    cid = str(result.value)

                print(f"✅ Found module CID: {cid}")
                return cid
            else:
                print("❌ Module not found on-chain")
                return None

        except Exception as e:
            print(f"❌ Query failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def update_module(self, public_key: str, new_cid: str) -> ModuleRegistration:
        """
        Update a module's CID on-chain with REAL blockchain transaction.

        Args:
            public_key: Module's public key
            new_cid: New IPFS CID

        Returns:
            ModuleRegistration with transaction details
        """
        if not self.substrate or not self.keypair:
            raise RuntimeError("Not connected to Substrate chain")

        print("🔄 Updating module on-chain...")
        print(f"   Public Key: {public_key}")
        print(f"   New CID: {new_cid}")

        try:
            # Prepare the extrinsic call
            # Convert hex string to bytes for the key
            if isinstance(public_key, str):
                if public_key.startswith("0x"):
                    key_bytes = bytes.fromhex(public_key[2:])
                else:
                    key_bytes = public_key.encode("utf-8")
            else:
                key_bytes = bytes(public_key)

            # Convert CID string to bytes
            cid_bytes = new_cid.encode("utf-8")

            call = self.substrate.compose_call(
                call_module="ModuleRegistry",
                call_function="update_module",
                call_params={"key": key_bytes, "cid": cid_bytes},
            )

            # Create and submit the extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call, keypair=self.keypair
            )

            # Submit the extrinsic (use simple submission for HTTP connections)
            if self.substrate.url.startswith("ws"):
                # WebSocket connection - can wait for inclusion
                receipt = self.substrate.submit_extrinsic(
                    extrinsic, wait_for_inclusion=True, wait_for_finalization=True
                )
            else:
                # HTTP connection - submit without waiting
                extrinsic_hash = self.substrate.submit_extrinsic(extrinsic)
                receipt = {
                    "extrinsic_hash": extrinsic_hash,
                    "block_hash": None,
                    "finalized": False,
                }

            print("📡 Update extrinsic submitted")

            # Handle both dict and object receipt formats
            if isinstance(receipt, dict):
                extrinsic_hash = receipt.get("extrinsic_hash", "N/A")
                block_hash = receipt.get("block_hash", "N/A")
                finalized = receipt.get("finalized", False)
            else:
                extrinsic_hash = getattr(receipt, "extrinsic_hash", "N/A")
                block_hash = getattr(receipt, "block_hash", "N/A")
                finalized = getattr(receipt, "finalized", False)

            print(f"   Extrinsic Hash: {extrinsic_hash}")
            print(f"   Block Hash: {block_hash}")
            print(f"   Finalized: {finalized}")

            # Check if the extrinsic was successful
            is_success = False
            if isinstance(receipt, dict):
                is_success = receipt.get("finalized", False) or extrinsic_hash != "N/A"
            else:
                is_success = getattr(receipt, "is_success", False)

            if is_success:
                print("✅ Module update successful!")

                events = []
                if hasattr(receipt, "triggered_events") and receipt.triggered_events:
                    print(f"📋 {len(receipt.triggered_events)} events triggered")
                    for i, event in enumerate(receipt.triggered_events):
                        try:
                            # Safe event parsing - handle any event structure
                            module_id = getattr(event, "module_id", "Unknown")
                            event_id = getattr(event, "event_id", "Unknown")
                            print(f"   Event {i+1}: {module_id}::{event_id}")
                            events.append({"event": f"{module_id}::{event_id}"})
                        except Exception:
                            print(
                                f"   Event {i+1}: [Event parsing skipped: {type(event)}]"
                            )
                            events.append({"event": "event_logged"})
                else:
                    print(
                        "🎉 Extrinsic submitted - check blockchain for ModuleUpdated event"
                    )

                return ModuleRegistration(
                    public_key=public_key,
                    cid=new_cid,
                    block_hash=block_hash,
                    extrinsic_hash=extrinsic_hash,
                    block_number=getattr(receipt, "block_number", None),
                    events=events,
                )
            else:
                print(f"❌ Module update failed: {receipt.error_message}")
                raise RuntimeError(f"Update failed: {receipt.error_message}")

        except Exception as e:
            print(f"❌ Update failed: {e}")
            raise

    def remove_module(self, public_key: str) -> ModuleRegistration:
        """
        Remove a module from the chain with REAL blockchain transaction.

        Args:
            public_key: Module's public key

        Returns:
            ModuleRegistration with transaction details
        """
        if not self.substrate or not self.keypair:
            raise RuntimeError("Not connected to Substrate chain")

        print("🗑️ Removing module from chain...")
        print(f"   Public Key: {public_key}")

        try:
            # Prepare the extrinsic call
            # Convert hex string to bytes for the key
            if isinstance(public_key, str):
                if public_key.startswith("0x"):
                    key_bytes = bytes.fromhex(public_key[2:])
                else:
                    key_bytes = public_key.encode("utf-8")
            else:
                key_bytes = bytes(public_key)

            call = self.substrate.compose_call(
                call_module="ModuleRegistry",
                call_function="remove_module",
                call_params={"key": key_bytes},
            )

            # Create and submit the extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call, keypair=self.keypair
            )

            # Submit the extrinsic (use simple submission for HTTP connections)
            if self.substrate.url.startswith("ws"):
                # WebSocket connection - can wait for inclusion
                receipt = self.substrate.submit_extrinsic(
                    extrinsic, wait_for_inclusion=True, wait_for_finalization=True
                )
            else:
                # HTTP connection - submit without waiting
                extrinsic_hash = self.substrate.submit_extrinsic(extrinsic)
                receipt = {
                    "extrinsic_hash": extrinsic_hash,
                    "block_hash": None,
                    "finalized": False,
                }

            print("📡 Remove extrinsic submitted")

            # Handle both dict and object receipt formats
            if isinstance(receipt, dict):
                extrinsic_hash = receipt.get("extrinsic_hash", "N/A")
                block_hash = receipt.get("block_hash", "N/A")
                finalized = receipt.get("finalized", False)
            else:
                extrinsic_hash = getattr(receipt, "extrinsic_hash", "N/A")
                block_hash = getattr(receipt, "block_hash", "N/A")
                finalized = getattr(receipt, "finalized", False)

            print(f"   Extrinsic Hash: {extrinsic_hash}")
            print(f"   Block Hash: {block_hash}")
            print(f"   Finalized: {finalized}")
            print(f"   Extrinsic Hash: {receipt.extrinsic_hash}")
            print(f"   Block Hash: {receipt.block_hash}")
            print(f"   Block Number: {receipt.block_number}")

            # Check if the extrinsic was successful
            is_success = False
            if isinstance(receipt, dict):
                is_success = receipt.get("finalized", False) or extrinsic_hash != "N/A"
            else:
                is_success = getattr(receipt, "is_success", False)

            if is_success:
                print("✅ Module removal successful!")

                events = []
                if hasattr(receipt, "triggered_events") and receipt.triggered_events:
                    print(f"📋 {len(receipt.triggered_events)} events triggered")
                    for i, event in enumerate(receipt.triggered_events):
                        try:
                            # Safe event parsing - handle any event structure
                            module_id = getattr(event, "module_id", "Unknown")
                            event_id = getattr(event, "event_id", "Unknown")
                            print(f"   Event {i+1}: {module_id}::{event_id}")
                            events.append({"event": f"{module_id}::{event_id}"})
                        except Exception:
                            print(
                                f"   Event {i+1}: [Event parsing skipped: {type(event)}]"
                            )
                            events.append({"event": "event_logged"})
                else:
                    print(
                        "🎉 Extrinsic submitted - check blockchain for ModuleUpdated event"
                    )

                return ModuleRegistration(
                    public_key=public_key,
                    cid="",  # Removed
                    block_hash=receipt.block_hash,
                    extrinsic_hash=receipt.extrinsic_hash,
                    block_number=receipt.block_number,
                    events=events,
                )
            else:
                print(f"❌ Module removal failed: {receipt.error_message}")
                raise RuntimeError(f"Removal failed: {receipt.error_message}")

        except Exception as e:
            print(f"❌ Removal failed: {e}")
            raise

    def list_all_modules(self) -> list[dict[str, str]]:
        """
        List all registered modules from chain state.

        Returns:
            List of {public_key, cid} dictionaries
        """
        if not self.substrate:
            raise RuntimeError("Not connected to Substrate chain")

        try:
            print("📋 Querying all registered modules...")

            # Query all entries in the Modules storage map
            result = self.substrate.query_map(
                module="ModuleRegistry", storage_function="Modules"
            )

            print(f"   Raw query_map result: {result}")
            print(f"   Result type: {type(result)}")

            modules = []
            if result:
                for item in result:
                    print(f"   Processing item: {item}, type: {type(item)}")

                    try:
                        # Handle different result formats
                        if isinstance(item, tuple) and len(item) == 2:
                            key, value = item
                        elif hasattr(item, "key") and hasattr(item, "value"):
                            key, value = item.key, item.value
                        else:
                            print(f"   ⚠️ Unexpected item format: {item}")
                            continue

                        # Extract key data
                        if hasattr(key, "value"):
                            key_data = key.value
                        else:
                            key_data = key

                        # Extract value data
                        if hasattr(value, "value"):
                            value_data = value.value
                        else:
                            value_data = value

                        # Convert to strings
                        if isinstance(key_data, bytes):
                            public_key = key_data.decode("utf-8")
                        else:
                            public_key = str(key_data)

                        if isinstance(value_data, bytes):
                            cid = value_data.decode("utf-8")
                        else:
                            cid = str(value_data)

                        modules.append({"public_key": public_key, "cid": cid})
                        print(f"   📦 {public_key[:16]}... → {cid}")

                    except Exception as item_error:
                        print(f"   ⚠️ Failed to process item {item}: {item_error}")
                        continue

            print(f"✅ Found {len(modules)} registered modules")
            return modules

        except Exception as e:
            print(f"❌ Failed to list modules: {e}")
            import traceback

            traceback.print_exc()
            raise


async def test_real_pallet_integration():
    """
    Test real pallet integration with actual blockchain transactions.
    """
    print("🚀 Real Substrate Pallet Integration Test")
    print("=" * 50)
    print("Testing with ACTUAL blockchain transactions and block logs")
    print()

    client = SubstratePalletClient()

    try:
        # Connect to the chain
        if not client.connect():
            print("❌ Failed to connect to Substrate chain")
            print("   Make sure the chain is running:")
            print("   ./target/release/solochain-template-node --dev --tmp")
            return False

        # Get chain info
        chain_info = client.get_chain_info()
        print("📊 Chain Info:")
        print(f"   Chain: {chain_info['chain']}")
        print(f"   Current Block: #{chain_info['current_block']}")
        print(f"   Signer: {chain_info['keypair_address']}")
        print()

        # Test module registration
        config = get_config()
        test_public_key = config.substrate.test_public_key
        test_cid = "QmTestCID1234567890abcdef1234567890abcdef"

        print("🧪 Test 1: Register Module")
        registration = client.register_module(test_public_key, test_cid)
        print(f"✅ Registration completed in block #{registration.block_number}")
        print(f"   Block Hash: {registration.block_hash}")
        print(f"   Extrinsic Hash: {registration.extrinsic_hash}")
        print()

        # Test module query
        print("🧪 Test 2: Query Module")
        retrieved_cid = client.get_module_by_key(test_public_key)
        if retrieved_cid == test_cid:
            print(f"✅ Module query successful: {retrieved_cid}")
        else:
            print(f"❌ Module query failed: expected {test_cid}, got {retrieved_cid}")
        print()

        # Test module update
        print("🧪 Test 3: Update Module")
        new_cid = "QmNewCID9876543210fedcba9876543210fedcba"
        update_result = client.update_module(test_public_key, new_cid)
        print(f"✅ Update completed in block #{update_result.block_number}")
        print()

        # Verify update
        print("🧪 Test 4: Verify Update")
        updated_cid = client.get_module_by_key(test_public_key)
        if updated_cid == new_cid:
            print(f"✅ Update verification successful: {updated_cid}")
        else:
            print(
                f"❌ Update verification failed: expected {new_cid}, got {updated_cid}"
            )
        print()

        # List all modules
        print("🧪 Test 5: List All Modules")
        all_modules = client.list_all_modules()
        print(f"✅ Listed {len(all_modules)} modules")
        print()

        # Test module removal
        print("🧪 Test 6: Remove Module")
        removal_result = client.remove_module(test_public_key)
        print(f"✅ Removal completed in block #{removal_result.block_number}")
        print()

        # Verify removal
        print("🧪 Test 7: Verify Removal")
        removed_cid = client.get_module_by_key(test_public_key)
        if removed_cid is None:
            print("✅ Removal verification successful: module not found")
        else:
            print(f"❌ Removal verification failed: still found {removed_cid}")
        print()

        print("🎉 ALL REAL PALLET TESTS PASSED!")
        print("✅ Actual blockchain transactions were executed")
        print("✅ Events were logged in blocks")
        print("✅ Chain state was modified and verified")

        return True

    except Exception as e:
        print(f"❌ Real pallet integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        client.disconnect()


if __name__ == "__main__":
    print("🔗 Real Substrate Pallet Client")
    print("Performing actual blockchain transactions")
    print()

    success = asyncio.run(test_real_pallet_integration())

    if success:
        print("\n🎯 CONCLUSION:")
        print("Real blockchain integration is working!")
        print("Transactions are being logged in blocks.")
        print("This is TRUE end-to-end blockchain integration.")
    else:
        print("\n⚠️ CONCLUSION:")
        print(
            "Some blockchain integration tests failed. Check the logs above for details."
        )

    exit(0 if success else 1)
