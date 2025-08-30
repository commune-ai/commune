#!/usr/bin/env python3
"""
Storage Query Debug Script

This script focuses on debugging the storage query issues by:
1. Testing different key encoding methods
2. Examining storage map structure
3. Verifying data persistence after successful transactions
"""

import asyncio

from substrate_pallet_client import SubstratePalletClient

from .config import get_config


async def debug_storage_queries():
    """Debug storage query issues step by step."""
    print("🔍 Storage Query Debug Session")
    print("=" * 50)

    client = SubstratePalletClient()
    config = get_config()

    try:
        # Connect to chain
        if not client.connect():
            print("❌ Failed to connect to chain")
            return False

        print("✅ Connected to chain")

        # Test key that should exist (from previous ModuleAlreadyExists error)
        test_key = config.substrate.test_public_key

        print(f"\n🧪 Testing storage queries for key: {test_key}")

        # Try different key encoding methods
        key_variants = [
            ("Raw string", test_key),
            ("UTF-8 bytes", test_key.encode("utf-8")),
            (
                "Hex decoded",
                (
                    bytes.fromhex(test_key[2:])
                    if test_key.startswith("0x")
                    else test_key.encode()
                ),
            ),
        ]

        for desc, key_variant in key_variants:
            print(f"\n📋 Trying {desc}: {key_variant}")
            try:
                if client.substrate is None:
                    raise RuntimeError("Client not properly connected")

                result = client.substrate.query(
                    module="ModuleRegistry",
                    storage_function="Modules",
                    params=[key_variant],
                )
                print(f"   Result: {result}")
                print(f"   Result value: {result.value}")
                print(f"   Result type: {type(result.value)}")

                if result.value is not None:
                    print(f"   ✅ Found data with {desc}!")
                    break

            except Exception as e:
                print(f"   ❌ Error with {desc}: {e}")

        # Try to list all storage entries to see what's actually stored
        print("\n📋 Attempting to list all storage entries...")
        try:
            if client.substrate is None:
                raise RuntimeError("Client not properly connected")

            result = client.substrate.query_map(
                module="ModuleRegistry", storage_function="Modules"
            )
            print(f"   Query map result: {result}")
            print(f"   Query map type: {type(result)}")

            if result:
                print(f"   Found {len(result)} entries:")
                for i, item in enumerate(result):
                    print(f"     Entry {i}: {item}")
                    print(f"     Entry type: {type(item)}")
                    if hasattr(item, "key") and hasattr(item, "value"):
                        print(f"       Key: {item.key} (type: {type(item.key)})")
                        print(f"       Value: {item.value} (type: {type(item.value)})")
            else:
                print("   No entries found in storage map")

        except Exception as e:
            print(f"   ❌ Query map error: {e}")
            import traceback

            traceback.print_exc()

        # Try to register a new module to see the storage in action
        print("\n🧪 Attempting fresh registration to observe storage...")
        # Generate a different test key for new registration
        new_test_key = (
            "0x" + config.substrate.test_public_key[2:34] + "abcdef1234567890abcdef12"
        )
        new_test_cid = "QmNewTestCID1234567890abcdef1234567890"

        try:
            client.register_module(new_test_key, new_test_cid)
            print("✅ Fresh registration successful!")

            # Immediately query the fresh registration
            print("🔍 Querying fresh registration...")
            fresh_result = client.get_module_by_key(new_test_key)
            print(f"   Fresh query result: {fresh_result}")

        except Exception as e:
            print(f"❌ Fresh registration failed: {e}")
            if "ModuleAlreadyExists" in str(e):
                print("   (Module already exists - good, storage is working)")

        return True

    except Exception as e:
        print(f"❌ Debug session failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        client.disconnect()


if __name__ == "__main__":
    print("🔗 Storage Query Debugging")
    print("Investigating storage key encoding issues")
    print()

    success = asyncio.run(debug_storage_queries())

    if success:
        print("\n🎯 DEBUG COMPLETE")
        print("Check output above for storage query insights")
    else:
        print("\n⚠️ DEBUG FAILED")
        print("Storage query issues need further investigation")

    exit(0 if success else 1)
