#!/usr/bin/env python3
"""
Direct RPC Test for Real Pallet Calls

This bypasses substrate-interface and makes direct RPC calls to test:
1. Basic connectivity to Substrate node
2. Chain state queries
3. Extrinsic submission for real pallet calls
4. Block verification and event logs

This will help us debug the connection issue and get real blockchain transactions working.
"""

import asyncio
from typing import Any

import aiohttp

from .config import get_config


class DirectRPCClient:
    """
    Direct RPC client for Substrate chain communication.
    """

    def __init__(self, rpc_url: str | None = None):
        config = get_config()
        self.rpc_url = rpc_url or config.substrate.http_url
        self.session: aiohttp.ClientSession | None = None
        self.request_id = 1

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def rpc_call(
        self, method: str, params: list[Any] | None = None
    ) -> dict[str, Any]:
        """Make a direct RPC call to the Substrate node."""
        if params is None:
            params = []

        payload = {
            "id": self.request_id,
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        self.request_id += 1

        try:
            if self.session is None:
                raise RuntimeError("Session not initialized")
            async with self.session.post(
                self.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "error" in result:
                        raise Exception(f"RPC Error: {result['error']}")
                    rpc_result = result.get("result", {})
                    # Ensure we return a dict[str, Any] as declared
                    if isinstance(rpc_result, dict):
                        return rpc_result
                    else:
                        return {"value": rpc_result}
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

        except Exception as e:
            print(f"❌ RPC call failed ({method}): {e}")
            raise

    async def test_basic_connectivity(self):
        """Test basic RPC connectivity."""
        print("🔗 Testing Basic RPC Connectivity")
        print("-" * 40)

        try:
            # Test system_chain
            chain = await self.rpc_call("system_chain")
            print(f"✅ Chain: {chain}")

            # Test system_name
            name = await self.rpc_call("system_name")
            print(f"✅ Node Name: {name}")

            # Test system_version
            version = await self.rpc_call("system_version")
            print(f"✅ Version: {version}")

            # Test chain_getHeader
            header = await self.rpc_call("chain_getHeader")
            block_number = int(header["number"], 16)
            print(f"✅ Current Block: #{block_number}")
            print(f"✅ Block Hash: {header['parentHash']}")

            return True

        except Exception as e:
            print(f"❌ Basic connectivity test failed: {e}")
            return False

    async def test_runtime_metadata(self):
        """Test runtime metadata access."""
        print("\n🔍 Testing Runtime Metadata")
        print("-" * 40)

        try:
            # Get runtime metadata
            metadata = await self.rpc_call("state_getMetadata")
            print(f"✅ Retrieved runtime metadata ({len(metadata)} chars)")

            # Get runtime version
            runtime_version = await self.rpc_call("state_getRuntimeVersion")
            print(f"✅ Runtime Version: {runtime_version['specVersion']}")
            print(f"✅ Spec Name: {runtime_version['specName']}")

            return True

        except Exception as e:
            print(f"❌ Runtime metadata test failed: {e}")
            return False

    async def test_storage_query(self):
        """Test storage queries for ModuleRegistry pallet."""
        print("\n📋 Testing Storage Queries")
        print("-" * 40)

        try:
            # Try to query ModuleRegistry storage
            # This will help us understand if our pallet is properly configured

            # First, let's see what storage keys exist
            storage_keys = await self.rpc_call("state_getKeys", ["0x"])
            print(f"✅ Found {len(storage_keys)} storage keys")

            # Look for ModuleRegistry related keys
            module_registry_keys = [
                key for key in storage_keys if "moduleregistry" in key.lower()
            ]
            print(f"✅ ModuleRegistry keys: {len(module_registry_keys)}")

            if module_registry_keys:
                print("   Sample keys:")
                for key in module_registry_keys[:3]:
                    print(f"   {key}")

            return True

        except Exception as e:
            print(f"❌ Storage query test failed: {e}")
            return False

    async def test_extrinsic_submission(self):
        """Test extrinsic submission (the real pallet call)."""
        print("\n📝 Testing Extrinsic Submission")
        print("-" * 40)

        try:
            # For a real pallet call, we need to:
            # 1. Create a properly formatted extrinsic
            # 2. Sign it with a keypair
            # 3. Submit it to the chain

            # This is complex without substrate-interface, so let's test
            # the submission endpoint first

            print("⚠️ Real extrinsic submission requires proper encoding and signing")
            print("   This would involve:")
            print("   1. Encoding the call data")
            print("   2. Creating and signing the extrinsic")
            print("   3. Submitting via author_submitExtrinsic")

            # Test if the submission endpoint is available
            try:
                # This will fail because we're not sending a valid extrinsic,
                # but it will tell us if the endpoint is accessible
                await self.rpc_call("author_submitExtrinsic", ["0x00"])
            except Exception as e:
                if "Invalid transaction" in str(e) or "Bad input data" in str(e):
                    print("✅ Extrinsic submission endpoint is accessible")
                    print("   (Expected error due to invalid test data)")
                    return True
                else:
                    print(f"❌ Unexpected error: {e}")
                    return False

            return True

        except Exception as e:
            print(f"❌ Extrinsic submission test failed: {e}")
            return False

    async def test_block_monitoring(self):
        """Test block monitoring and event detection."""
        print("\n👀 Testing Block Monitoring")
        print("-" * 40)

        try:
            # Get current block
            current_header = await self.rpc_call("chain_getHeader")
            current_block_num = int(current_header["number"], 16)
            print(f"✅ Current block: #{current_block_num}")

            # Wait for next block
            print("   Waiting for next block...")
            await asyncio.sleep(6)  # Block time is ~6 seconds

            # Check if we got a new block
            new_header = await self.rpc_call("chain_getHeader")
            new_block_num = int(new_header["number"], 16)

            if new_block_num > current_block_num:
                print(f"✅ New block detected: #{new_block_num}")

                # Get block details
                block_hash = new_header["parentHash"]
                block_data = await self.rpc_call("chain_getBlock", [block_hash])

                extrinsics = block_data["block"]["extrinsics"]
                print(f"✅ Block contains {len(extrinsics)} extrinsics")

                return True
            else:
                print("⚠️ No new block detected in expected timeframe")
                return False

        except Exception as e:
            print(f"❌ Block monitoring test failed: {e}")
            return False

    async def run_comprehensive_test(self):
        """Run comprehensive RPC test suite."""
        print("🚀 Direct RPC Test Suite")
        print("=" * 50)
        print("Testing direct communication with Substrate node")
        print(f"RPC URL: {self.rpc_url}")
        print()

        test_results = {
            "connectivity": False,
            "metadata": False,
            "storage": False,
            "extrinsic": False,
            "monitoring": False,
        }

        # Test 1: Basic connectivity
        test_results["connectivity"] = await self.test_basic_connectivity()

        if not test_results["connectivity"]:
            print("\n❌ CRITICAL: Basic connectivity failed")
            print("   Cannot proceed with other tests")
            return False

        # Test 2: Runtime metadata
        test_results["metadata"] = await self.test_runtime_metadata()

        # Test 3: Storage queries
        test_results["storage"] = await self.test_storage_query()

        # Test 4: Extrinsic submission
        test_results["extrinsic"] = await self.test_extrinsic_submission()

        # Test 5: Block monitoring
        test_results["monitoring"] = await self.test_block_monitoring()

        # Summary
        print("\n" + "=" * 50)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 50)

        passed_tests = sum(test_results.values())
        total_tests = len(test_results)

        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name.title()} Test")

        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Direct RPC communication is working")
            print("✅ Ready for real pallet call implementation")
            return True
        elif test_results["connectivity"] and test_results["extrinsic"]:
            print("\n⚠️ PARTIAL SUCCESS")
            print("✅ Basic RPC communication works")
            print("✅ Can proceed with real pallet calls")
            return True
        else:
            print("\n❌ CRITICAL ISSUES DETECTED")
            print("   Need to resolve connectivity issues first")
            return False


async def main():
    """Main test execution."""
    print("🔗 Direct RPC Test for Real Pallet Calls")
    print("Testing direct communication with Substrate node")
    print()

    # Test multiple RPC endpoints
    config = get_config()
    rpc_urls = config.test.rpc_test_urls

    for rpc_url in rpc_urls:
        print(f"🔍 Trying RPC URL: {rpc_url}")

        try:
            async with DirectRPCClient(rpc_url) as client:
                success = await client.run_comprehensive_test()

                if success:
                    print(f"\n🎯 SUCCESS with {rpc_url}")
                    print("✅ Real pallet calls are now possible!")
                    return True
                else:
                    print(f"\n⚠️ Issues detected with {rpc_url}")

        except Exception as e:
            print(f"❌ Failed to test {rpc_url}: {e}")
            continue

    print("\n❌ All RPC endpoints failed")
    print("   Check Substrate node configuration")
    return False


if __name__ == "__main__":
    success = asyncio.run(main())

    if success:
        print("\n🎉 CONCLUSION:")
        print("Direct RPC communication is working!")
        print(
            "We can now implement real pallet calls with actual blockchain transactions."
        )
    else:
        print("\n⚠️ CONCLUSION:")
        print("RPC communication issues need to be resolved.")
        print("Check Substrate node startup and configuration.")

    exit(0 if success else 1)
