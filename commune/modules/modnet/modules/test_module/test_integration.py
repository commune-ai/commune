#!/usr/bin/env python3
"""
Integration Test Script for Test Module

This script demonstrates:
1. Test module functionality
2. Integration with Module Registry
3. IPFS metadata storage workflow
4. End-to-end testing scenarios
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import the module
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Local imports after path modification - isort:skip
from module import TestModule  # noqa: E402

from scripts.config import get_config  # noqa: E402


async def test_basic_functionality():
    """Test basic module functionality."""
    print("🧪 Testing Basic Module Functionality")
    print("=" * 50)

    # Initialize test module
    module = TestModule(name="test-integration-module")

    # Test info function
    print("\n📋 Module Info:")
    info = module.info()
    print(json.dumps(info, indent=2, default=str))

    # Test computational functions
    print("\n🧮 Testing Computational Functions:")

    # Test compute function
    compute_result = module.compute("add", 5, 10, 15)
    print(f"Compute (add 5+10+15): {compute_result['result']}")
    assert compute_result["result"] == 30, "Addition test failed"

    # Test fibonacci
    fib_result = module.fibonacci(7)
    print(f"Fibonacci(7): {fib_result['sequence']}")
    expected_fib = [0, 1, 1, 2, 3, 5, 8]
    assert fib_result["sequence"] == expected_fib, "Fibonacci test failed"

    # Test prime check
    prime_result = module.prime_check([7, 8, 9, 10, 11])
    prime_numbers = [r["number"] for r in prime_result["results"] if r["is_prime"]]
    print(f"Prime numbers in [7,8,9,10,11]: {prime_numbers}")
    assert prime_numbers == [7, 11], "Prime check test failed"

    # Test data transform
    transform_result = module.data_transform("hello world", "reverse")
    print(f"Reverse 'hello world': {transform_result['result']}")
    assert transform_result["result"] == "dlrow olleh", "Data transform test failed"

    print("✅ All basic functionality tests passed!")
    return module


async def test_forward_mechanism():
    """Test the forward mechanism (commune pattern)."""
    print("\n🔄 Testing Forward Mechanism")
    print("=" * 30)

    module = TestModule(name="forward-test-module")

    # Test forward calls
    try:
        # Test valid forward call
        result = module.forward("compute", "multiply", 3, 4, 5)
        print(f"Forward compute multiply: {result['result']}")
        assert result["result"] == 60, "Forward multiply test failed"

        # Test invalid function call
        try:
            module.forward("nonexistent_function")
            raise AssertionError("Should have raised AttributeError")
        except AttributeError as e:
            print(f"✅ Correctly caught invalid function: {e}")

        # Test unexposed function call (if we had any)
        print("✅ Forward mechanism tests passed!")

    except Exception as e:
        print(f"❌ Forward mechanism test failed: {e}")
        raise


async def test_health_and_metadata():
    """Test health check and metadata functions."""
    print("\n🏥 Testing Health Check and Metadata")
    print("=" * 40)

    module = TestModule(name="health-test-module")

    # Test health check
    health = module.health_check()
    print(f"Health Status: {health['status']}")
    print(f"All Tests Passed: {health['all_tests_passed']}")

    if not health["all_tests_passed"]:
        print("❌ Health check failed:")
        for test_name, passed in health["tests"].items():
            status = "✅" if passed else "❌"
            print(f"  {status} {test_name}")
    else:
        print("✅ Health check passed!")

    # Test metadata
    metadata = module.get_metadata()
    print("\n📋 Module Metadata:")
    print(f"  Name: {metadata['name']}")
    print(f"  Version: {metadata['version']}")
    print(f"  Description: {metadata['description']}")
    print(f"  Tags: {metadata['tags']}")

    assert metadata["name"] == "health-test-module", "Metadata name mismatch"
    assert "runtime_info" in metadata, "Missing runtime info in metadata"

    print("✅ Health and metadata tests passed!")


async def test_registry_integration():
    """Test Module Registry integration."""
    print("\n🗂️ Testing Module Registry Integration")
    print("=" * 40)

    config = get_config()
    module = TestModule(
        name="registry-test-module", registry_url=config.test.test_module_registry_url
    )

    # Test registry registration (mock)
    registration_result = await module.register_in_registry()
    print(f"Registration Status: {registration_result['status']}")

    if registration_result["status"] == "registered":
        print(f"✅ Module registered with CID: {registration_result['cid']}")
        print(f"Registry URL: {registration_result['registry_url']}")
    else:
        print(f"⚠️ Registration returned: {registration_result}")

    # In a real scenario, we would test with actual commune-ipfs backend:
    print("\n💡 Note: This is a mock registration.")
    print("   For real integration, ensure commune-ipfs backend is running.")
    print("   The integration_client.py can be used for actual IPFS operations.")


async def test_comprehensive_module_test():
    """Run the module's built-in comprehensive test suite."""
    print("\n🧪 Running Comprehensive Module Test Suite")
    print("=" * 45)

    module = TestModule(name="comprehensive-test-module")

    # Run the module's test method
    test_results = module.test()

    print("\n📊 Test Results Summary:")
    summary = test_results["summary"]
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")

    if summary["overall_passed"]:
        print("✅ All comprehensive tests passed!")
    else:
        print("❌ Some comprehensive tests failed:")
        for test_name, test_result in test_results["tests"].items():
            status = "✅" if test_result.get("passed", False) else "❌"
            print(f"  {status} {test_name}")
            if "error" in test_result:
                print(f"      Error: {test_result['error']}")

    return test_results["summary"]["overall_passed"]


async def test_performance_benchmarks():
    """Run performance benchmarks on the module."""
    print("\n⚡ Performance Benchmarks")
    print("=" * 25)

    module = TestModule(name="benchmark-module")

    # Benchmark fibonacci calculation
    print("Fibonacci Performance:")
    for n in [10, 20, 30]:
        result = module.fibonacci(n, method="iterative")
        print(f"  fib({n}): {result['duration']:.4f}s")

    # Benchmark prime checking
    print("\nPrime Check Performance:")
    test_numbers = list(range(100, 200))
    result = module.prime_check(test_numbers)
    print(f"  Checked {len(test_numbers)} numbers in {result['duration']:.4f}s")
    print(f"  Found {result['prime_count']} primes")

    # Benchmark compute operations
    print("\nCompute Operations Performance:")
    operations = [
        ("add", list(range(1000))),
        ("multiply", [2, 3, 4, 5]),
        ("factorial", [10]),
    ]

    for op, args in operations:
        result = module.compute(op, *args)
        if "duration" in result:
            print(f"  {op}: {result['duration']:.4f}s")

    print("✅ Performance benchmarks completed!")


async def demonstrate_real_world_usage():
    """Demonstrate real-world usage scenarios."""
    print("\n🌍 Real-World Usage Demonstration")
    print("=" * 35)

    # Scenario 1: Data processing pipeline
    print("Scenario 1: Data Processing Pipeline")
    module = TestModule(name="data-processor")

    # Process some sample data
    raw_data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

    # Sort the data
    sorted_result = module.data_transform(raw_data, "sort")
    print(f"  Sorted data: {sorted_result['result']}")

    # Check which numbers are prime
    prime_result = module.prime_check(raw_data)
    primes = [r["number"] for r in prime_result["results"] if r["is_prime"]]
    print(f"  Prime numbers: {primes}")

    # Calculate some statistics
    sum_result = module.compute("add", *raw_data)
    print(f"  Sum: {sum_result['result']}")

    # Scenario 2: Mathematical computation service
    print("\nScenario 2: Mathematical Computation Service")
    math_module = TestModule(name="math-service")

    # Calculate factorials
    factorials = []
    for i in range(1, 6):
        result = math_module.compute("factorial", i)
        factorials.append(f"{i}! = {result['result']}")
    print(f"  Factorials: {', '.join(factorials)}")

    # Generate Fibonacci sequence
    fib_result = math_module.fibonacci(12)
    print(f"  Fibonacci(12): {fib_result['sequence']}")

    print("✅ Real-world usage scenarios completed!")


async def main():
    """Main test execution function."""
    print("🚀 Test Module Integration Test Suite")
    print("=" * 50)
    print("Testing the Test Module for Mod-Net Module Registry")
    print()

    try:
        # Run all test suites
        await test_basic_functionality()
        await test_forward_mechanism()
        await test_health_and_metadata()
        await test_registry_integration()

        comprehensive_passed = await test_comprehensive_module_test()

        await test_performance_benchmarks()
        await demonstrate_real_world_usage()

        # Final summary
        print("\n" + "=" * 50)
        print("🎉 Integration Test Suite Completed!")

        if comprehensive_passed:
            print("✅ All tests passed successfully!")
            print("\n💡 Next Steps:")
            print(
                "   1. Start commune-ipfs backend: cd commune-ipfs && uv run python main.py"
            )
            print("   2. Run integration_client.py for real IPFS integration")
            print("   3. Test with actual Substrate pallet calls")
            print("   4. Deploy module to commune network")
        else:
            print("⚠️ Some tests failed - check output above for details")

        return comprehensive_passed

    except Exception as e:
        print(f"\n❌ Integration test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the integration tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
