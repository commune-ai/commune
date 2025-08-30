#!/usr/bin/env python3
"""
Test Module for Mod-Net Module Registry

This is a demonstration module that shows how to create a commune-compatible module
that can be registered in the Module Registry system with IPFS metadata storage.

The module follows the commune pattern:
1. Inherits from or works with the Mod base class
2. Provides discoverable functions via expose/fns attributes
3. Can be served as a network-accessible service
4. Integrates with the Module Registry for decentralized discovery
"""

import json
import time
from datetime import datetime
from typing import Any

from scripts.config import get_config

# Import commune if available, otherwise create a minimal mock
try:
    import commune as c

    HAS_COMMUNE = True
except ImportError:
    HAS_COMMUNE = False

    # Create minimal mock for standalone operation
    class MockMod:
        def print(self, *args, **kwargs):
            print(*args)

        def time(self):
            return time.time()

    c = MockMod()


class TestModule:
    """
    Test Module for demonstrating Module Registry integration.

    This module provides:
    - Basic computational functions
    - Module registry integration
    - IPFS metadata management
    - Network serving capabilities
    """

    # Commune module attributes
    expose = [
        "info",
        "forward",
        "compute",
        "fibonacci",
        "prime_check",
        "data_transform",
        "get_metadata",
        "register_in_registry",
        "health_check",
    ]

    # Module metadata for registry
    metadata = {
        "name": "test-module",
        "version": "1.0.0",
        "description": "A test module demonstrating Module Registry integration with computational functions",
        "author": "mod-net-developer@example.com",
        "license": "MIT",
        "repository": "https://github.com/Bakobiibizo/mod-net-modules",
        "dependencies": ["commune", "asyncio", "json"],
        "tags": ["test", "computation", "fibonacci", "prime", "demo"],
        "chain_type": "ed25519",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    def __init__(
        self,
        name: str = "test-module",
        version: str = "1.0.0",
        description: str = "A test module for the Module Registry",
        public_key: str | None = None,
        registry_url: str | None = None,
        **kwargs,
    ):
        config = get_config()
        """
        Initialize the test module.

        Args:
            name: Module name for identification
            public_key: Public key for module registry (will generate if None)
            registry_url: URL of the module registry backend
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.version = version
        self.description = description
        self.registry_url = registry_url or config.test.test_module_registry_url
        self.public_key = public_key or self.generate_mock_key()
        self.call_count = 0
        self.start_time = time.time()

        # Update metadata with instance-specific info
        self.metadata.update(
            {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "public_key": self.public_key,
                "updated_at": datetime.now().isoformat(),
            }
        )

        c.print(f"🚀 TestModule '{self.name}' initialized", color="green")

    def generate_mock_key(self) -> str:
        """Generate a mock public key for testing purposes."""
        import hashlib

        key_material = f"test-module-{time.time()}-{self.name}"
        return "0x" + hashlib.sha256(key_material.encode()).hexdigest()[:40]

    def info(self) -> dict[str, Any]:
        """
        Get module information (required by commune pattern).

        Returns:
            Dictionary containing module info
        """
        uptime = time.time() - self.start_time
        return {
            "name": self.name,
            "version": self.metadata["version"],
            "description": self.metadata["description"],
            "author": self.metadata["author"],
            "public_key": self.public_key,
            "uptime": uptime,
            "call_count": self.call_count,
            "status": "active",
            "functions": self.expose,
            "registry_url": self.registry_url,
            "timestamp": c.time() if HAS_COMMUNE else time.time(),
        }

    def forward(self, fn: str, *args, **kwargs) -> Any:
        """
        Forward function calls (commune pattern).

        Args:
            fn: Function name to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function call
        """
        self.call_count += 1

        if not hasattr(self, fn):
            raise AttributeError(f"Function '{fn}' not found in module")

        if fn not in self.expose:
            raise PermissionError(f"Function '{fn}' is not exposed")

        func = getattr(self, fn)
        return func(*args, **kwargs)

    def compute(self, operation: str, *args, **kwargs) -> dict[str, Any]:
        """
        Perform various computational operations.

        Args:
            operation: Type of computation ('add', 'multiply', 'power', etc.)
            *args: Operation arguments
            **kwargs: Additional parameters

        Returns:
            Computation result with metadata
        """
        start_time = time.time()

        try:
            if operation == "add":
                result = sum(args)
            elif operation == "multiply":
                result = 1
                for arg in args:
                    result *= arg
            elif operation == "power":
                if len(args) != 2:
                    raise ValueError("Power operation requires exactly 2 arguments")
                result = args[0] ** args[1]
            elif operation == "factorial":
                if len(args) != 1 or not isinstance(args[0], int) or args[0] < 0:
                    raise ValueError("Factorial requires a single non-negative integer")
                n = args[0]
                result = 1
                for i in range(1, n + 1):
                    result *= i
            else:
                raise ValueError(f"Unknown operation: {operation}")

            duration = time.time() - start_time

            return {
                "operation": operation,
                "arguments": args,
                "result": result,
                "duration": duration,
                "timestamp": time.time(),
                "module": self.name,
            }

        except Exception as e:
            return {
                "operation": operation,
                "arguments": args,
                "error": str(e),
                "duration": time.time() - start_time,
                "timestamp": time.time(),
                "module": self.name,
            }

    def fibonacci(self, n: int, method: str = "iterative") -> dict[str, Any]:
        """
        Calculate Fibonacci sequence.

        Args:
            n: Number of Fibonacci numbers to calculate
            method: Calculation method ('iterative' or 'recursive')

        Returns:
            Fibonacci sequence and metadata
        """
        start_time = time.time()

        try:
            if n < 0:
                raise ValueError("n must be non-negative")

            if method == "iterative":
                sequence = []
                a, b = 0, 1
                for _i in range(n):
                    sequence.append(a)
                    a, b = b, a + b
            elif method == "recursive":

                def fib_recursive(x):
                    if x <= 1:
                        return x
                    return fib_recursive(x - 1) + fib_recursive(x - 2)

                sequence = [fib_recursive(i) for i in range(n)]
            else:
                raise ValueError("Method must be 'iterative' or 'recursive'")

            duration = time.time() - start_time

            return {
                "sequence": sequence,
                "length": n,
                "method": method,
                "duration": duration,
                "last_value": sequence[-1] if sequence else 0,
                "timestamp": time.time(),
                "module": self.name,
            }

        except Exception as e:
            return {
                "error": str(e),
                "n": n,
                "method": method,
                "duration": time.time() - start_time,
                "timestamp": time.time(),
                "module": self.name,
            }

    def prime_check(self, numbers: int | list[int]) -> dict[str, Any]:
        """
        Check if numbers are prime.

        Args:
            numbers: Single number or list of numbers to check

        Returns:
            Prime check results
        """
        start_time = time.time()

        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True

        if isinstance(numbers, int):
            numbers = [numbers]

        results = []
        for num in numbers:
            results.append(
                {
                    "number": num,
                    "is_prime": is_prime(num),
                    "factors": (
                        []
                        if is_prime(num)
                        else [i for i in range(2, num) if num % i == 0][:10]
                    ),  # Limit factors
                }
            )

        duration = time.time() - start_time
        prime_count = sum(1 for r in results if r["is_prime"])

        return {
            "results": results,
            "total_numbers": len(numbers),
            "prime_count": prime_count,
            "composite_count": len(numbers) - prime_count,
            "duration": duration,
            "timestamp": time.time(),
            "module": self.name,
        }

    def data_transform(self, data: Any, operation: str = "json") -> dict[str, Any]:
        """
        Transform data between different formats.

        Args:
            data: Input data to transform
            operation: Transformation operation

        Returns:
            Transformed data with metadata
        """
        start_time = time.time()

        try:
            if operation == "json":
                if isinstance(data, str):
                    result = json.loads(data)
                else:
                    result = json.dumps(data, default=str)
            elif operation == "hash":
                import hashlib

                data_str = json.dumps(data, sort_keys=True, default=str)
                result = hashlib.sha256(data_str.encode()).hexdigest()
            elif operation == "reverse":
                if isinstance(data, list | tuple):
                    result = list(reversed(data))
                elif isinstance(data, str):
                    result = data[::-1]
                else:
                    result = str(data)[::-1]
            elif operation == "sort":
                if isinstance(data, list | tuple):
                    result = sorted(data)
                else:
                    result = sorted(str(data))
            else:
                raise ValueError(f"Unknown operation: {operation}")

            duration = time.time() - start_time

            return {
                "input": data,
                "operation": operation,
                "result": result,
                "input_type": type(data).__name__,
                "result_type": type(result).__name__,
                "duration": duration,
                "timestamp": time.time(),
                "module": self.name,
            }

        except Exception as e:
            return {
                "input": data,
                "operation": operation,
                "error": str(e),
                "duration": time.time() - start_time,
                "timestamp": time.time(),
                "module": self.name,
            }

    def get_metadata(self) -> dict[str, Any]:
        """
        Get module metadata for registry registration.

        Returns:
            Module metadata dictionary
        """
        return {
            **self.metadata,
            "runtime_info": self.info(),
            "last_accessed": datetime.now().isoformat(),
        }

    async def register_in_registry(
        self, registry_url: str | None = None
    ) -> dict[str, Any]:
        """
        Register this module in the Module Registry.

        Args:
            registry_url: Registry backend URL (uses instance default if None)

        Returns:
            Registration result
        """
        registry_url = registry_url or self.registry_url

        try:
            # This would integrate with the commune-ipfs integration client
            # For now, return a mock registration result
            metadata = self.get_metadata()

            # In a real implementation, this would call the integration client:
            # from integration_client import ModuleRegistryClient, ModuleMetadata
            # async with ModuleRegistryClient(registry_url) as client:
            #     result = await client.register_module_metadata(ModuleMetadata(**metadata))

            # Mock registration result
            mock_cid = f"Qm{hash(json.dumps(metadata, sort_keys=True))}"[:46]

            return {
                "status": "registered",
                "cid": mock_cid,
                "registry_url": registry_url,
                "metadata": metadata,
                "timestamp": time.time(),
                "module": self.name,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "registry_url": registry_url,
                "timestamp": time.time(),
                "module": self.name,
            }

    def health_check(self) -> dict[str, Any]:
        """
        Perform module health check.

        Returns:
            Health status information
        """
        uptime = time.time() - self.start_time

        # Perform basic functionality tests
        tests = {
            "compute_test": False,
            "fibonacci_test": False,
            "prime_test": False,
            "data_transform_test": False,
        }

        try:
            # Test compute function
            result = self.compute("add", 1, 2, 3)
            tests["compute_test"] = result.get("result") == 6

            # Test fibonacci function
            result = self.fibonacci(5)
            tests["fibonacci_test"] = result.get("sequence") == [0, 1, 1, 2, 3]

            # Test prime check
            result = self.prime_check([2, 3, 4])
            tests["prime_test"] = len(result.get("results", [])) == 3

            # Test data transform
            result = self.data_transform([3, 1, 2], "sort")
            tests["data_transform_test"] = result.get("result") == [1, 2, 3]

        except Exception as e:
            c.print(f"Health check error: {e}", color="red")

        all_tests_passed = all(tests.values())

        return {
            "status": "healthy" if all_tests_passed else "degraded",
            "uptime": uptime,
            "call_count": self.call_count,
            "tests": tests,
            "all_tests_passed": all_tests_passed,
            "memory_usage": "N/A",  # Could add actual memory monitoring
            "timestamp": time.time(),
            "module": self.name,
        }

    def test(self) -> dict[str, Any]:
        """
        Run comprehensive module tests (commune pattern).

        Returns:
            Test results
        """
        c.print(f"🧪 Running tests for {self.name}", color="blue")

        test_results: dict[str, Any] = {
            "module": self.name,
            "timestamp": time.time(),
            "tests": {},
        }

        # Test all exposed functions
        for fn_name in self.expose:
            if fn_name in ["info", "forward", "test", "register_in_registry"]:
                continue  # Skip meta functions

            try:
                tests_dict = test_results["tests"]
                if fn_name == "compute":
                    result = self.compute("add", 5, 10)
                    tests_dict[fn_name] = {
                        "passed": result.get("result") == 15,
                        "result": result,
                    }
                elif fn_name == "fibonacci":
                    result = self.fibonacci(6)
                    tests_dict[fn_name] = {
                        "passed": len(result.get("sequence", [])) == 6,
                        "result": result,
                    }
                elif fn_name == "prime_check":
                    result = self.prime_check([7, 8, 9])
                    tests_dict[fn_name] = {
                        "passed": len(result.get("results", [])) == 3,
                        "result": result,
                    }
                elif fn_name == "data_transform":
                    result = self.data_transform("hello", "reverse")
                    tests_dict[fn_name] = {
                        "passed": result.get("result") == "olleh",
                        "result": result,
                    }
                elif fn_name == "get_metadata":
                    result = self.get_metadata()
                    tests_dict[fn_name] = {
                        "passed": "name" in result and "version" in result,
                        "result": result,
                    }
                elif fn_name == "health_check":
                    result = self.health_check()
                    tests_dict[fn_name] = {
                        "passed": result.get("all_tests_passed", False),
                        "result": result,
                    }

            except Exception as e:
                tests_dict = test_results["tests"]
                tests_dict[fn_name] = {"passed": False, "error": str(e)}

        # Overall test result
        tests_dict = test_results["tests"]
        passed_tests = sum(
            1 for test in tests_dict.values() if test.get("passed", False)
        )
        total_tests = len(tests_dict)
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_passed": passed_tests == total_tests,
        }

        summary_dict = test_results["summary"]
        status_color = "green" if summary_dict["overall_passed"] else "red"
        c.print(
            f"✅ Tests completed: {passed_tests}/{total_tests} passed",
            color=status_color,
        )

        return test_results


# For commune compatibility
def main():
    """Main entry point for commune module system."""
    module = TestModule()
    return module


# For standalone execution
if __name__ == "__main__":
    # Standalone test execution
    print("🚀 TestModule Standalone Execution")
    print("=" * 50)

    module = TestModule()

    # Run basic tests
    print("\n📋 Module Info:")
    info = module.info()
    print(json.dumps(info, indent=2, default=str))

    print("\n🧮 Testing Compute Functions:")
    print("Addition:", module.compute("add", 10, 20, 30))
    print("Fibonacci:", module.fibonacci(8))
    print("Prime Check:", module.prime_check([17, 18, 19]))

    print("\n🏥 Health Check:")
    health = module.health_check()
    print(json.dumps(health, indent=2, default=str))

    print("\n🧪 Running Full Test Suite:")
    test_results = module.test()
    print(
        f"Overall Result: {'✅ PASSED' if test_results['summary']['overall_passed'] else '❌ FAILED'}"
    )
    print(f"Success Rate: {test_results['summary']['success_rate']:.1%}")
