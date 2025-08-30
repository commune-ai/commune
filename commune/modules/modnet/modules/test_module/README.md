# Test Module for Mod-Net Module Registry

A demonstration module that showcases integration with the Module Registry system and IPFS metadata storage, following the commune module pattern.

## Overview

This test module provides:

- **Computational Functions**: Mathematical operations, Fibonacci sequences, prime checking
- **Data Transformation**: JSON processing, hashing, sorting, reversing
- **Module Registry Integration**: Metadata storage and retrieval via IPFS
- **Health Monitoring**: Built-in health checks and performance monitoring
- **Commune Compatibility**: Follows commune module patterns for network serving

## Features

### Core Functions

- `info()` - Get module information and status
- `forward(fn, *args, **kwargs)` - Forward function calls (commune pattern)
- `compute(operation, *args)` - Perform mathematical computations
- `fibonacci(n, method)` - Generate Fibonacci sequences
- `prime_check(numbers)` - Check if numbers are prime
- `data_transform(data, operation)` - Transform data between formats
- `get_metadata()` - Get module metadata for registry
- `register_in_registry()` - Register module in the Module Registry
- `health_check()` - Perform comprehensive health check
- `test()` - Run full test suite

### Computational Operations

- **Addition**: Sum multiple numbers
- **Multiplication**: Multiply multiple numbers
- **Power**: Calculate exponentials
- **Factorial**: Calculate factorials
- **Fibonacci**: Generate sequences (iterative/recursive)
- **Prime Checking**: Identify prime numbers with factorization

### Data Transformations

- **JSON**: Parse/stringify JSON data
- **Hashing**: SHA256 hashing of data
- **Reverse**: Reverse strings, lists, or tuples
- **Sort**: Sort collections

## Installation

### Prerequisites

- Python 3.8+
- Optional: commune framework for network serving

### Setup

```bash
# Navigate to the test module directory
cd modules/test_module

# Install dependencies (minimal - mostly built-in modules)
pip install -r requirements.txt

# Or if using uv:
uv pip install -r requirements.txt
```

## Usage

### Standalone Execution

```bash
# Run the module directly
python module.py

# Run integration tests
python test_integration.py
```

### Programmatic Usage

```python
from modules.test_module import TestModule

# Initialize module
module = TestModule(name="my-test-module")

# Get module info
info = module.info()
print(f"Module: {info['name']} v{info['version']}")

# Perform computations
result = module.compute("add", 10, 20, 30)
print(f"Sum: {result['result']}")  # 60

# Generate Fibonacci sequence
fib = module.fibonacci(10)
print(f"Fibonacci(10): {fib['sequence']}")

# Check prime numbers
primes = module.prime_check([17, 18, 19, 20, 21])
for r in primes['results']:
    if r['is_prime']:
        print(f"{r['number']} is prime")

# Transform data
transformed = module.data_transform([3, 1, 4, 1, 5], "sort")
print(f"Sorted: {transformed['result']}")  # [1, 1, 3, 4, 5]

# Run health check
health = module.health_check()
print(f"Health: {health['status']}")
```

### Commune Integration

```python
# For commune network serving
import commune as c

# Serve the module on the network
c.serve('modules.test_module', name='test-module', port=8080)

# Call the module remotely
result = c.call('test-module/compute', 'factorial', 5)
print(f"5! = {result['result']}")  # 120
```

### Module Registry Integration

```python
import asyncio
from modules.test_module import TestModule

async def register_module():
    module = TestModule(
        name="production-module",
        registry_url="http://localhost:8004"
    )

    # Register in Module Registry with IPFS metadata
    result = await module.register_in_registry()
    print(f"Registered with CID: {result['cid']}")

    return result

# Run registration
result = asyncio.run(register_module())
```

## Testing

### Run Integration Tests

```bash
# Comprehensive integration test suite
python test_integration.py
```

The integration tests cover:

- ✅ Basic functionality testing
- ✅ Forward mechanism (commune pattern)
- ✅ Health checks and metadata
- ✅ Module Registry integration (mock)
- ✅ Comprehensive test suite
- ✅ Performance benchmarks
- ✅ Real-world usage scenarios

### Built-in Test Suite

```python
# Run the module's built-in tests
module = TestModule()
test_results = module.test()

if test_results['summary']['overall_passed']:
    print("✅ All tests passed!")
else:
    print(f"❌ {test_results['summary']['failed_tests']} tests failed")
```

## Module Registry Integration

This module is designed to work with the Mod-Net Module Registry system:

### Metadata Structure

```json
{
  "name": "test-module",
  "version": "1.0.0",
  "description": "A test module demonstrating Module Registry integration",
  "author": "mod-net-developer@example.com",
  "license": "MIT",
  "repository": "https://github.com/Bakobiibizo/mod-net-modules",
  "dependencies": ["commune", "asyncio", "json"],
  "tags": ["test", "computation", "fibonacci", "prime", "demo"],
  "chain_type": "ed25519",
  "public_key": "0x...",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:00:00"
}
```

### IPFS Integration

The module integrates with the commune-ipfs backend for:

- **Metadata Storage**: Module metadata stored on IPFS
- **Content Addressing**: CID-based retrieval
- **Decentralized Discovery**: Network-wide module discovery
- **Version Management**: Immutable version history

### Registration Workflow

1. **Prepare Metadata**: Module generates comprehensive metadata
2. **IPFS Storage**: Metadata uploaded to IPFS via commune-ipfs backend
3. **CID Registration**: CID registered in Substrate pallet
4. **Network Discovery**: Module becomes discoverable on network

## Performance

### Benchmarks

Typical performance on modern hardware:

- **Fibonacci(30)**: ~0.001s (iterative)
- **Prime Check (100 numbers)**: ~0.01s
- **Factorial(10)**: ~0.0001s
- **Data Transform (1000 items)**: ~0.001s

### Optimization

- Uses iterative algorithms for better performance
- Minimal memory footprint
- Efficient data structures
- Built-in performance monitoring

## Architecture

### Module Structure

```
test_module/
├── __init__.py          # Package initialization
├── module.py            # Main module implementation
├── test_integration.py  # Integration tests
├── requirements.txt     # Dependencies
├── README.md           # This documentation
└── pyproject.toml      # Project configuration
```

### Class Hierarchy

```
TestModule
├── Core Functions
│   ├── info()
│   ├── forward()
│   └── test()
├── Computational Functions
│   ├── compute()
│   ├── fibonacci()
│   └── prime_check()
├── Data Functions
│   └── data_transform()
├── Registry Functions
│   ├── get_metadata()
│   └── register_in_registry()
└── Monitoring Functions
    └── health_check()
```

## Configuration

### Environment Variables

- `REGISTRY_URL`: Module Registry backend URL (default: http://localhost:8004)
- `MODULE_NAME`: Override module name
- `PUBLIC_KEY`: Module public key for registry

### Initialization Parameters

```python
module = TestModule(
    name="custom-module",           # Module name
    public_key="0x123...",          # Public key for registry
    registry_url="http://...",      # Registry backend URL
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure Python path includes the modules directory
   export PYTHONPATH="/path/to/mod-net/modules:$PYTHONPATH"
   ```

2. **Registry Connection**
   ```bash
   # Ensure commune-ipfs backend is running
   cd commune-ipfs
   uv run python main.py --port 8004
   ```

3. **IPFS Daemon**
   ```bash
   # Start IPFS daemon
   ipfs daemon
   ```

### Debug Mode

```python
# Enable verbose logging
module = TestModule(name="debug-module")

# Run with detailed output
result = module.test()
print(json.dumps(result, indent=2))
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/Bakobiibizo/mod-net-modules
cd mod-net-modules/modules/test_module

# Install development dependencies
uv pip install -r requirements.txt

# Run tests
python test_integration.py
```

## License

MIT License - see the main repository for details.

## Support

For issues and questions:

- GitHub Issues: https://github.com/Bakobiibizo/mod-net-modules/issues
- Documentation: See main repository README
- Integration Guide: See commune-ipfs documentation

---

**Note**: This is a demonstration module for the Mod-Net Module Registry system. It showcases the integration patterns and best practices for creating commune-compatible modules with IPFS metadata storage.
