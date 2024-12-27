# Quick Start Guide

This guide will help you get started with Commune by walking through basic concepts and common use cases.

## Basic Usage

### 1. Creating Your First Module

```python
import commune as c

# Create a simple module
class MyFirstModule(c.Module):
    def __init__(self):
        super().__init__()
        self.message = "Hello from Commune!"
    
    def forward(self, text: str) -> str:
        return f"{self.message} You said: {text}"

# Save this in a file named my_first_module.py
```

### 2. Running Your Module

```bash
# Start the module server
c serve MyFirstModule

# In another terminal, call the module
c call MyFirstModule/forward "Hello World"
```

### 3. Web Integration Example

```python
import commune as c

# Create a web-enabled module
class WebModule(c.Module):
    def __init__(self):
        super().__init__()
        self.web = c.module('web')
    
    def search(self, query: str) -> str:
        return self.web.search(query)

# Use the module
module = WebModule()
results = module.search("Commune blockchain")
```

## Common Patterns

### 1. Module Management

```python
# List available modules
c modules

# Get module info
c module_info MyFirstModule

# Stop a module
c stop MyFirstModule
```

### 2. Remote Execution

```python
# Deploy module to network
c serve MyFirstModule --network mainnet

# Call remote module
c call remote:MyFirstModule/forward "Hello from remote!"
```

### 3. Module Composition

```python
import commune as c

class ComposedModule(c.Module):
    def __init__(self):
        super().__init__()
        # Load other modules
        self.web = c.module('web')
        self.storage = c.module('storage')
    
    def process(self, query: str) -> dict:
        # Use multiple modules together
        search_result = self.web.search(query)
        self.storage.save(query, search_result)
        return {
            'query': query,
            'result': search_result
        }
```

## Working with Blockchain

### 1. Basic Blockchain Integration

```python
import commune as c

# Connect to blockchain
chain = c.module('subtensor')

# Get network info
info = chain.get_network_info()
print(info)
```

### 2. Custom Validator

```python
class MyValidator(c.Module):
    def validate(self, data: dict) -> bool:
        # Implement validation logic
        return True

# Register validator
c serve MyValidator --network mainnet --validator
```

## Next Steps

1. Explore [Core Concepts](10-Module-System.md) for deeper understanding
2. Learn about [Network Architecture](11-Network-Architecture.md)
3. Try [Advanced Tutorials](20-First-Module.md)

## Tips and Best Practices

1. **Module Design**
   - Keep modules focused and single-purpose
   - Use descriptive names for methods
   - Implement proper error handling

2. **Performance**
   - Use async methods for I/O operations
   - Implement caching where appropriate
   - Monitor resource usage

3. **Security**
   - Never hardcode sensitive data
   - Use environment variables for configuration
   - Implement proper access controls

## Common Issues

1. **Connection Issues**
```bash
# Check network status
c network status

# Reset connection
c network reset
```

2. **Module Errors**
```bash
# Enable debug logging
c set_log_level debug

# Check module logs
c logs MyFirstModule
```

## Getting Help

- Join our [Discord](https://discord.gg/commune)
- Check [GitHub Issues](https://github.com/commune-ai/commune/issues)
- Read the [API Documentation](30-CLI-Reference.md)
