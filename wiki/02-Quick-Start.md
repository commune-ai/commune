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
        """Process input text with greeting."""
        return f"{self.message} You said: {text}"

# Initialize the module
module = MyFirstModule()
```

### 2. Running Your Module

```python
# Method 1: Direct usage
module = MyFirstModule()
result = module.forward("Hello World")
print(result)  # "Hello from Commune! You said: Hello World"

# Method 2: Using module system
c.serve('MyFirstModule', port=8000)  # In one terminal
result = c.call('MyFirstModule.forward', "Hello World")  # In another terminal
```

### 3. Web Integration Example

```python
import commune as c
from typing import List, Dict

class WebModule(c.Module):
    def __init__(self):
        super().__init__()
        self.web = c.module('web')
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """Search the web for given query."""
        results = self.web.search(query)
        return [
            {
                'title': result.title,
                'url': result.url,
                'snippet': result.snippet
            }
            for result in results[:5]  # Top 5 results
        ]

    def summarize(self, url: str) -> str:
        """Summarize content from URL."""
        content = self.web.get_content(url)
        return self.web.summarize(content)

# Example usage
def main():
    module = WebModule()
    results = module.search("Commune blockchain")
    
    if results:
        summary = module.summarize(results[0]['url'])
        print(f"Summary of top result: {summary}")

if __name__ == '__main__':
    main()
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

## Advanced Features

### 1. Module Configuration

```python
import commune as c
from typing import Optional

class ConfigurableModule(c.Module):
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        super().__init__()
        self.config = {
            'api_key': api_key or c.get_api_key('default'),
            'max_retries': max_retries,
            'timeout': timeout
        }
    
    def get_config(self) -> dict:
        """Return non-sensitive config."""
        return {
            'max_retries': self.config['max_retries'],
            'timeout': self.config['timeout']
        }

# Example usage
module = ConfigurableModule(max_retries=5)
print(module.get_config())
```

### 2. Error Handling

```python
import commune as c
from typing import Any

class RobustModule(c.Module):
    def __init__(self):
        super().__init__()
    
    def safe_process(self, data: Any) -> dict:
        """Process data with error handling."""
        try:
            result = self.process(data)
            return {
                'success': True,
                'result': result,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def process(self, data: Any) -> Any:
        """Internal processing logic."""
        if not data:
            raise ValueError("Data cannot be empty")
        return {'processed': data}

# Example usage
module = RobustModule()
print(module.safe_process(None))  # Handles error gracefully
print(module.safe_process("test"))  # Processes successfully
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
