# Module System

The Module System is the core building block of Commune. It provides a flexible, extensible framework for creating and managing distributed applications.

## Core Concepts

### What is a Module?

A module in Commune is a self-contained unit that:
- Encapsulates functionality and state
- Can be served as a network endpoint
- Supports remote execution
- Can be composed with other modules
- Maintains its own configuration

## Module Structure

### Basic Module Anatomy

```python
import commune as c
from typing import Any, Dict

class MyModule(c.Module):
    """Example module demonstrating core features."""
    
    # Default function called when no method is specified
    default_fn = 'forward'
    
    # Module description
    description = "My custom module"
    
    def __init__(self, config_param: int = 1):
        """Initialize module with configuration."""
        super().__init__()
        self.config_param = config_param
    
    def forward(self, x: Any) -> Any:
        """Default processing method."""
        return x * self.config_param

# Example usage
module = MyModule(config_param=2)
result = module.forward(3)  # Returns 6
```

### Core Features

1. **Auto-Configuration**
```python
import commune as c
from typing import Dict, Optional

class ConfigModule(c.Module):
    def __init__(
        self,
        param1: str = "default",
        param2: int = 42,
        config: Optional[Dict] = None
    ):
        super().__init__()
        # Save all init parameters as config
        self.set_config(locals())
        
        # Override with provided config
        if config:
            self.update_config(config)
    
    def get_param(self, name: str) -> Any:
        """Get parameter value from config."""
        return self.config.get(name)

# Example usage
module = ConfigModule(param1="custom")
print(module.get_param("param1"))  # "custom"
```

2. **State Management**
```python
import commune as c
from typing import Dict, List, Any

class StateModule(c.Module):
    def __init__(self):
        super().__init__()
        self.state = {
            'counter': 0,
            'history': []
        }
    
    def update_state(self, data: Any) -> Dict:
        """Update module state."""
        self.state['counter'] += 1
        self.state['history'].append(data)
        return self.state
    
    def get_state(self) -> Dict:
        """Return current state."""
        return self.state
    
    def reset_state(self) -> None:
        """Reset state to initial values."""
        self.state = {
            'counter': 0,
            'history': []
        }

# Example usage
module = StateModule()
module.update_state("test")
print(module.get_state())
```

3. **Module Communication**
```python
import commune as c
from typing import List

class CompositeModule(c.Module):
    def __init__(self):
        super().__init__()
        # Load other modules
        self.storage = c.module('storage')
        self.processor = c.module('processor')
    
    async def process_and_store(
        self,
        data: List[Any]
    ) -> Dict[str, Any]:
        """Process data and store results."""
        # Process data
        results = await self.processor.batch_process(data)
        
        # Store results
        storage_key = await self.storage.store(results)
        
        return {
            'key': storage_key,
            'results': results
        }

# Example usage
async def main():
    module = CompositeModule()
    result = await module.process_and_store([1, 2, 3])
    print(result)
```

## Error Handling

1. **Module Not Found**
```python
import commune as c
from typing import Optional

def safe_load_module(
    name: str,
    default: Optional[str] = None
) -> Optional[c.Module]:
    """Safely load a module with fallback."""
    try:
        return c.module(name)
    except ModuleNotFoundError:
        if default:
            return c.module(default)
        return None

# Example usage
module = safe_load_module('custom_module', 'default_module')
```

2. **Network Errors**
```python
import commune as c
from typing import Any, Dict

class ResilientModule(c.Module):
    def __init__(self, retries: int = 3):
        super().__init__()
        self.retries = retries
    
    async def safe_call(
        self,
        module_name: str,
        method: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Make resilient module calls."""
        for attempt in range(self.retries):
            try:
                module = c.module(module_name)
                result = await getattr(module, method)(*args, **kwargs)
                return {
                    'success': True,
                    'result': result,
                    'attempts': attempt + 1
                }
            except Exception as e:
                if attempt == self.retries - 1:
                    return {
                        'success': False,
                        'error': str(e),
                        'attempts': attempt + 1
                    }
                continue

# Example usage
async def main():
    module = ResilientModule()
    result = await module.safe_call('api_module', 'fetch_data')
    print(result)
```

## Module Lifecycle

### 1. Creation

```python
# Direct creation
module = c.module('my_module')

# With configuration
module = c.module('my_module', kwargs={'param1': 'value'})

# From path
module = c.module('path/to/module.py')
```

### 2. Serving

```python
# Basic serving
c.serve('my_module')

# With specific port
c.serve('my_module', port=8000)

# As validator
c.serve('my_module', validator=True)
```

### 3. Communication

```python
# Direct call
result = module.forward("input")

# Remote call
result = c.call('my_module/forward', "input")

# Async call
result = await c.acall('my_module/forward', "input")
```

## Module Composition

### 1. Basic Composition

```python
class ComposedModule(c.Module):
    def __init__(self):
        super().__init__()
        self.submodule1 = c.module('module1')
        self.submodule2 = c.module('module2')
    
    def process(self, data):
        result1 = self.submodule1.process(data)
        return self.submodule2.process(result1)
```

### 2. Module Inheritance

```python
class BaseModule(c.Module):
    def base_method(self):
        return "base functionality"

class ExtendedModule(BaseModule):
    def extended_method(self):
        base_result = self.base_method()
        return f"extended {base_result}"
```

## Advanced Features

### 1. Module Caching

```python
# Enable caching
module = c.module('heavy_module', cache=True)

# Clear cache
c.clear_cache()
```

### 2. Module Resolution

```python
# Resolve module path
full_path = c.resolve_module('shortcut_name')

# Get module code
code = module.code()

# Get code hash
hash = module.code_hash()
```

### 3. Custom Endpoints

```python
class APIModule(c.Module):
    # Define custom endpoints
    endpoints = ['search', 'process', 'forward']
    
    def search(self, query):
        return f"Searching for {query}"
    
    def process(self, data):
        return f"Processing {data}"
```

## Best Practices

1. **Module Design**
   - Keep modules focused and single-purpose
   - Use descriptive names for methods
   - Document module functionality
   - Handle errors gracefully

2. **State Management**
   - Use configuration for mutable state
   - Implement proper cleanup methods
   - Cache expensive computations

3. **Security**
   - Validate input parameters
   - Implement access controls
   - Secure sensitive data

## Common Patterns

### 1. Factory Pattern

```python
class ModuleFactory(c.Module):
    def create_module(self, module_type):
        return c.module(module_type)
```

### 2. Observer Pattern

```python
class ObserverModule(c.Module):
    def __init__(self):
        super().__init__()
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.append(observer)
    
    def notify(self, event):
        for observer in self.observers:
            observer.update(event)
```

### 3. Singleton Pattern

```python
class SingletonModule(c.Module):
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

## Troubleshooting

### Common Issues

1. **Module Not Found**
   ```python
   # Check module path
   print(c.resolve_module('missing_module'))
   
   # List available modules
   print(c.modules())
   ```

2. **Configuration Errors**
   ```python
   # Check current config
   print(module.config())
   
   # Reset config
   module.reset_config()
   ```

3. **Connection Issues**
   ```python
   # Check module status
   c.status('my_module')
   
   # Restart module
   c.restart('my_module')
   ```

## Next Steps

1. Explore [Network Architecture](11-Network-Architecture.md)
2. Learn about [Blockchain Integration](12-Blockchain-Integration.md)
3. Try building [Custom Modules](20-First-Module.md)
