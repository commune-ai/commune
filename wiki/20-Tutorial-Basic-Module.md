# Tutorial: Creating Your First Commune Module

This tutorial will guide you through creating a basic Commune module, covering essential concepts and best practices.

## Prerequisites
- Commune installed and configured
- Basic Python knowledge
- Understanding of async/await (optional)

## Basic Module Structure

Let's create a simple calculator module:

```python
import commune as c

class Calculator(c.Module):
    def __init__(self):
        super().__init__()
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers and store in history."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a and store in history."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def get_history(self) -> list:
        """Return calculation history."""
        return self.history
```

## Step-by-Step Explanation

### 1. Module Class Definition
```python
class Calculator(c.Module):
```
- Inherit from `c.Module` to get Commune functionality
- This provides networking, serialization, and state management

### 2. Initialization
```python
def __init__(self):
    super().__init__()
    self.history = []
```
- Always call `super().__init__()` first
- Initialize any module-specific state

### 3. Module Methods
```python
def add(self, a: float, b: float) -> float:
```
- Use type hints for better documentation
- Keep methods focused and single-purpose
- Document with docstrings

## Making the Module Network-Accessible

```python
# Save as calculator.py
import commune as c

class Calculator(c.Module):
    # Previous code here...
    
    @property
    def functions(self):
        """List available module functions."""
        return ['add', 'subtract', 'get_history']
    
    def info(self):
        """Module information."""
        return {
            'name': 'calculator',
            'version': '0.1.0',
            'functions': self.functions
        }

if __name__ == '__main__':
    # Start the module
    module = Calculator()
    module.serve()
```

## Using the Module

### 1. Local Usage
```python
# Create instance
calc = c.module('calculator')

# Use methods
result = calc.add(5, 3)
print(result)  # 8

history = calc.get_history()
print(history)  # ['5 + 3 = 8']
```

### 2. Network Usage
```python
# Start the module server
c.serve('calculator')

# Connect from another process/machine
remote_calc = c.connect('calculator')
result = remote_calc.add(10, 5)
```

## Adding State Management

```python
class Calculator(c.Module):
    def __init__(self):
        super().__init__()
        self.load_state()
    
    def save_state(self):
        """Save history to persistent storage."""
        state = {'history': self.history}
        self.put('state', state)
    
    def load_state(self):
        """Load history from storage."""
        state = self.get('state', {})
        self.history = state.get('history', [])
```

## Error Handling

```python
class Calculator(c.Module):
    def divide(self, a: float, b: float) -> float:
        """Divide a by b with error handling."""
        try:
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
            self.history.append(f"{a} / {b} = {result}")
            return result
        except Exception as e:
            self.history.append(f"Error: {str(e)}")
            raise
```

## Testing the Module

```python
def test_calculator():
    calc = Calculator()
    
    # Test addition
    assert calc.add(2, 3) == 5
    
    # Test subtraction
    assert calc.subtract(5, 3) == 2
    
    # Test history
    assert len(calc.get_history()) == 2
    
    # Test error handling
    try:
        calc.divide(1, 0)
        assert False, "Should raise error"
    except ValueError:
        pass
```

## Best Practices

1. **Documentation**
   - Use docstrings for all methods
   - Include type hints
   - Document exceptions

2. **Error Handling**
   - Handle expected errors gracefully
   - Log unexpected errors
   - Provide meaningful error messages

3. **State Management**
   - Use `self.put()` and `self.get()` for persistence
   - Handle state loading failures
   - Implement state cleanup if needed

4. **Testing**
   - Write unit tests
   - Test edge cases
   - Test network functionality

## Next Steps

1. Try adding more operations to the calculator
2. Implement async methods for long-running calculations
3. Add input validation and error handling
4. Create a web interface using Commune's built-in server

## Common Issues

1. **Module Not Found**
   ```python
   # Ensure module is in Python path
   import sys
   sys.path.append('/path/to/modules')
   ```

2. **Network Connection Failed**
   ```python
   # Check if module is running
   c.check_module('calculator')
   
   # Restart module if needed
   c.restart_module('calculator')
   ```

3. **State Not Persisting**
   ```python
   # Explicitly save state after changes
   calc.save_state()
   ```

## Related Documentation

1. [Module System](10-Module-System.md)
2. [Network Architecture](11-Network-Architecture.md)
3. [Advanced Module Development](21-Tutorial-Advanced-Module.md)
