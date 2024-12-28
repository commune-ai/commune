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
from typing import List, Dict, Any, Union
from dataclasses import dataclass

@dataclass
class CalculatorConfig:
    """Configuration for Calculator module."""
    precision: int = 2
    max_history: int = 100

class Calculator(c.Module):
    """A simple calculator module with history tracking."""
    
    def __init__(self, config: CalculatorConfig = None):
        """Initialize calculator with optional configuration."""
        super().__init__()
        self.config = config or CalculatorConfig()
        self.history: List[str] = []
    
    def add(self, a: Union[int, float], b: Union[int, float]) -> float:
        """Add two numbers and store in history.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b rounded to configured precision
        """
        result = round(a + b, self.config.precision)
        self._add_to_history(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: Union[int, float], b: Union[int, float]) -> float:
        """Subtract b from a and store in history.
        
        Args:
            a: Number to subtract from
            b: Number to subtract
            
        Returns:
            Difference of a and b rounded to configured precision
        """
        result = round(a - b, self.config.precision)
        self._add_to_history(f"{a} - {b} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """Return calculation history.
        
        Returns:
            List of calculation history entries
        """
        return self.history
    
    def _add_to_history(self, entry: str) -> None:
        """Add entry to history, maintaining max size.
        
        Args:
            entry: History entry to add
        """
        self.history.append(entry)
        if len(self.history) > self.config.max_history:
            self.history.pop(0)

# Example usage
def main():
    # Create calculator with custom config
    calc = Calculator(CalculatorConfig(precision=3, max_history=5))
    
    # Perform calculations
    result1 = calc.add(5.123, 3.456)
    result2 = calc.subtract(10.1, 3.2)
    
    # View history
    print(f"Results: {result1}, {result2}")
    print(f"History: {calc.get_history()}")

if __name__ == '__main__':
    main()
```

## Making the Module Network-Accessible

```python
import commune as c
from typing import List, Dict, Any
import asyncio

class NetworkCalculator(Calculator):
    """Network-enabled calculator module."""
    
    async def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000
    ) -> None:
        """Serve calculator over network.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
        """
        server = await c.serve(
            self,
            host=host,
            port=port
        )
        print(f"Calculator serving on {host}:{port}")
        return server
    
    @property
    def functions(self) -> List[str]:
        """List available module functions.
        
        Returns:
            List of function names
        """
        return ['add', 'subtract', 'get_history']
    
    def info(self) -> Dict[str, Any]:
        """Get module information.
        
        Returns:
            Dictionary containing module metadata
        """
        return {
            'name': 'calculator',
            'version': '0.1.0',
            'functions': self.functions,
            'config': {
                'precision': self.config.precision,
                'max_history': self.config.max_history
            }
        }

async def main():
    """Start network calculator service."""
    try:
        calc = NetworkCalculator()
        await calc.serve()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down calculator service...")

if __name__ == '__main__':
    asyncio.run(main())
```

## Using the Module

### 1. Local Usage
```python
import commune as c
from typing import Dict, Any

async def use_calculator() -> None:
    """Example of using calculator module locally."""
    try:
        # Create instance
        calc = await c.connect('calculator')
        
        # Use methods
        result = await calc.add(5, 3)
        print(f"5 + 3 = {result}")
        
        history = await calc.get_history()
        print(f"Calculation history: {history}")
    except Exception as e:
        print(f"Error using calculator: {e}")

# Run example
asyncio.run(use_calculator())
```

### 2. Remote Usage
```python
import commune as c
from typing import Optional

async def connect_calculator(
    address: str = "localhost:8000"
) -> Optional[c.Module]:
    """Connect to remote calculator.
    
    Args:
        address: Remote calculator address
        
    Returns:
        Connected calculator module or None if connection fails
    """
    try:
        calc = await c.connect(
            module='calculator',
            address=address
        )
        print(f"Connected to calculator at {address}")
        return calc
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None

async def main():
    calc = await connect_calculator()
    if calc:
        result = await calc.add(10, 5)
        print(f"10 + 5 = {result}")

if __name__ == '__main__':
    asyncio.run(main())
```

## Error Handling

```python
import commune as c
from typing import Dict, Any, Optional
import asyncio

class RobustCalculator(Calculator):
    """Calculator with enhanced error handling."""
    
    async def safe_calculate(
        self,
        operation: str,
        a: Union[int, float],
        b: Union[int, float]
    ) -> Dict[str, Any]:
        """Safely perform calculation with error handling.
        
        Args:
            operation: Operation to perform ('add' or 'subtract')
            a: First number
            b: Second number
            
        Returns:
            Dictionary containing result or error information
        """
        try:
            method = getattr(self, operation)
            result = await method(a, b)
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
    
    @staticmethod
    def validate_input(
        value: Any
    ) -> Optional[str]:
        """Validate numeric input.
        
        Args:
            value: Value to validate
            
        Returns:
            Error message if validation fails, None otherwise
        """
        try:
            float(value)
            return None
        except (TypeError, ValueError):
            return f"Invalid numeric value: {value}"

# Example usage
async def main():
    calc = RobustCalculator()
    
    # Test with valid input
    result1 = await calc.safe_calculate('add', 5, 3)
    print(f"Valid calculation: {result1}")
    
    # Test with invalid input
    result2 = await calc.safe_calculate('add', 'invalid', 3)
    print(f"Invalid calculation: {result2}")

if __name__ == '__main__':
    asyncio.run(main())
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
