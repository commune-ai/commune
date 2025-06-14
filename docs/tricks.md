# Commune Tips & Tricks

## Navigation Shortcuts

### Quick Module Navigation
- `c go {module}` - Takes you to any module instantly
  - Example: `c go model.openai` jumps to the OpenAI module
  - Works with nested modules: `c go dev.tool.cmd`

### Listing and Discovery
- `c ls` - Lists all available modules in your current context
- `c ls {path}` - Lists modules in a specific path
- `c find {pattern}` - Search for modules matching a pattern

## Module Development Tricks

### Quick Module Creation
```bash
# Create a module with a template
c new mymodule --template api

# Create and immediately edit
c new mymodule && c edit mymodule
```

### Testing Shortcuts
```bash
# Test with live reload
c test mymodule --watch

# Test specific methods
c test mymodule.method_name
```

## Network Tricks

### Efficient Staking
```bash
# Stake on multiple modules at once
c stake module1,module2,module3 100

# Auto-restake rewards
c stake mymodule 100 --auto-restake
```

### Module Discovery
```bash
# Find high-performing modules
c top modules --by-stake

# Find modules by tag
c search --tag "ai" --tag "vision"
```

## Development Productivity

### Hot Reloading
```bash
# Serve with auto-reload on code changes
c serve mymodule --reload
```

### Batch Operations
```bash
# Call multiple modules in parallel
c call module1.method,module2.method,module3.method

# Pipe outputs
c call data.fetch | c call processor.clean | c call model.predict
```

## Debugging Tips

### Verbose Logging
```bash
# Enable debug mode for detailed logs
c call mymodule.method --debug

# Filter logs by level
c logs mymodule --level error
```

### Performance Profiling
```bash
# Profile module performance
c profile mymodule.method

# Monitor resource usage
c monitor mymodule
```

## Advanced Tricks

### Module Composition
```python
# In your module code, easily compose other modules
class MyModule(c.Module):
    def forward(self, x):
        # Call other modules seamlessly
        processed = c.call('processor.clean', x)
        result = c.call('model.predict', processed)
        return result
```

### Dynamic Module Loading
```python
# Load modules dynamically based on config
model_name = c.config.get('model', 'model.openai')
model = c.module(model_name)
result = model.forward(input_data)
```

### Caching Results
```bash
# Cache expensive computations
c call expensive.computation --cache --ttl 3600
```

### Environment Management
```bash
# Switch between environments
c env dev
c env prod

# Export/import module configurations
c export mymodule > config.json
c import mymodule < config.json
```

## Shortcuts for Common Tasks

### Quick Deployment
```bash
# One-liner to create, test, and serve
c new api && c test api && c serve api
```

### Module Templates
```bash
# Use templates for quick starts
c new myapi --template fastapi
c new mymodel --template transformer
c new mydata --template scraper
```

### Batch Module Management
```bash
# Update multiple modules
c update module1,module2,module3

# Check health of all your modules
c health --mine
```

## Pro Tips

1. **Use Tab Completion**: Most commands support tab completion for faster navigation
2. **Alias Common Commands**: Create shell aliases for frequently used commands
3. **Module Namespacing**: Organize your modules with clear namespaces (e.g., `myorg.service.api`)
4. **Version Control**: Always version your modules for easy rollbacks
5. **Documentation**: Use `c docs {module}` to quickly access module documentation

## Hidden Features

- **Interactive Mode**: Run `c interactive` for a REPL-like experience
- **Module Visualization**: `c viz {module}` shows module dependencies graphically
- **Benchmarking**: `c bench {module}` runs performance benchmarks
- **Auto-documentation**: `c autodoc {module}` generates documentation from code

Remember: The power of Commune lies in its composability. Don't hesitate to chain commands and modules together to create powerful workflows!