# Commune API Module

## Overview

The Commune API module provides a FastAPI-based interface for managing and interacting with modules in the Commune framework. It enables module registration, discovery, and management through a RESTful API interface.

## Features

- **Module Management**: Add, remove, update, and retrieve module information
- **Module Discovery**: List available modules with filtering and pagination support
- **Background Processing**: Run background tasks to keep module information up-to-date
- **Persistent Storage**: Store module information in a local file system

## Installation

The API module is part of the Commune framework. Ensure you have Commune installed:

```bash
pip install commune
```

## Usage

### Initializing the API

```python
import commune as c

# Initialize the API module
api = c.mod('api')()

# Start the API with background processing
api = c.mod('api')(background=True)
```

### Module Operations

```python
# List all modules
modules = api.modules()

# Add a new module
api.add_module(
    name="my_module",
    key="module_key",
    url="http://localhost:8000",
    code="import commune as c\n\nclass MyModule:\n    def forward(self, x):\n        return x"
)

# Get information about a specific module
module_info = api.get_module("my_module")

# Remove a module
api.remove("my_module")
```

### Background Processing

The API can run in the background to periodically update module information:

```python
# Start the background process
c.serve('api:background')

# Kill the background process when done
c.kill('api:background')
```

## API Endpoints

The API module exposes the following endpoints:

- `modules`: List all modules with filtering and pagination
- `add_module`: Register a new module
- `remove`: Remove a module
- `update`: Update module information
- `test`: Run tests on the API functionality
- `get_module`: Get information about a specific module
- `info`: Get detailed information about a module
- `functions`: List available functions

## Configuration

The API module stores data in the following locations:

- Module information: `~/.commune/api/modules/`
- General API data: `~/.commune/api/`

## Contributing

Contributions to the API module are welcome. Please ensure that your code follows the existing patterns and includes appropriate tests.

## License

The Commune framework, including the API module, is open-source software licensed under the MIT license.
