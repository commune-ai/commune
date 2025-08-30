 # start of file
# Commune Storage Module

A robust, flexible storage interface for the Commune framework that provides seamless access to various storage backends.

## Features

- Multiple backend support (Local, S3, IPFS, etc.)
- Consistent API across different storage providers
- Automatic serialization/deserialization of data
- Configurable caching mechanisms
- Thread-safe operations

## Usage

```python
import commune as c

# Initialize storage with default settings
storage = c.module('storage')

# Store data
storage.put('my_key', {'data': 'value'})

# Retrieve data
data = storage.get('my_key')

# Use specific backend
s3_storage = c.module('storage', backend='s3', bucket='my-bucket')
s3_storage.put('remote_file.txt', 'content')

# List keys
keys = storage.ls()
```

## Configuration

Configure the storage module through the config file or by passing parameters during initialization:

```python
storage_config = {
    'backend': 'local',  # Options: local, s3, ipfs, etc.
    'path': './data',    # Base path for local storage
    'cache_size': 100,   # Number of items to cache
    'compression': True  # Enable/disable compression
}

storage = c.module('storage', **storage_config)
```

## Extending

Create custom storage backends by implementing the StorageBackend interface:

```python
from commune.modules.storage import StorageBackend

class MyCustomStorage(StorageBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
        
    def get(self, key, **kwargs):
        # Implementation
        
    def put(self, key, value, **kwargs):
        # Implementation
        
    # Implement other required methods
```

## Contributing

Contributions are welcome! Please check the contribution guidelines before submitting PRs.
