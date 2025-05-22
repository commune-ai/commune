# Commune Store Module

## Overview

The Store module provides a simple, file-based storage system for JSON data with support for encryption, path management, and file operations. It's designed to be a lightweight persistence layer for Commune applications.

## Features

- **JSON Data Storage**: Store and retrieve JSON-serializable data
- **Path Management**: Automatically handles file paths and extensions
- **Encryption Support**: Secure your data with encryption/decryption capabilities
- **File Operations**: Create, read, update, delete, and list stored items
- **Directory Management**: Create and remove directories as needed
- **Content Hashing**: Generate content identifiers (CIDs) for files and directories

## Installation

The Store module is part of the Commune framework. If you have Commune installed, you already have access to the Store module.

## Basic Usage

```python
import commune as c

# Create a store instance with a custom folder path
store = c.Store(folder_path='~/.commune/mydata', suffix='json')

# Store data
store.put('user/profile', {'name': 'John', 'age': 30})

# Retrieve data
profile = store.get('user/profile')
print(profile)  # {'name': 'John', 'age': 30}

# Check if a path exists
exists = store.exists('user/profile')
print(exists)  # True

# Remove data
store.rm('user/profile')
```

## Advanced Features

### Encryption

```python
import commune as c

store = c.Store(folder_path='~/.commune/secure')
key = c.key()  # Generate a key or provide your own

# Store and encrypt data
store.put('secret', {'api_key': '12345'})
store.encrypt('secret', key)

# Check if data is encrypted
print(store.is_encrypted('secret'))  # True

# Decrypt data
decrypted = store.decrypt('secret', key)
print(decrypted)  # {'api_key': '12345'}
```

### Content Identification

```python
import commune as c

store = c.Store()

# Generate a content identifier for a file or directory
cid = store.cid('/path/to/file')
print(cid)  # Hash of the file content
```

### Listing and Searching

```python
import commune as c

store = c.Store(folder_path='~/.commune/data')

# List all paths in the store
paths = store.paths()

# Search for specific paths
paths = store.paths(search='user')

# Get all items as data
items = store.items()

# Get items as a pandas DataFrame
df = store.items(df=True)
```

## API Reference

### Constructor

- `Store(folder_path='~/.commune/test', suffix='json')`: Create a new Store instance

### Core Methods

- `put(path, data)`: Store data at the specified path
- `get(path, default=None, max_age=None, update=False)`: Retrieve data from the specified path
- `rm(path)`: Remove a file at the specified path
- `exists(path)`: Check if a path exists

### Path Management

- `get_path(path, suffix=None)`: Get the full path with proper suffix
- `abspath(path)`: Convert a path to an absolute path
- `paths(search=None, avoid=None, max_age=None)`: List all paths in the store with optional filtering
- `ls(path=None, search=None, avoid=None)`: List files in a directory

### Directory Operations

- `rmdir(path)`: Remove a directory and all its contents

### Data Operations

- `items(search=None, df=False, features=None)`: Get all items as data or DataFrame
- `n()`: Count the number of items in the store
- `_rm_all()`: Remove all items in the store

### File Metadata

- `path2age()`: Get the age of all files in seconds
- `item2age()`: Get the age of all items in seconds
- `get_age(path)`: Get the age of a specific file in seconds
- `get_text(path)`: Get the text content of a file

### Cryptography

- `hash(content, hash_type='sha256')`: Hash content using specified algorithm
- `cid(path, ignore_names=['__pycache__', '.DS_Store', '.git', '.gitignore'])`: Get the content ID of a file or directory
- `encrypt(path, key=None)`: Encrypt a file using the given key
- `decrypt(path, key=None)`: Decrypt a file using the given key
- `is_encrypted(path)`: Check if a file is encrypted
- `encrypted_paths(path, key=None)`: Get the paths of encrypted files

## Testing

The Store module includes a TestStore class for testing the functionality:

```python
import commune as c

# Create a test store instance
test_store = c.TestStore()

# Test basic operations
result = test_store.test_basics()
print(result)  # {'success': True, 'msg': 'Passed all tests'}

# Test encryption/decryption
result = test_store.test_encrypt()
print(result)  # {'success': True, 'msg': 'Passed all tests'}
```

## License

Part of the Commune framework. See the Commune license for details.