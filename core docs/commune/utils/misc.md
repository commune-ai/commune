# RecursiveNamespace, SimpleNamespace and get_module

This Python module provides facilities for manipulating namespaces and accessing module-level objects dynamically.

## Classes:

### `SimpleNamespace`

- `__init__(self, **kwargs)`: Initializes a `SimpleNamespace` object with keys and values from the kwargs.

### `RecursiveNamespace`

- `__init__(self, **kwargs)`: Initializes a `RecursiveNamespace` object. This object behaves like a `SimpleNamespace` but also treats any dictionaries within the namespace as `RecursiveNamespace` objects, allowing for hierarchical access to nested data.

## Functions:

### `get_module(path,prefix = 'commune')`

This looks for and returns a Python module by fully qualified name. If the provided path doesn't have the prefix, it adds it before attempting to import. If the module cannot be found, it returns `None`.

- `path`: A dot-separated string representing the path to the module.
- `prefix`: A string to prepend to `path` if not already present.

### `cache(path='/tmp/cache.pkl', mode='memory')`

A function that returns a decorator for caching the output of other functions. The cache can either be in memory or in a local file.

This function is useful for speeding up access to costly or slow functions by saving their output for future use.

- `path`: The path where the cache should be stored. In 'local' mode, this is a file path. In 'memory' mode, it's a dictionary key.
- `mode`: Where to store the cache: 'local' for file, 'memory' for in this process's memory.

When applied to a function with `@cache`, the function's output will be cached. Future calls with the same arguments will return the cached output instantly.

## Usage:

These utilities are useful for dealing with dynamic access to modules and functions in Python code, which can be used for greater flexibility or customizability in large software systems. The caching function is useful for caching the result of expensive functions, to speed up subsequent calls with the same arguments.