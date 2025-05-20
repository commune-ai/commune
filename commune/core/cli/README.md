# Commune CLI Module

## Overview

The Commune CLI is a pythonic command-line interface that provides a simple way to interact with the Commune library. Unlike traditional CLI tools that use argparse, Commune's CLI offers a more intuitive, Python-like experience for testing functions and modules.

## Basic Usage

The CLI follows two main patterns:

```bash
# Pattern 1: Default module is "module"
c {function_name} *args **kwargs

# Pattern 2: Specify both module and function
c {module_name}/{function_name} *args **kwargs
```

### Examples

```bash
# List files in current directory using the default module
c ls ./

# Equivalent to the above but with explicit module
c module/ls ./

# Get code of a module
c module/code
```

## Module Naming Conventions

Commune uses simplified naming conventions:

- `commune/module.py` → `commune`
- `storage/module.py` → `storage`
- `storage/storage/module.py` → `storage`

The root module is the one closest to the commune/ repository.

## Common Operations

### Creating a New Module

```bash
# CLI command
c new_module agi

# Equivalent Python code
# import commune as c
# c.new_module("agi")
```

This creates a new module called `agi` in the `modules` directory.

### Getting Module Configuration

```bash
# CLI command
c agi/config

# Equivalent Python code
# import commune as c
# c.module("agi").config()
```

If a module doesn't have a config or YAML file, keyword arguments will be used as the config.

### Getting Module Code

```bash
# CLI command
c agi/code

# Equivalent Python code
# import commune as c
# c.module("agi").code()
```

### Serving a Module

```bash
c serve module
```

### Calling Module Functions

```bash
# Basic function call
c call module/ask hey
# Equivalent to: c.call('module/ask', 'hey')
# or: c.connect('module').ask('hey')

# With additional parameters
c call module/ask hey stream=1
# Equivalent to: c.call('module/ask', 'hey', stream=1)
# or: c.connect('module').ask('hey', stream=1)
```

## Shortcuts and Tips

- `c` (with no arguments) navigates to the Commune repository
- `c module/` calls the module's forward function
- `c module/forward` explicitly calls the forward function
- `c module/add a=1 b=1` is equivalent to `c module/add 1 1`
- `c ai what is the point of love` calls the AI module with a prompt

## Current Limitations

- Lists and dictionaries are not directly supported in CLI arguments
- Only positional arguments are supported
- Only one function can be called at a time

## Python Equivalent

All CLI commands have equivalent Python code using the Commune library:

```python
import commune as c

# Example: listing files
c.ls('./')

# Example: calling a module function
c.call('module/ask', 'hey', stream=1)
```
