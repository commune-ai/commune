# Commune Module: A Comprehensive Guide

This document provides a detailed guide to using the `commune` library for managing and serving code modules in Python. The `commune` library simplifies module management and deployment, allowing you to easily manage, serve, and interact with code modules.

## Table of Contents

- [Introduction to Modules](#introduction-to-modules)
- [Installation](#installation)
- [Setting Up Commune](#setting-up-commune)
  - [With Docker](#with-docker)
  - [Without Docker](#without-docker)
- [Basic Usage](#basic-usage)
  - [Creating a New Module](#creating-a-new-module)
  - [Finding Modules](#finding-modules)
  - [Serving Modules](#serving-modules)
- [Commune CLI](#commune-cli)
  - [CLI Syntax](#cli-syntax)
  - [Examples](#examples)
  - [CLI Tricks](#cli-tricks)
  - [Limitations](#limitations)
- [Module Management](#module-management)
  - [Accessing Module Information](#accessing-module-information)
  - [Viewing Module Configuration](#viewing-module-configuration)
  - [Listing Module Functions](#listing-module-functions)
  - [Function Schema](#function-schema)
- [Key Management](#key-management)
  - [Listing Keys](#listing-keys)
  - [Adding and Removing Keys](#adding-and-removing-keys)
  - [Getting Key Info](#getting-key-info)
  - [Saving and Loading Keys](#saving-and-loading-keys)
  - [Balance and Stake](#balance-and-stake)
- [Server Management](#server-management)
  - [Viewing Available Servers](#viewing-available-servers)
  - [Viewing Server Logs](#viewing-server-logs)
  - [Connecting to a Served Module](#connecting-to-a-served-module)
  - [Restarting a Module](#restarting-a-module)
- [Serializer](#serializer)
- [Port Range](#port-range)
- [Modules Folder](#modules-folder)
- [Conclusion](#conclusion)

---

## Introduction to Modules

A module in `commune` is a self-contained unit of code that can be managed and served easily. Modules can be created, searched, and served using the `commune` library, providing a streamlined approach to code management.

## Installation

**Environment Requirements**

- Python 3.10 or higher
- Node.js 14 or higher
- npm 6 or higher

1.  **Clone the Commune Repository**:

    ```bash
    git clone https://github.com/commune-ai/commune.git
    ```

2.  **Install Commune**:

    ```bash
    pip install -e ./commune
    ```

3.  **Run Tests**:

    To ensure the installation is working correctly, run the tests:

    ```bash
    c test
    ```

## Setting Up Commune

### With Docker

1.  **Install Docker**:

    Ensure Docker is installed on your machine. Follow the official Docker installation guide for your operating system.

2.  **Build the Docker Image**:

    Navigate to the cloned Commune repository and build the Docker image using the provided `Dockerfile`. This can be done via the `docker-compose` file:

    ```bash
    make start
    ```

3.  **Start Container**:

    Start a Docker container with the Commune image:

    ```bash
    make start
    ```

4.  **Enter the Container**:

    Enter the Docker container:

    ```bash
    make enter
    ```

    To exit the container, run:

    ```bash
    exit
    ```

    To run commands inside the container:

    ```bash
    docker exec -it commune bash -c "c modules"
    ```

5.  **Kill the Container**:

    To stop and remove the container:

    ```bash
    make down
    ```

### Without Docker

1.  **Install Dependencies**:

    Navigate to the cloned Commune repository and install the required dependencies:

    ```bash
    cd commune
    pip install -e ./
    ```

2.  **Install npm and pm2**:

    This is required for the web server to run:

    ```bash
    chmod +x ./run/*
    sudo ./run/install_npm_env.sh
    ```

3.  **Verify Installation**:

    Check if Commune is installed correctly:

    ```bash
    c modules
    ```

## Basic Usage

### Creating a New Module

To create a new module, use the `commune` command line tool:

```bash
c new_module agi
```

Alternatively, you can use the Pythonic interface:

```python
import commune as c
c.new_module('agi')
```

This will create a new module named `agi` in the `modules` directory.

### Finding Modules

To search for a specific module, use the `c.modules()` function with a search query:

```bash
c modules model.openai
```

Or in Python:

```python
import commune as c
c.modules('model.openai')
```

Output:

```
['model.openai']
```

### Serving Modules

To serve a module, use the `c serve` command:

```bash
c serve agi
```

This command starts the `agi` module as a service.

## Commune CLI

The `commune` CLI provides a Pythonic interface for interacting with the `commune` library. It allows you to test functions and modules without the need for extensive argument parsing.

### CLI Syntax

There are two primary ways to use the CLI:

1.  **Default Module Context**:

    ```bash
    c {fn} *args **kwargs
    ```

    In this case, the default module is "module".

2.  **Specify Module and Function**:

    ```bash
    c {module}/{fn} *args **kwargs
    ```

    Here, you explicitly specify the module and function to be executed.

    Example:

    ```bash
    c module/ls ./
    ```

### Examples

1.  **Listing Directory Contents**:

    ```bash
    c ls ./  # Same as c module/ls ./
    ```

    Equivalent Python code:

    ```python
    import commune as c
    c.ls('./')
    ```

2.  **Getting Module Configuration**:

    ```bash
    c agi/config
    ```

    This retrieves the configuration of the `agi` module.

    Equivalent Python code:

    ```python
    import commune as c
    c.module("agi").config()
    ```

3.  **Getting Module Code**:

    ```bash
    c agi/code
    ```

    This retrieves the code of the `agi` module.

4.  **Calling Module Functions**:

    To call the `ask` function of the `model.openai` module:

    ```bash
    c call module/ask hey
    ```

    To include positional arguments:

    ```bash
    c call module/ask hey stream=1
    ```

### CLI Tricks

-   `c` (without arguments): Navigates to the `commune` repository.

### Limitations

-   Lists and dictionaries are not directly supported as CLI arguments.
-   Only positional arguments are supported.
-   Only one function can be called at a time.

## Module Management

### Accessing Module Information

To access a module, use the `c.module()` function:

```python
demo = c.module('demo')
c.print('## Code for demo module')
c.print(demo.code())
```

### Viewing Module Configuration

View the configuration of a module using the `config()` method:

```python
demo.config()
```

Output:

```
{
    'name': 'demo',
    'version': '0.1.0',
    'description': 'A demo module for testing purposes',
    'author': 'John Doe',
    'email': '',
    'license': 'MIT',
}
```

### Listing Module Functions

List the functions of a module using the `fns()` method:

```python
demo.fns()
```

Output:

```
['test', 'forward']
```

### Function Schema

Retrieve the schema of a specific function using the `schema()` method:

```python
c.module('model.openai').schema()
```

Output:

```
{
    '__init__': {'input': {'a': 'int'}, 'default': {'a': 1}, 'output': {}, 'docs': None, 'type': 'self'},
    'call': {'input': {'b': 'int'}, 'default': {'b': 1}, 'output': {}, 'docs': None, 'type': 'self'}
}
```

## Key Management

### Listing Keys

List all available keys using the `keys()` function:

```bash
c keys
```

Or in Python:

```python
c.keys()
```

### Adding and Removing Keys

#### Adding a New Key

To add a new key, use the `add_key()` function:

```python
c.add_key('fam')
```

Or:

```bash
c add_key fam
```

#### Removing a Key

To remove a key, use the `rm_key()` function:

```python
c.rm_key('demo')
```

### Getting Key Info

Retrieve key information using the `key_info()` function:

```python
c.key_info('fam')
```

### Saving and Loading Keys

#### Saving Keys

To save the keys, use the `save_keys()` function:

```python
c.save_keys(path='./keys.json')
```

#### Loading Keys

To load the saved keys, use the `load_keys()` function:

```python
c.load_keys('./keys.json')
```

### Balance and Stake

#### Balance

To get the balance for a key, use the `get_balance()` function:

```python
c.balance('fam')
```

Or:

```bash
c balance fam
```

#### Get Stake of the Key

get a map of the validators that you are staked to.

```bash
c staketo fam
```

Or:

```python
c.staketo('fam', netuid='text')
```





## Server Management

### Viewing Available Servers

View the available servers using the `servers()` method:

```bash
c servers
```

Or:

```python
c.servers()
```

### Viewing Server Logs

View the logs of a served module using the `logs()` method:

```python
c logs demo
```

Or:

```python
c.logs('demo')
```

### Connecting to a Served Module

Connect to a served module using the `connect()` method:

```python
c call demo/info
```

### Restarting a Module

Restart a served module using the `restart()` method:

```python
c.restart('demo')
```

## Serializer

The serializer is responsible for ensuring that objects are JSON serializable. It handles various data types, including dictionaries, lists, and primitive types. For non-JSON serializable types, it uses custom serialization 
functions.

To add a new serializable type, define `serialize_{type}` and `deserialize_{type}` functions:

```python
# File: commune/serializer/serializer.py
from typing import Any, Dict, Union

def serialize_{type}(obj: {type}) -> Dict:
    return {"value": obj.value}

def deserialize_{type}(data: Dict[str, Any]) -> {type}:
    return {type}(data["value"])
```

## Port Range

To check the current port range:

```bash
c port_range
```

To set a new port range:

```bash
c set_port_range 8000 9000
```

## Modules Folder

The modules folder, located by default in `~/modules`, contains the code for imported modules. You can define this path using `c.home_path`.

## Conclusion

This tutorial covered the essential aspects of module management using the `commune` library. You've learned how to create, find, manage, serve, and interact with modules, as well as how to manage keys and configure servers. This library greatly simplifies the process of managing and deploying code modules in your projects.