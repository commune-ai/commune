# asyncio Utility Functions

This Python script contains utility functions to work with asyncio, aiofiles and files I/O operations. Specifically it includes functions for reading and writing to files asynchronously, creating new event loops, getting an existing event loop, and a decorator for wrapping synchronous functions to make them asynchronous.

## Code Details
- `async_read(path, mode='r')` is an asynchronous function for reading data from a file at a specified path.
- `async_write(path, data, mode='w')` is an asynchronous function for writing data to a file at a specified path.
- `get_new_event_loop(nest_asyncio:bool = False)` is a function for creating a new asyncio event loop. If `nest_asyncio` is true, it allows for the event loop to be nested using the nest_asyncio module.
- `get_event_loop(new_event_loop:bool = False, nest_asyncio:bool = False)` is a function for getting an existing event loop. If one does not exist, it creates a new one. If `new_event_loop` or `nest_asyncio` is true, it behaves as their respective functions.
- `sync_wrapper(fn)` is a decorator for turning a synchronous function into an asynchronous one.

## Required Libraries
The required libraries for this script are asyncio, aiofiles, and optionally nest_asyncio.

## Requirements and Installation
You need Python 3+ to run this script. You can install required packages using pip:

```bash
pip install asyncio aiofiles nest_asyncio
```

## How to Use
You can use the functions in this script by importing them into your project.

```python
from your_module import async_read, async_write, get_new_event_loop, get_event_loop, sync_wrapper
```

You can then use them in your code like so:

```python
data = async_read("/path/to/file")
async_write("/path/to/file", data)
loop = get_event_loop(new_event_loop=True)
function = sync_wrapper(your_function)
```

**Note:** Replace 'your_module' in the import statement with the name of your Python file. Ensure you have tested your code thoroughly before deploying it to any production environment.
