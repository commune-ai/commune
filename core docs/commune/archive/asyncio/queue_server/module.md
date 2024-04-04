# AsyncQueueServer Module

This script is a Python module for creating an Asynchronous Queue Server using asyncio and Ray. 

## Features 
1. **Create Queues**: The server allows for the creation of named queues in memory using the `create_queue` method.
2. **Batch Operations**: Allows for the addition of batches of jobs and retrieval of batches of jobs using the `put_batch` and `get_batch` methods respectively.
3. **Asynchronous Operations**: The server can perform asynchronous operation execution, including 'put' and 'get' operations from a queue. 
4. **Manage Queues**: Queues can be checked, added, and removed with their respective methods. The queue size and status (empty or full) can also be determined.

## Installation & Usage

To use, simply import the module at the top of your Python script:
```python
import ray
from commune.block.ray.queue import Queue
from commune import Module
from commune.utils import dict_put,dict_get,dict_has,dict_delete
from copy import deepcopy
```
A server object can be instantiated using `server = AsyncQueueServer()`. The module can be run as a standalone script to conduct a demo using Streamlit.

## Example

Below is a demonstration of how to use the module:
```python
import asyncio
from async_queue_server import AsyncQueueServer
import streamlit as st

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server = AsyncQueueServer()

st.write(server.put_batch('key', ['bro']*10, sync=True))
st.write(server.put_batch('bro', ['bro']*10, sync=True))
st.write(server.get_batch('key', batch_size=10, sync=True))
```

## Contributing
We welcome contributions! Please include a detailed description of the changes, tests where necessary, and pass any pre-existing tests.
