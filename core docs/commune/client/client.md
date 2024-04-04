# README

The provided script defines a class `Client` that establishes a client-server connection using the Python asyncio library, commune library and aiohttp library along with some other standard libraries. `commune` is a Python library for inter-process communication (IPC) based on asyncio and contextvars. This class enables sending HTTP POST requests to the server with specified parameters and handling the response.

The `__init__` method initializes the client with a specific IP and port number, sets up the event loop, and assigns a serializer for encoding and decoding the messages. 

The `set_client` method sets the IP and port, the `resolve_client` method sets the client's IP and port if they are not None.

To send requests to a server, `async_forward` and `forward` methods could be used. The `async_forward` is an asynchronous function that sends an HTTP POST request to the server. It takes the function to be executed, its arguments, and some optional network parameters as inputs. It also handles event-driven responses from the server in different formats - event stream, json, or plain text. The results are processed and returned in the appropriate format. 

The `forward` method runs the HTTP POST request using the asyncio event loop, it also has an optional flag to return the future object instead of waiting for completion.

The `process_output` method processes the response received from the server and extracts the 'data' part of the response. 

The class method `history` returns previously saved transaction history for a specific key; `all_history` returns all transaction histories saved. 

The `virtual` method returns a virtual client for testing. 

To run it, please make sure to install the necessary libraries such as aiohttp, asyncio, and commune.

```bash
pip install aiohttp asyncio commune
```
Then, create the `Client` instance and use it:

```python
client = Client(ip="1.2.3.4", port=1234)
response = client.forward('function_name', args=['arg1', 'arg2'], kwargs={'arg3':'value3'})
```
The `response` stores the server's response to the sent HTTP POST request.
