# WebSocket Module

This Python module, WebSocket, uses the `commune` and `websockets` libraries to manage WebSocket connections on a specified IP address and port. This module essentially serves as a WebSockets server and client.

## Features

- Handles WebSocket connections and messages dynamically.
- Supports sending and receiving messages.
- It can be used as a WebSocket server and client.
- It can process the received message and send back a response.
- Can run on a specific IP address and port.

## Usage

Importing:
```python
from WebSocket import WebSocket
```

Initialize a WebSocket server:
```python
ws = WebSocket(ip='localhost', port=8080)
```

Set server IP address and port:
```python
ws.set_server(ip='localhost', port=8080)
```

Handle connection:
```python
ws.handle_connection()
```

Running the server:
```python
ws.run()
```

Sending requests from WebSocket client:
```python
WebSocket.send_requests(ip='localhost', port=8080)
```

The function `serve(cls, **kwargs)` can be used to serve the WebSocket server.

## Requirements

- Python 3.7+
- asyncio (For asynchronous operations)
- websockets (For WebSocket related functionalities)
- commune (For network related functionalities)

## Test

You can test the functionalities of the class `WebSocket` by initializing a server, connecting a client, and sending messages.
```python
ws = WebSocket(ip='localhost', port=8080)
ws.run()
WebSocket.send_requests(ip='localhost', port=8080)
```

## Note:

- The `set_server(ip:str = 'localhost', port=None,)` function sets the server's IP address and port.
- The `def run(self)` function starts the WebSocket server and waits for client requests.
- The `async_send_requests(ip:str = 'localhost', port:int = 8080)` function sends asynchronous requests to the WebSocket server.
