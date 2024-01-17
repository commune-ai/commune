import asyncio
import websockets
import commune as c
from threading import Thread


def test_websocket_communication():
    # Define server IP and port
    server_ip = '0.0.0.0'
    server_port = 50087

    # Start the server in a separate thread
    def start_server():
        server = c.module('server.ws')(ip=server_ip, port=server_port)
        asyncio.get_event_loop().run_forever()

    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()

    # Give the server a moment to start
    c.sleep(1)

    # Create a client and connect to the server
    client = c.module('server.ws.client')(address=f'{server_ip}:{server_port}')

    # Send a message to the server and get the response
    test_message = "Hello, WebSocket!"
    response = client.forward(data=test_message)
    
    # Print the response
    print("Received response:", response)

# Run the test
test_websocket_communication()
