import asyncio
import websockets


class WebSocketServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        

    async def handle_client(self, websocket, path):
        # This method handles incoming messages from a client
        async for message in websocket:
            # Process the received message
            response = self.process_message(message)
            # Send the response back to the client
            await websocket.send(response)

    def process_message(self, message):
        # This method processes the received message and returns a response
        # You can implement your own logic here
        return f"Received message: {message}"

    def start(self):
        # Start the WebSocket server
        start_server = websockets.serve(self.handle_client, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    def test(self):
        # This method is used for testing the WebSocket server
        # It sends a sample message to the server and prints the response
        self.start()
        async def test_client():
            async with websockets.connect(f"ws://{self.host}:{self.port}") as websocket:
                message = "Hello, server!"
                await websocket.send(message)
                response = await websocket.recv()
                print(f"Received response: {response}")

        asyncio.get_event_loop().run_until_complete(test_client())

# Usage example
if __name__ == "__main__":
    server = WebSocketServer("localhost", 8765)
    server.test()
