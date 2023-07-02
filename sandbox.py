import asyncio
import websockets
import commune as c

class WebSocket(c.Module):
    def __init__(self, **kwargs):
        self.config = self.set_config(kwargs=kwargs)

    async def handle_connection(self, websocket, path):
        # Handle WebSocket connection here
        while True:
            message = await websocket.recv()
            print(f"Received message: {message}")

            # Process the message and send a response
            response = f"Processed message: {message}"
            await websocket.send(response)

    def run(self):
        server_address = (self.config.get('host', 'localhost'), self.config.get('port', 8080))
        print(f"WebSocket server started on: {server_address}")

        start_server = websockets.serve(self.handle_connection, *server_address)

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
        
    @staticmethod
    async def send_requests():
        async with websockets.connect('ws://localhost:8080/') as websocket:
            while True:
                message = input("Enter a message to send: ")
                await websocket.send(message)
                response = await websocket.recv()
                print("Received response: ", response)
    @classmethod    
    def start(cls):
        cls().run()

WebSocket.start()
