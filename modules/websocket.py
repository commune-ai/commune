import asyncio
import websockets
import commune as c

class WebSocket(c.Module):
    def __init__(self,
                 ip :str = 'localhost',
                 port:int = None,
                 run:bool = True):
        
        self.set_server(ip=ip, port=port)
        if run:
            self.run()

    async def handle_connection(self, websocket, path):
        # Handle WebSocket connection here
        while True:
            message = await websocket.recv()
            print(f"Received message: {message}")

            # Process the message and send a response
            response = f"Processed message: {message}"
            await websocket.send(response)

    def set_server(self, ip:str = 'localhost', port=None,):
        port = c.resolve_port(port)
        self.server = websockets.serve(self.handle_connection, ip, port)

    def run(self):
        asyncio.get_event_loop().run_until_complete(self.server)
        asyncio.get_event_loop().run_forever()
        
    @classmethod
    def send_requests(cls, ip:str = 'localhost', port:int = 8080):
        asyncio.get_event_loop().run_until_complete(cls.async_send_requests(ip=ip, port=port))
    @staticmethod
    async def async_send_requests(ip:str = 'localhost', port:int = 8080):
        async with websockets.connect(f'ws://{ip}:{port}/') as websocket:
            while True:
                message = input("Enter a message to send: ")
                await websocket.send(message)
                response = await websocket.recv()
                print("Received response: ", response)
    
    @classmethod    
    def serve(cls, **kwargs):
        c.print(kwargs)
        cls(**kwargs).run()
        

