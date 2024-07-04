import asyncio
import websockets
import commune as c

class WebSocket(c.Module):
    def __init__(self,
                 module = None,
                 ip :str = 'localhost',
                 port:int = None,
                 run:bool = True):
        
        self.module = module
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
        self.websocket_server = websockets.serve(self.handle_connection, ip, port)
        response = f"WebSocket server running on ws://{ip}:{port}/"
        c.print(response)
        return response 

    def run(self):
        asyncio.get_event_loop().run_until_complete(self.websocket_server)
        asyncio.get_event_loop().run_forever()
        
    @classmethod
    def send_requests(cls, module, fn, data,  port:int = 8080, ip:str = 'localhost'):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cls.async_send_requests(ip=ip, port=port))
    @staticmethod
    async def async_send_requests(module, fn, data, address='0.0.0.0:8000'):

        async with websockets.connect(f'ws://{address}/') as websocket:
            while True:
                await websocket.send(message)
                response = await websocket.recv()
                print("Received response: ", response)
    
    @classmethod    
    def serve(cls, **kwargs):
        cls(**kwargs).run()
        

