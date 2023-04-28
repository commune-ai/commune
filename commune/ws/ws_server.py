import threading
import asyncio
import websockets

import commune

class WSServer(commune.Module):
    threads = []
    def __init__(self, config = None):
        config = self.set_config(config)
        self.set_server(host=config.host, port=config.port)
        
    async def handle_client(self, websocket, path):
        # This function will be called whenever a new client connects to the server.
        # You can use this function to define how to handle incoming messages from the client.
        # For example, you can use a while loop to keep listening for messages until the client disconnects:
        while True:
            message = await websocket.recv()
            print(f"Received message from client: {message}")
            
            # You can also send messages back to the client using the `send()` method:
            response = f"You sent me this message: {message}"
            await websocket.send(response)



    # Define a function to create a new WebSocket server and run it on a separate thread:
    

    def set_server(self, host:str = None, port:int = None):

        host = self.config.host if host == None else host
        port = self.config.port if port == None else port


    def start_server(self):
        thread = threading.Thread(target=self.run_server, args=(host, port))
        thread.deamon = self.config.deamon  # Set the daemon attribute to True
        thread.start()
        self.threads.append(thread)
        
    def run_server(self, host:str = None, port:int = None):
        asyncio.set_event_loop(asyncio.new_event_loop())
        server = websockets.serve(self.handle_client, host, port)
        asyncio.get_event_loop().run_until_complete(server)
        asyncio.get_event_loop().run_forever()

        
    @classmethod
    def test(cls):
        self = cls()
        
WSServer()