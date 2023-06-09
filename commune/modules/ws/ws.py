import asyncio
import websockets
import commune as c

class WS(c.Module):
    
    
    def __init__(self,
                 port:int=None,
                 buffer_size:int=-1):
        self.set_queue(buffer_size=buffer_size)
        self.queue.put(1000)
        self.set_server(port=port)
    
    @staticmethod
    async def send_file(filename, address):
        async with websockets.connect(address) as websocket:
            with open(filename, 'rb') as file:
                while True:
                    chunk = file.read(1024)
                    if not chunk:
                        break
                    await websocket.send(chunk)
                await websocket.send('END')

    @staticmethod
    async def recv(address):
        chunks = []
        async with websockets.connect(address) as websocket:
            chunk = await websocket.recv(address)
            chunks.append(chunk)

    def put(self, chunk):
        return self.queue.put(chunk)
    
    
    async def forward(self, websocket):
        while True:
            chunk = self.queue.get()
            if not chunk:
                break
            await websocket.send(chunk)
        
        await websocket.send('END')

    def set_queue(self, buffer_size:int):
        import queue
        self.queue = queue.Queue(buffer_size)
        

    def set_server(self,port=None, ip = '0.0.0.0'):
        port = self.resolve_port(port)  
        start_server = websockets.serve(self.forward, ip, port)
        self.print(f'Serving on {ip}:{port}')
        
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()



if __name__ == "__main__":
    ws = WS()
