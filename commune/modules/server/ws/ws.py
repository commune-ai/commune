import asyncio
import websockets
import commune as c

class WS(c.Module):
    
    
    def __init__(self,
                 ip = '0.0.0.0',
                 port:int=None,
                 queue_size:int=-1,
                 verbose:bool = False,
                 start:bool = True):
        
        self.ip = c.resolve_ip(ip)
        self.port = c.resolve_port(port)
        self.verbose = verbose
        self.queue = c.queue(queue_size)
        if start == True:
            self.start()

    def start(self):
        asyncio.get_event_loop().run_until_complete(websockets.serve(self.forward, self.ip, self.port))
        asyncio.get_event_loop().run_forever()

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
            data = await websocket.recv()
            c.print(f'chunk -> {chunk}')
            await websocket.send(data)
            c.print(f'sent -> {chunk}')
        
        await websocket.send('END')




    @classmethod
    def test(cls):
        self =  cls()
        self.put('hello')

