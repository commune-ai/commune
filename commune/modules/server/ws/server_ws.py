import asyncio
import websockets
import commune as c

class WS(c.Module):
    
    
    def __init__(self,
                 ip = '0.0.0.0',
                 port:int=None,
                 queue_size:int=-1,
                 verbose:bool = True):
        self.set_server(ip=ip, port=port, queue_size=queue_size, verbose=verbose)

    @staticmethod
    def start(**kwargs):
        WS(**kwargs)

    def set_server(self, ip = '0.0.0.0', port = None, queue_size = -1, verbose = False):
        self.ip = c.resolve_ip(ip)
        self.port = c.resolve_port(port)
        self.queue = c.queue(queue_size)
        self.address = f'ws://{self.ip}:{self.port}'
        self.server = websockets.serve(self.forward, self.ip, self.port)
        c.print(f'Starting Server on {self.ip}:{self.port}')
        asyncio.get_event_loop().run_until_complete(self.server)
        asyncio.get_event_loop().run_forever()


    def put(self, chunk):
        return self.queue.put(chunk)
    
    async def forward(self, websocket):
        c.print(f'Starting Server Forwarding from {self.ip}:{self.port}')

        while True:
            try:
                c.print('waiting for data')
                data = await websocket.recv()
                c.print(f'chunk -> {data}')
                await websocket.send(data)
                c.print(f'sent -> {data}')
            except Exception as e:
                c.print(f'An error occurred: {e}')  

    @classmethod
    def test(cls):
        c.print('Starting test')
        cls.remote_fn(fn='start', kwargs={})
        c.print('Finished test')


