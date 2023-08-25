import asyncio
import websockets
import commune as c

class WS(c.Module):
    
    
    def __init__(self,
                 ip = '0.0.0.0',
                 port:int=None,
                 queue_size:int=-1,
                 verbose:bool = True, 
                 module = None):
        self.serializer = c.module('serializer')()

        if module == None:
            self.module = c.module(module)()
        self.set_server(ip=ip, port=port, queue_size=queue_size, verbose=verbose)



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

        async for m in websocket:
            m = self.serializer.deserialize(m)


            m = self.serializer.serialize(m)
            await websocket.send(m)
            
            await websocket.send('<BREAK>')


    def __del__(self):
        c.deregister_server(self.name)



