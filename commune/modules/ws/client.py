import asyncio
import websockets
import commune as c

class WSClient(c.Module):
    
    
    def __init__(self,
                 ip = '0.0.0.0',
                 port:int=None,
                 queue_size:int=-1,
                 verbose:bool = False,
                 start:bool = True):
        
        if ':' in ip:
            ip, port = ip.split(':')
        self.ip = c.resolve_ip(ip)
        self.port = c.resolve_port(port)
        self.address = f'ws://{self.ip}:{self.port}'
        self.verbose = verbose

    def resolve_address(cls, address=None):
        if address == None:
            address = self.address
        if not 'ws://' in address:
            address = f'ws://{address}'
        assert isinstance(address, str), f'address must be a string, not {type(address)}'
        return address

    async def async_forward(self, data='hello', address = None):

        address = self.resolve_address(address=address)
        

        
        
        async with websockets.connect(address) as websocket:
            await websocket.send(data)
            response = await websocket.recv()
        
        return response
    
    def forward(self, data='hello', address = None):
        return asyncio.get_event_loop().run_until_complete(self.async_forward(data=data, address=address))

    @staticmethod
    async def recv(address):
        chunks = []
        async with websockets.connect(address) as websocket:
            chunk = await websocket.recv(address)
            chunks.append(chunk)
