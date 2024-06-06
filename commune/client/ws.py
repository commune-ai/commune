import asyncio
import websockets
import commune as c
import json

class WSClient(c.Module):
    
    
    def __init__(self,
                 address:str = '0.0.0.0:50087', 
                 start:bool = True, 
                 network: dict = None,
                 ):
        if ':' in address:
            ip, port = address.split(':')
        self.ip = ip
        self.port = port
        namespace = c.namespace(network=network)
        self.address = namespace.get(address, None)
        

    def resolve_address(self, address=None):
        if address == None:
            address = self.address
        if not 'ws://' in address:
            address = f'ws://{address}'
        assert isinstance(address, str), f'address must be a string, not {type(address)}'
        return address

    def forward(self, data, address=None):
        return 
    
    async def async_forward(self, data='hello', address = None, **kwargs):
        address = self.resolve_address(address=address, **kwargs)
        async with websockets.connect(address) as websocket:
            await websocket.send(data)
            response = await websocket.recv()
        return response
    
    def forward(self, 
                fn:str = 'fn', 
                args:list = [], 
                kwargs:dict = {}, 
                address:str = None,
                timeout:int = 10,
                **extra_kwargs):
        
        data = {
            'fn': fn,
            'args': args,
            'kwargs': kwargs,
            **extra_kwargs
        }
        data = json.dumps(data)
        loop = asyncio.get_event_loop()
        future = self.async_forward(data=data, address=address)
        future = asyncio.wait_for(future, timeout=timeout)
        result = loop.run_until_complete(future)
        return result
    
    
    async def recv(self, address):
        chunks = []
        async with websockets.connect(address) as websocket:
            chunk = await websocket.recv(address)
            chunks.append(chunk)
        return chunks
    

    def test(self):
        print('testing')
        c.thread(self.async_forward, 'hello', self.address)
