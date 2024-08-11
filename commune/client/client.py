

from typing import *
import asyncio
import commune as c
import aiohttp
import json
from .pool import ClientPool
from .virtual import VirtualClient
# from .pool import ClientPool

class Client(c.Module, ClientPool):
    network2namespace = {}
    def __init__( 
            self,
            module : str = '0.0.0.0:8000',
            network: bool = 'local',
            key = None,
            loop = None,
            **kwargs
        ):

        self.serializer = c.module('serializer')()
        self.session = aiohttp.ClientSession()
        self.network = network
        self.loop = c.get_event_loop() if loop == None else loop
        self.key  = c.get_key(key, create_if_not_exists=True)
        self.module = module



    def resolve_namespace(self, network):
        if not network in self.network2namespace:
            self.network2namespace[network] = c.get_namespace(network=self.network)
        return self.network2namespace[network]


    @classmethod
    def call(cls, 
                fn:str = 'info',
                *args,
                kwargs = None,
                params = None,
                module : str = 'module',
                network:str = 'local',
                key:str = None,
                timeout=40,
                **extra_kwargs) -> None:
        if '/' in str(fn):
            module = '.'.join(fn.split('/')[:-1])
            fn = fn.split('/')[-1]
        else:
            module = fn
            fn = 'info'
        client = cls.connect(module, virtual=False, network=network)
        return  client.forward(fn=fn, 
                                args=args,
                                kwargs=kwargs, 
                                params=params,
                                key=key  ,
                                timeout=timeout, 
                                **extra_kwargs)

    @classmethod
    def connect(cls,
                module:str = 'module', 
                network : str = 'local',
                virtual:bool = True, 
                **kwargs):
        
        client = cls(module=module, 
                    virtual=virtual, 
                    network=network,
                    **kwargs)
        # if virtual turn client into a virtual client, making it act like if the server was local
        if virtual:
            return client.virtual()
        
        return client
    
    def test(self, module='module::test_client'):
        c.serve(module)
        c.sleep(1)

        info = c.call(module+'/info')
        key  = c.get_key(module)
        assert info['key'] == key.ss58_address
        return {'info': info, 'key': str(key)}

    def __str__ ( self ):
        return "Client({})".format(self.address) 
    def __repr__ ( self ):
        return self.__str__()


    def virtual(self):
        return VirtualClient(module = self)
    
    def __repr__(self) -> str:
        return super().__repr__()


    def get_params(self, args=None, kwargs=None, params=None, version=1):
        if version == 1:
            input =  { 
                        "args": args or [],
                        "kwargs": params or kwargs or {},
                        "timestamp": c.timestamp(),
                        }
            input = self.serializer.serialize(input)
            input = self.key.sign(input, return_json=True)
        else:
            raise ValueError(f"Invalid version: {version}")

        return input


    def get_url(self, fn, mode='http', network=None):
        network = network or self.network
        if '://' in str(fn):
            mode ,fn = fn.split('://')
        if '/' in str(fn):  
            module, fn = module.split('/')
        else:
            module = self.module
        if '/' in str(fn):
            module, fn = fn.split('/')
        if not c.is_address(module):
            namespace = self.resolve_namespace(network)
            module = namespace[module]
        url = f"{module}/{fn}/"
        url = f'{mode}://' + url if not url.startswith(f'{mode}://') else url
        return url        


    def forward(self, *args, **kwargs):
        return self.loop.run_until_complete(self.async_forward(*args, **kwargs))

    async def async_forward(self, 
                            fn  = 'info', 
                            params = None, 
                            args=None,
                            kwargs = None,
                           timeout:int=10, 
                           module = None,
                           key = None,
                           headers = {'Content-Type': 'application/json'},
                           verbose=False, 
                           network = None,
                           version = 1,
                           mode = 'http',
                           **extra_kwargs):
        key = self.resolve_key(key)
        network = network or self.network
        url = self.get_url(fn=fn,mode=mode,  network=network)
        kwargs = {**(kwargs or {}), **extra_kwargs}
        input = self.get_params( args=args, 
                                   kwargs=kwargs, 
                                   params = params,
                                   version=version)
        c.print(f"🛰️ Call {url} 🛰️  (🔑{self.key})", color='green', verbose=verbose)

        try:
            response =  await self.session.post(url, json=input, headers=headers)
            if response.content_type == 'text/event-stream':
                return self.iter_over_async(self.stream_generator(response))
            if response.content_type == 'application/json':
                result = await asyncio.wait_for(response.json(), timeout=timeout)
            elif response.content_type == 'text/plain':
                result = await asyncio.wait_for(response.text(), timeout=timeout)
            else:
                result = await asyncio.wait_for(response.read(), timeout=timeout)
                # if its an error we will raise it
                if response.status != 200:
                    raise Exception(result)
            result = self.serializer.deserialize(result)
        except Exception as e:
            result = c.detailed_error(e)

        return result
    
    def __del__(self):
        try:
            if hasattr(self, 'session'):
                asyncio.run(self.session.close())
        except:
            pass

    def iter_over_async(self, ait):
        # helper async fn that just gets the next element
        # from the async iterator
        def get_next():
            try:
                obj = self.loop.run_until_complete(ait.__anext__())
                return obj
            except StopAsyncIteration:
                return 'done'
        # actual sync iterator (implemented using a generator)
        while True:
            obj = get_next() 
            if obj == 'done':
                break
            yield obj

    async def stream_generator(self, response):
        async for line in response.content:
            event =  self.process_stream_line(line)
            if event == '':
                continue
            yield event
        
        


    def resolve_key(self,key=None):
        if key == None:
            key = self.key
        if isinstance(key, str):
            key = c.get_key(key)
        return key
    
    def process_stream_line(self, line ):
        STREAM_PREFIX = 'data: '
        event_data = line.decode('utf-8')
        event_data = event_data[len(STREAM_PREFIX):] if event_data.startswith(STREAM_PREFIX) else event_data
        event_data = event_data.strip() # remove leading and trailing whitespaces
        if event_data == "": # skip empty lines if the event data is empty
            return ''
        if isinstance(event_data, str):
            if event_data.startswith('{') and event_data.endswith('}') and 'data' in event_data:
                event_data = json.loads(event_data)['data']
        return event_data


    @classmethod
    async def async_check_connection(cls, module, timeout=5, **kwargs):
        try:
            module = await cls.async_connect(module, return_future=False, virtual=False, **kwargs)
        except Exception as e:
            return False
        server_name =  await module(fn='server_name',  return_future=True)
        if cls.check_response(server_name):
            return True
        else:
            return False
        
    @staticmethod
    def check_response(x) -> bool:
        if isinstance(x, dict) and 'error' in x:
            return False
        else:
            return True
    
    @classmethod
    def check_connection(cls, *args, **kwargs):
        return c.gather(cls.async_check_connection(*args, **kwargs))

    @classmethod
    def module2connection(cls,modules = None, network=None):
        if modules == None:
            modules = c.servers(network=network)
        connections = c.gather([ c.async_check_connection(m) for m in modules])

        module2connection = dict(zip(modules, connections))
    
        return module2connection

  