

from typing import *
import asyncio
import commune as c
import aiohttp
import json
from .pool import ClientPool
from .virtual import VirtualClient
# from .pool import ClientPool

class Client(c.Module, ClientPool):
    def __init__( 
            self,
            module : str = '0.0.0.0:8000',
            network: bool = 'local',
            key = None,
            loop = None,
            **kwargs
        ):

        self.serializer = c.module('serializer')()

        self.loop = c.get_event_loop() if loop == None else loop
        self.key  = c.get_key(key, create_if_not_exists=True)
        # we dont want to load the namespace if we have the address
        if not c.is_address(module):
            namespace = c.get_namespace(search=module, network=network)
            if module in namespace:
                module = namespace[module]
        self.module = module

    @classmethod
    def call(cls, 
                module : str, 
                fn:str = 'info',
                *args,
                kwargs = None,
                params = None,
                network:str = 'local',
                key:str = None,
                stream = False,
                timeout=40,
                **extra_kwargs) -> None:
          
        # if '
        if '//' in module:
            module = module.split('//')[-1]
            mode = module.split('//')[0]
        if '/' in module:
            if fn != None:
                args = [fn] + list(args)
            module , fn = module.split('/')

        module = cls.connect(module=module,
                           network=network,  
                           virtual=False, 
                           key=key)

        if params != None:
            kwargs = params

        if kwargs == None:
            kwargs = {}

        kwargs.update(extra_kwargs)

        return  module.forward(fn=fn, args=args, kwargs=kwargs, stream=stream, timeout=timeout)

    @classmethod
    def connect(cls,
                module:str, 
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
        c.print(c.server_exists(module))
        c.print('Module started')

        info = c.call(module+'/info')
        key  = c.get_key(module)
        assert info['key'] == key.ss58_address
        return {'info': info, 'key': str(key)}



    def __del__(self):
        if hasattr(self, 'session'):
            asyncio.run(self.session.close())


    def __str__ ( self ):
        return "Client({})".format(self.address) 
    def __repr__ ( self ):
        return self.__str__()


    def virtual(self):
        return VirtualClient(module = self)
    
    def __repr__(self) -> str:
        return super().__repr__()

    def forward(self,
                       *args,
                       timeout:int=10,
                          **kwargs):
        return self.loop.run_until_complete(asyncio.wait_for(self.async_forward(*args, **kwargs), timeout=timeout))

    async def async_forward(self, 
                             fn  = 'info', 
                             params = None, 
                             args=None,
                                kwargs=None,
                           timeout:int=10, 
                           stream = False,
                           module = None,
                           key = None,
                           headers = {'Content-Type': 'application/json'},
                           verbose=False, 
                           **extra_kwargs):
        

        if '/' in str(fn):
            module, fn = fn.split('/')
        key = self.resolve_key(key)
        module = module or self.module
        if '/' in module.split('://')[-1]:
            module = module.split('://')[-1]
        url = f"{module}/{fn}/"
        url = 'http://' + url if not url.startswith('http://') else url

        if params != None:
            assert type(params) in [list, dict], f'params must be a list or dict, not {type(params)}'
            if isinstance(params, list):
                args = params
            elif isinstance(params, dict):
                kwargs = params  

        input =  { 
                    "args": args or [],
                    "kwargs": {**(kwargs or {}), **extra_kwargs},
                    "timestamp": c.timestamp(),
                    }
        input = self.serializer.serialize(input)
        input = self.key.sign(input, return_json=True)  

        # start a client session and send the request
        c.print(f"🛰️ Call {url} 🛰️  (🔑{self.key.ss58_address})", color='green', verbose=verbose)
        self.session = aiohttp.ClientSession()
        try:
            response =  await self.session.post(url, json=input, headers=headers)
            if response.content_type == 'application/json':
                result = await asyncio.wait_for(response.json(), timeout=timeout)
            elif response.content_type == 'text/plain':
                result = await asyncio.wait_for(response.text(), timeout=timeout)
            elif response.content_type == 'text/event-stream':
                if stream:           
                    return self.stream_generator(response)
                else:
                    result = []  
                    async for line in response.content:
                        event =  self.process_stream_line(line)
                        if event == '':
                            continue
                        result += [event]
                    # process the result if its a json string
                    if isinstance(result, str):
                        if (result.startswith('{') and result.endswith('}')) or result.startswith('[') and result.endswith(']'):
                            result = ''.join(result)
                            result = json.loads(result)
            else:
                raise ValueError(f"Invalid response content type: {response.content_type}")
        except Exception as e:
            result = c.detailed_error(e)
        if type(result) in [str, dict, int, float, list, tuple, set, bool, type(None)]:
            result = self.serializer.deserialize(result)
            if isinstance(result, dict) and 'data' in result:
                result = result['data']
        else: 
            result = self.iter_over_async(result)

        await self.session.close()
        return result
    
    def __del__(self):
        if hasattr(self, 'session'):
            self.loop.run_until_complete(self.session.close())

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

  