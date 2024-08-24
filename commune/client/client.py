

from typing import *
import asyncio
import commune as c
import aiohttp
from .tools import ClientTools
from .sse import ClientSSE
from .virtual import ClientVirtual

# from .pool import ClientPool

class Client(c.Module, ClientTools):
    network2namespace = {}
    def __init__( 
            self,
            module : str = 'module',
            network: bool = 'local',
            key = None,
            stream_prefix = 'data: ',
            fn2max_age = {'info': 60, 'name': 60},
            virtual = False,
            **kwargs
        ):
        self.serializer = c.module('serializer')()
        self.network = network
        self.loop = asyncio.get_event_loop()
        self.key  = c.get_key(key, create_if_not_exists=True)
        self.module = module
        self.sse = ClientSSE()
        self.fn2max_age = fn2max_age
        self.stream_prefix = stream_prefix
        self.address = self.resolve_module_address(module, network=network)
        self.virtual = virtual

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
        response =  client.forward(fn=fn, 
                                args=args,
                                kwargs=kwargs, 
                                params=params,
                                key=key,
                                timeout=timeout, 
                                **extra_kwargs)

        return response

    @classmethod
    def connect(cls,
                module:str = 'module', 
                network : str = 'local',
                virtual:bool = True, 
                **kwargs):
        client =  cls(module=module, 
                    network=network,
                    **kwargs)
        if virtual:
            return ClientVirtual(client=client)
        return client
    
    def test(self, module='module::test_client'):
        c.serve(module)
        c.sleep(1)
        info = c.call(module+'/info')
        key  = c.get_key(module)
        assert info['key'] == key.ss58_address
        return {'info': info, 'key': str(key)}

    def __str__ ( self ):
        return "Client(address={}, virtual={})".format(self.address, self.virtual) 
    def __repr__ ( self ):
        return self.__str__()

    def __repr__(self) -> str:
        return super().__repr__()

    def resolve_module_address(self, module, mode='http', network=None):
        network = network or self.network
        if not c.is_address(module):
            namespace = self.resolve_namespace(network)
            if not module in namespace:
                namespace = c.get_namespace(network=network, update=1)
            url = namespace[module]
        else:
            url = module
        url = f'{mode}://' + url if not url.startswith(f'{mode}://') else url
        return url

    def get_url(self, fn, mode='http', network=None):
        
        if '://' in str(fn):
            mode ,fn = fn.split('://')
        if '/' in str(fn):  
            module, fn = module.split('/')
        else:
            module = self.module
        module_address = self.resolve_module_address(module, mode=mode, network=network)
        url = f"{module_address}/{fn}/"
        return url        

    async def async_request(self, url : str, data:dict, headers:dict, timeout:int=10):
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:   
                try:             
                    if response.content_type == 'text/event-stream':
                        return self.sse.stream(response)
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
    

    def request(self, url : str, data:dict, headers:dict, timeout:int=10):
        return self.loop.run_until_complete(self.async_request(url=url, data=data, headers=headers, timeout=timeout))
    
    def get_headers(self, data:Any, key=None, headers=None):
        key = self.resolve_key(key)
        if headers:
            # for relayed requests
            return headers
        headers = {'Content-Type': 'application/json', 
                    'key': key.ss58_address, 
                    'hash': c.hash(data),
                    'crypto_type': str(key.crypto_type),
                    'timestamp': str(c.timestamp())
                   }
        signature_data = {'data': headers['hash'], 'timestamp': headers['timestamp']}
        headers['signature'] = key.sign(signature_data).hex()
        return headers

    def get_data(self, args=[], kwargs={}, params={}, **extra_kwargs):
        # derefernece
        args = c.copy(args or [])
        kwargs = c.copy(kwargs or {})
        params = c.copy(params or {})
        if isinstance(args, dict):
            kwargs = {**kwargs, **args}
        if isinstance(params, dict):
            kwargs = {**kwargs, **params}
        elif isinstance(params, list):
            args = args + params
        if params:
            kwargs = {**kwargs, **params}
        if extra_kwargs:
            kwargs = {**kwargs, **extra_kwargs}
        data =  { 
                    "args": args,
                    "kwargs": kwargs,
                    }
        data = self.serializer.serialize(data)

        return data

    def forward(self, 
                fn  = 'info', 
                args : str = [],
                kwargs : str = {},
                params : dict = {}, 
                timeout:int=10, 
                key : str = None,
                network : str = None,
                mode: str  = 'http',
                headers = None,
                return_future = False,
                data = None,
                loop = None,
                fn2max_age = None,
                **extra_kwargs):
        network = network or self.network
        fn2max_age = fn2max_age or self.fn2max_age

        if fn in fn2max_age:
            c.print(f"'{fn}': {self.fn2max_age[fn]}")
            path = f'{self.address}/{fn}'
            max_age = self.fn2max_age[fn]
            response = self.get(path, None, max_age=max_age)
            if response != None:
                return response
            
        url = self.get_url(fn=fn, mode=mode,  network=network)
        data = data or self.get_data(args=args,  kwargs=kwargs,   params=params, key=key, **extra_kwargs)
        headers = headers or self.get_headers(data=data, key=key)
        kwargs = {**(kwargs or {}), **extra_kwargs}
        future = self.async_request( url=url,data=data,headers= headers,timeout= timeout)
        if  return_future:
            return future
        loop = loop or self.loop
        result =  self.loop.run_until_complete(future)
        return result
    
    def __del__(self):
        try:
            if hasattr(self, 'session'):
                asyncio.run(self.session.close())
        except:
            pass

    def resolve_key(self,key=None):
        if key == None:
            key = self.key
        if isinstance(key, str):
            key = c.get_key(key)
        return key
