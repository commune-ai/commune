

from typing import *
import asyncio
import commune as c
import json
import requests
from .tools import ClientTools
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
            loop = None, 
            **kwargs
        ):
        self.serializer = c.module('serializer')()
        self.network = network
        self.loop =  c.get_event_loop()
        self.key  = c.get_key(key, create_if_not_exists=True)
        self.module = module
        self.fn2max_age = fn2max_age
        self.stream_prefix = stream_prefix
        self.address = self.resolve_module_address(module, network=network)
        self.virtual = virtual
        self.session = requests.Session()

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
        client = cls.connect(module, virtual=False, key=key, network=network)
        response =  client.forward(fn=fn, 
                                args=args,
                                kwargs=kwargs, 
                                params=params,
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
    def request(self, url: str, data: dict, headers: dict, timeout: int = 10, stream: bool = True):
        response = self.session.post(url, json=data, headers=headers, timeout=timeout, stream=stream)
        try:             
            if 'text/event-stream' in response.headers.get('Content-Type', ''):
                return self.stream(response)
            if 'application/json' in response.headers.get('Content-Type', ''):
                result = response.json()
            elif 'text/plain' in response.headers.get('Content-Type', ''):
                result = response.text
            else:
                result = response.content
                if response.status_code != 200:
                    raise Exception(result)
            result = self.serializer.deserialize(result)
        except Exception as e:
            result = c.detailed_error(e)
        return result
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
                **extra_kwargs):
        network = network or self.network
        url = self.get_url(fn=fn, mode=mode,  network=network)
        data = data or self.get_data(args=args,  kwargs=kwargs, params=params, **extra_kwargs)
        headers = headers or self.get_headers(data=data, key=key)
        kwargs = {**(kwargs or {}), **extra_kwargs}
        result = self.request( url=url,data=data,headers= headers,timeout= timeout)
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


    def stream(self, response):
        buffer = ""
        
        for chunk in response.iter_lines():
            line = self.process_stream_line(chunk)
            yield line


    def tokenize(self, data):
        # Customize tokenization logic here. For example, split by spaces.
        # Returning a list of tokens and the last incomplete token (if any) as a string.
        tokens = data.split()
        return tokens
    def process_stream_line(self, line , stream_prefix=None):
        stream_prefix = stream_prefix or self.stream_prefix
        event_data = line.decode('utf-8')
        if event_data.startswith(stream_prefix):
            event_data = event_data[len(stream_prefix):] 
        event_data = event_data.strip() # remove leading and trailing whitespaces
        if event_data == "": # skip empty lines if the event data is empty
            return ''
        if isinstance(event_data, str):
            if event_data.startswith('{') and event_data.endswith('}') and 'data' in event_data:
                event_data = json.loads(event_data)['data']
        return event_data