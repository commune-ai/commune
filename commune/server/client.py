

from typing import *
import asyncio
import json
import requests
import os
import commune as c

class Client:
    def __init__( self,  
                 module : str = 'module',  
                 key : Optional[str]= None,  
                 network: Optional[bool] = 'local', 
                 **kwargs):
        self.key  = c.get_key(key, create_if_not_exists=True)
        self.namespace = c.namespace(network=network)
        self.url = module if c.is_url(module) else self.namespace.get(module)
        self.session = requests.Session()
        
    @classmethod
    def call(cls, 
                fn:str = 'info',
                *args,
                kwargs = None,
                params = None,
                module : str = 'module',
                network:str = 'local',
                key: Optional[str] = None, # defaults to module key (c.default_key)
                timeout=40,
                **extra_kwargs) -> None:
        
        if not fn.startswith('http'):
            if '/' in str(fn):
                module = '.'.join(fn.split('/')[:-1])
                fn = fn.split('/')[-1]
            else:
                module = fn 
                fn = 'info'
        kwargs = (params or kwargs) or {}
        kwargs = {**kwargs, **extra_kwargs}
        return cls(module=module, network=network).forward(fn=fn, 
                                                            args=args, 
                                                            kwargs=kwargs, 
                                                            params=params,
                                                            timeout=timeout, 
                                                            key=key)

    @classmethod
    def client(cls, module:str = 'module', network : str = 'local', virtual:bool = True, **kwargs):
        client =  cls(module=module, network=network,**kwargs)
        return Client.Virtual(client=client) if virtual else client

    def get_url(self, fn, mode='http'):
        if '/' in str(fn):  
            url, fn = '/'.join(fn.split('/')[:-1]), fn.split('/')[-1]
            if url in self.namespace:
                url = self.namespace[url]
        else:
            url = self.url
        url = url if url.startswith(mode) else f'{mode}://{url}'
        return f"{url}/{fn}/"

    def get_params(self,params: Union[list, dict] = None, args = None, kwargs = None):
        params = params or {}
        args = args or []
        kwargs = kwargs or {}
        if params:
            if isinstance(params, dict):
                kwargs = {**kwargs, **params}
            elif isinstance(params, list):
                args = params
            else:
                raise Exception(f'Invalid params {params}')
        params =  {"args": args, "kwargs": kwargs}
        return params

    def forward(self, 
                fn  = 'info', 
                params: Optional[Union[list, dict]] = None, # if you want to pass params as a list or dict
                timeout:int=2,  
                key : str = None,  
                mode: str  = 'http', 
                stream:bool = False, 
                # if you want to pass positional arguments to the function, use args 
                args : Optional[list] = [], 
                kwargs : Optional[dict] = {},              
    ):
                
        key = self.resolve_key(key)
        url = self.get_url(fn=fn, mode=mode)
        params = self.get_params(params=params, args=args, kwargs=kwargs )
        headers =self.get_header(params=params, key=key)
        response = self.session.post(url, json=params, headers=headers, timeout=timeout, stream=stream)
        result = self.process_response(response)
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
    
    def process_response(self, response):
        if 'text/event-stream' in response.headers.get('Content-Type', ''):
            return self.stream(response)
        if 'application/json' in response.headers.get('Content-Type', ''):
            result = response.json()
        elif 'text/plain' in response.headers.get('Content-Type', ''):
            result = response.text
        else:
            # if the response is not json or text, return the content
            result = response.content
            if response.status_code != 200:
                raise Exception(result)
        return result

    def stream(self, response):
        try:
            for chunk in response.iter_lines():
                yield self.process_stream_line(chunk)
        except Exception as e:
            yield c.detailed_error(e)

    def process_stream_line(self, line , stream_prefix = 'data: '):
        event_data = line.decode('utf-8')
        if event_data.startswith(stream_prefix):
            event_data = event_data[len(stream_prefix):] 
        if event_data == "": # skip empty lines if the event data is empty
            return ''
        if isinstance(event_data, str):
            if event_data.startswith('{') and event_data.endswith('}') and 'data' in event_data:
                event_data = json.loads(event_data)['data']
        return event_data
        
    class Virtual:
        def __init__(self, client: str ='ReactAgentModule'):
            if isinstance(client, str):
                client = c.connect(client)
            self.client = client
        def remote_call(self, *args, remote_fn, timeout:int=10, key=None, **kwargs):
            result =  self.client.forward(fn=remote_fn, args=args, kwargs=kwargs, timeout=timeout, key=key)
            return result
        def __getattr__(self, key):
            if key in [ 'client', 'remote_call'] :
                return getattr(self, key)
            else:
                return lambda *args, **kwargs : self.remote_call(*args, remote_fn=key, **kwargs)

    def get_header(self, params, key: 'Key'):
        time_str = str(c.time())
        return {
            'key': key.ss58_address,
            'crypto_type': str(key.crypto_type),
            'time': time_str,
            'Content-Type': 'application/json',
            'signature':  key.sign({'params': params, 'time': time_str}).hex()
        } 
    
    @classmethod
    def connect(cls, module, **kwargs):
        return cls.client(module, **kwargs)
