

from typing import *
import asyncio
import json
import requests
import os
import commune as c

class Client:
    default_fn = 'info'
    def __init__( self,  
                 module : str = 'module',  
                 key : Optional[str]= None,  
                 network: Optional[bool] = 'local', 
                 **kwargs):
        self.key  = c.get_key(key)
        self.url = module if c.is_url(module) else c.namespace().get(module,module)
        self.session = requests.Session()
        
    @classmethod
    def call(cls, 
                fn:str = 'info',
                *args,
                kwargs = None,
                params = None,
                module : str = None,
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
                fn = cls.default_fn
        kwargs = (params or kwargs) or {}
        kwargs = {**kwargs, **extra_kwargs}
        return cls(module=module, network=network).forward(fn=fn, 
                                                            args=args, 
                                                            kwargs=kwargs, 
                                                            params=params,
                                                            timeout=timeout, 
                                                            key=key)

    def get_url(self, fn:str, mode='http'):
        if '/' in str(fn):  
            url, fn = '/'.join(fn.split('/')[:-1]), fn.split('/')[-1]
        else:
            url, fn = self.url, fn
        if not c.is_url(url):
            url = c.namespace().get(url, url)
        assert c.is_url(url), f'{url}'
        url = url if url.startswith(mode) else f'{mode}://{url}'
        url = f"{url}/{fn}/"
        return url

    def get_request(self,params: Union[list, dict] = None, 
                    key = None,
                    args = None, 
                    kwargs = None):
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
        params = {"args": args, "kwargs": kwargs}

        time_str = str(c.time())
        headers =  {
            'key': key.ss58_address,
            'crypto_type': str(key.crypto_type),
            'time': time_str,
            'signature':  key.sign({'params': params, 'time': time_str}).hex(),
            'Content-Type': 'application/json',
        } 


        return {'params': params, 'headers': headers}
    
        

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
    
        request = self.get_request(params=params, key=key, args=args, kwargs=kwargs  )
        print(request, url)
        response = self.session.post(
                                url, 
                                json=request['params'], 
                                headers=request['headers'], 
                                timeout=timeout, 
                                stream=stream
                                )
        return self.process_response(response)

        
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
        def process_stream_line(line , stream_prefix = 'data: '):
            event_data = line.decode('utf-8')
            if event_data.startswith(stream_prefix):
                event_data = event_data[len(stream_prefix):] 
            if event_data == "": # skip empty lines if the event data is empty
                return ''
            if isinstance(event_data, str):
                if event_data.startswith('{') and event_data.endswith('}') and 'data' in event_data:
                    event_data = json.loads(event_data)['data']
            return event_data
        try:
            for chunk in response.iter_lines():
                yield process_stream_line(chunk)
        except Exception as e:
            yield c.detailed_error(e)

    @staticmethod
    def is_url( url:str) -> bool:
        if not isinstance(url, str):
            return False
        if '://' in url:
            return True
        conds = []
        conds.append(isinstance(url, str))
        conds.append(':' in url)
        conds.append(c.is_int(url.split(':')[-1]))
        return all(conds)

    @staticmethod
    def client(module:str = 'module', network : str = 'local', virtual:bool = True, **kwargs):
        """
        Create a client instance.
        """
        
        class ClientVirtual:
            def __init__(self, client):
                self.client = client
            def remote_call(self, *args, remote_fn, timeout:int=10, key=None, **kwargs):
                return self.client.forward(fn=remote_fn, args=args, kwargs=kwargs, timeout=timeout, key=key)
            def __getattr__(self, key):
                if key in [ 'client', 'remote_call'] :
                    return getattr(self, key)
                else:
                    return lambda *args, **kwargs : self.remote_call(*args, remote_fn=key, **kwargs)
        client = Client(module=module)
        return ClientVirtual(client) if virtual else client
        return client

    @staticmethod
    def connect( module:str, **kwargs):
        """
        Connect to a module and return a client instance.
        """
        return Client.client(module, **kwargs)
    
    def __del__(self):
        try:
            if hasattr(self, 'session'):
                asyncio.run(self.session.close())
        except:
            pass