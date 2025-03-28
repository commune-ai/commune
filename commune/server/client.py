

from typing import *
import asyncio
import json
import requests
import os
import commune as c

class Client:
    def __init__( self,  
                 url : str = 'module',  
                 key : Optional[str]= None,  
                 network: Optional[bool] = 'local', 
                 auth = 'server.auth.jwt',
                 mode='http',
                 history_path = '~/.commune/client/history',

                 **kwargs):
        self.auth = c.module(auth)()
        self.key  = c.get_key(key)
        self.url = self.get_url(url, mode=mode)
        self.history_path = os.path.abspath(os.path.expanduser(history_path))

    def forward(self, 
                fn  = 'info', 
                params: Optional[Union[list, dict]] = None, # if you want to pass params as a list or dict
                # if you want to pass positional arguments to the function, use args 
                args : Optional[list] = [], 
                kwargs : Optional[dict] = {},      
                ## adduitional parameters
                timeout:int=2,  # the timeout for the request
                key : str = None,  # the key to use for the request
                mode: str  = 'http', # the mode of the request
                stream: bool = False, # if the response is a stream
                **extra_kwargs 
    ):
        if '/' in str(fn):
            url, fn = '/'.join(fn.split('/')[:-1]), fn.split('/')[-1]
        else :
            url = self.url
        url = self.get_url(url, mode=mode)
        key = self.get_key(key) # step 1: get the key
        params = self.get_params(params=params, args=args, kwargs=kwargs, extra_kwargs=extra_kwargs) # step 3: get the params
        headers = self.auth.get_headers({'fn': fn, 'params': params}, key=key) # step 4: get the headers

        with requests.Session() as conn:
            response = conn.post( f"{url}/{fn}/", json=params,  headers=headers, timeout=timeout, stream=stream)
        ## handle the response
        if response.status_code != 200:
            raise Exception(response.text)
        if 'text/event-stream' in response.headers.get('Content-Type', ''):
            result = self.stream(response)
        if 'application/json' in response.headers.get('Content-Type', ''):
            result = response.json()
        elif 'text/plain' in response.headers.get('Content-Type', ''):
            result = response.text
        else:
            result = response.content
            if response.status_code != 200:
                raise Exception(result)
        if 'result' in result:
            result = result['result']
        return result
    
    def get_key(self,key=None):
        if key == None:
            return self.key
        if isinstance(key, str):
            key = c.get_key(key)
        return key

    def get_params(self, params=None, args=[], kwargs={}, extra_kwargs={}):
        params = params or {}
        args = args or []
        kwargs = kwargs or {}
        kwargs.update(extra_kwargs)
        if params:
            if isinstance(params, dict):
                kwargs = {**kwargs, **params}
            elif isinstance(params, list):
                args = params
            else:
                raise Exception(f'Invalid params {params}')
        params = {"args": args, "kwargs": kwargs}
        return params


    def get_url(self, url, mode='http'):
        if c.is_url(url):
            url = url
        elif c.is_int(url):
            url = f'0.0.0.0:{url}'
        else:
            url = c.namespace().get(str(url), url)
        url = url if url.startswith(mode) else f'{mode}://{url}'
        return url


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
                default_fn = 'info',
                **extra_kwargs) -> None:
        fn = str(fn)
        if not fn.startswith('http'):
            if '/' in str(fn):
                module = '.'.join(fn.split('/')[:-1])
                fn = fn.split('/')[-1]
            else:
                module = fn 
                fn = default_fn
        kwargs = (params or kwargs) or {}
        kwargs = {**kwargs, **extra_kwargs}
        return cls(url=module, network=network).forward(fn=fn, 
                                                            args=args, 
                                                            kwargs=kwargs, 
                                                            params=params,
                                                            timeout=timeout, 
                                                            key=key)

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
        client = Client(url=module)
        return ClientVirtual(client) if virtual else client
        return client

    @staticmethod
    def connect( module:str, **kwargs):
        """
        Connect to a module and return a client instance.
        """
        return Client.client(module, **kwargs)