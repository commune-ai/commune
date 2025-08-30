
from typing import *
import asyncio
import json
import requests
import os
import commune as c

class Client:
    def __init__( self,  
                 url: Optional[str] = None,  # the url of the commune server
                 key : Optional[str]= None,  
                 timeout = 10,
                 auth = 'auth',
                 storage_path = '~/.commune/client',
                 **kwargs):
        self.url = url
        self.auth = c.mod(auth)()
        self.key  = c.get_key(key)
        self.store = c.mod('store')(storage_path)
        self.timeout = timeout

        # ensure info from the server is fetched

    def forward(self, 
                fn  = 'info', 
                params: Optional[Union[list, dict]] = {}, # if you want to pass params as a list or dict
                timeout:int=None,  # the timeout for the request
                key : str = None,  # the key to use for the request
                mode: str  = 'http', # the mode of the request
                cost=0,
                update_info: bool = False, # whether to update the info from the server
                stream: bool = True, # if the response is a stream
                **extra_kwargs 
    ):

        # step 1: get the url and fn
        if '/' in str(fn):
            url, fn = '/'.join(fn.split('/')[:-1]), fn.split('/')[-1]
        else: 
            if self.url == None:
                url = fn
                fn = 'info'
            else: 
                url = self.url
        # step 2 : get the key
        key = self.get_key(key)
        c.print(f'Client({url}/{fn} key={key.name})', color='yellow')
        # step 3: get the params
        params = params or {}
        params.update(extra_kwargs)   
        url = self.get_url(url, mode=mode)
        # step 4: get the headers/auth if it is not provided
        headers = self.auth.forward({'fn': fn, 'params': params}, key=key, cost=cost)

        result = self.post(
            url=url, 
            fn=fn,
            params=params, 
            headers=headers, 
            timeout=timeout, 
            stream=stream
        )
        return result

    def post(self, url, fn,  params=None, headers=None, timeout=None, stream=False):
        # step 5: make the request
        timeout = timeout or self.timeout
        with requests.Session() as conn:
            response = conn.post( f"{url}/{fn}/", json=params,  headers=headers, timeout=timeout, stream=stream)

        # step 6: handle the response
        if response.status_code != 200:
            raise Exception(response.text)
        if 'text/event-stream' in response.headers.get('Content-Type', ''):
            print('Streaming response...')
            result = self.stream_generator(response)
        else:
            if 'application/json' in response.headers.get('Content-Type', ''):
                result = response.json()
            elif 'text/plain' in response.headers.get('Content-Type', ''):
                result = response.text
            else:
                result = response.content
                if response.status_code != 200:
                    raise Exception(result)
        return result

    
    def get_key(self,key=None):
        key = key or  self.key
        if isinstance(key, str):
            key = c.get_key(key)
        return key

    def get_url(self, url, mode='http'):
        """
        gets the url and makes sure its legit
        """
        if c.is_url(url):
            url = url
        elif c.is_int(url):
            url = f'0.0.0.0:{url}'
        if not hasattr(self, 'namespace'):
            self.namespace = c.namespace()
        url = self.namespace.get(str(url), url)
        if not url.startswith(mode):
            url = f'{mode}://{url}'
        return url

    call = forward

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

        
    def stream_generator(self, response):
        try:
            for chunk in response.iter_lines():
                yield self.process_stream_line(chunk)
        except Exception as e:
            yield c.detailed_error(e)

    def is_url(self,  url:str) -> bool:
        if not isinstance(url, str):
            return False
        if '://' in url:
            return True
        conds = []
        conds.append(isinstance(url, str))
        conds.append(':' in url)
        conds.append(c.is_int(url.split(':')[-1]))
        return all(conds)

    def client(self,  url:str = 'module', key:str = None, virtual:bool = True,  **client_kwargs):
        """
        Create a client instance.
        """
        client =  Client(url, key=key, **client_kwargs)
        return self.virtual_client(client) if virtual else client

    def virtual_client(self, client = None):
        client = client or self
        class ClientVirtual:
            def __init__(self, client):
                self._client = client
                for key in dir(client):
                    if key.startswith('_') or key in ['_client', '_remote_call']:
                        continue
                    if callable(getattr(client, key)):
                        setattr(self, key, getattr(client, key))
            def _remote_call(self, *args, remote_fn, timeout:int=10, key=None, **kwargs):
                return self._client.forward(fn=remote_fn, args=args, kwargs=kwargs, key=key, timeout=timeout)
            def __getattr__(self, key):
                if key in [ '_client', '_remote_call'] :
                    return getattr(self, key)
                else:
                    return lambda *args, **kwargs : self._remote_call(*args, remote_fn=key, **kwargs)
        return ClientVirtual(client)

    conn = connect = client # alias for client method
