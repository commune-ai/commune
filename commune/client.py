

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
                serializer: Optional[c.Module] = 'serializer',
               **kwargs
        ):
        self.serializer = c.module(serializer)()
        self.key  = c.get_key(key, create_if_not_exists=True)
        if  c.is_address(module):
            address = module
        else:
            address = c.namespace(network=network).get(module)
        self.network = network
        self.address = address
        self.session = requests.Session()

    @classmethod
    def call(cls, 
                fn:str = 'info',
                *args,
                kwargs = None,
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
        client =  cls(module=module, network=network)
        kwargs = kwargs or {}
        kwargs = {**kwargs, **extra_kwargs}
        return client.forward(fn=fn, args=args, kwargs=kwargs, timeout=timeout, key=key)

    @classmethod
    def connect(cls,
                module:str = 'module', 
                network : str = 'local',
                virtual:bool = True, 
                **kwargs):
        client =  cls(module=module, network=network,**kwargs)
        if virtual:
            return Client.Virtual(client=client)
        else:
            return client
    
    def test(self, module='module::test_client'):
        c.serve(module)
        c.sleep(1)
        info = c.call(module+'/info')
        key  = c.get_key(module)
        assert info['key'] == key.ss58_address
        return {'info': info, 'key': str(key)}

    def get_url(self, fn, mode='http'):
        if '/' in str(fn):  
            address, fn = address.split('/')
        else:
            address = self.address
        print('address', address, self.address)
        address = address if address.startswith(mode) else f'{mode}://{address}'
        return f"{address}/{fn}/"

    def get_data(self, args=[], kwargs={}, params = None):
        # derefernece
        args = c.copy(args or [])
        kwargs = c.copy(kwargs or {})
        if isinstance(args, dict):
            kwargs = {**kwargs, **args}
            args = []
        if params:
            if isinstance(params, dict):
                kwargs = {**kwargs, **params}
            elif isinstance(params, list):
                args = params
            else:
                raise Exception(f'Invalid params {params}')
        data =  {  "args": args, "kwargs": kwargs}
        data = self.serializer.serialize(data)
        return data

    def forward(self, 
                fn  = 'info', 
                params: Optional[Union[list, dict]] = None,
                args : Optional[list] = [], 
                kwargs : Optional[dict] = {},  
                timeout:int=2,  
                key : str = None,  
                mode: str  = 'http', 
                headers = None, 
                stream:bool = False):
                
        key = self.resolve_key(key)
        url = self.get_url(fn=fn, mode=mode)
        data = self.get_data(params=params, args=args, kwargs=kwargs )
        headers = headers or self.get_header(data=data, key=key)
        try: 
            response = self.session.post(url, json=data, headers=headers, timeout=timeout, stream=stream)
            result = self.process_response(response)
        except Exception as e:
            result = c.detailed_error(e)
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
            result = response.content
            if response.status_code != 200:
                raise Exception(result)
        result = self.serializer.deserialize(result)
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

    def get_header(self, data, key: 'Key'):
        time_str = str(c.time())
        return {
            'key': key.ss58_address,
            'crypto_type': str(key.crypto_type),
            'time': time_str,
            'Content-Type': 'application/json',
            'signature':  key.sign({'data': data, 'time': time_str}).hex()
        } 