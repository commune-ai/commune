

from typing import *
import asyncio
import json
import requests
import os
import commune as c

class Client(c.Module):

    def __init__( self, module : str = 'module', 
                key : Optional[str]= None ,
                network: Optional[bool] = 'local',
                mode: Optional[str] = 'http',
                serializer: Optional[c.Module] = 'serializer',
            **kwargs
        ):
        self.serializer = c.module(serializer)()
        self.key  = c.get_key(key, create_if_not_exists=True)
        self.set_address(module, network=network, mode=mode)

    def set_address(self, module, network='local', mode='http'):
        if  c.is_address(module):
            address = module
        else:
            namespace = c.namespace(network=network)
            if module in namespace:
                address = namespace[module]
            else:
                raise Exception(f'Module {module} not found in namespace {namespace}')
        prefix = f'{mode}://'
        self.network = network
        self.mode = mode
        self.address = prefix + address if not address.startswith(prefix) else address
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
            module, fn = module.split('/')
        else:
            module = self.module
        module_address = self.address
        ip = c.ip()
        if ip in module_address:
            module_address = module_address.replace(ip, '0.0.0.0')
        url = f"{module_address}/{fn}/"
        return url   


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
                data=None, 
                headers = None, 
                stream:bool = False):
                
        key = self.resolve_key(key)
        url = self.get_url(fn=fn, mode=mode)
        data = data or self.get_data(params=params, args=args, kwargs=kwargs, )
        headers = headers or self.get_header(data=data, key=key)
        try:             
            response = self.session.post(url, json=data, headers=headers, timeout=timeout, stream=stream)
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
        try:
            for chunk in response.iter_lines():
                line = self.process_stream_line(chunk)
                yield line
        except Exception as e:
            print(f'Error in stream: {e}')
            yield None

    def process_stream_line(self, line, stream_prefix = 'data: '):
        event_data = line.decode('utf-8')
        if event_data.startswith(stream_prefix):
            event_data = event_data[len(stream_prefix):] 
        if event_data == "": # skip empty lines if the event data is empty
            return ''
        if isinstance(event_data, str):
            if event_data.startswith('{') and event_data.endswith('}') and 'data' in event_data:
                event_data = json.loads(event_data)['data']
        return event_data
    
    @staticmethod
    def check_response(x) -> bool:
        if isinstance(x, dict) and 'error' in x:
            return False
        else:
            return True
        
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

    def get_header(self, data, key):
        headers = {
            'Content-Type': 'application/json',
            'key': key.ss58_address,
            'crypto_type': str(key.crypto_type),
            'time': str(c.time()),
        }
        headers['signature'] =  key.sign({'data': data, 'time': headers['time']}).hex()

        return headers 

    def forcurl(self, 
                fn: str = 'info', 
                args: list = None, 
                kwargs: dict = None, 
                timeout: int = 2,
                key: str = None,
                **extra_kwargs) -> str:
        # Resolve the key and URL
        key = self.resolve_key(key)
        url = self.get_url(fn=fn)
        
        # Prepare the data
        data = self.get_data(args=args or [], kwargs=kwargs or {}, **extra_kwargs)
        headers = self.get_header(data=data, key=key)
        # Prepare headers
        
        # Build curl command
        curl_cmd = ['curl', '-X POST']
        
        # Add headers
        for header_name, header_value in headers.items():
            curl_cmd.append(f"-H '{header_name}: {header_value}'")
        
        # Add data
        if isinstance(data, str):
            data_str = data
        else:
            data_str = json.dumps(data)
        curl_cmd.append(f"-d '{data_str}'")
        curl_cmd.append(f"'{url}'")
        curl_cmd.append(f'--max-time {timeout}')
        response = os.popen(' '.join(curl_cmd)).read()
        return response
        

    def __str__ ( self ):
        return "Client(address={})".format(self.address) 
    def __repr__ ( self ):
        return self.__str__()