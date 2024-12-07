

from typing import *
import asyncio
import json
import requests
import commune as c

class Client(c.Module):
    network2namespace = {}
    stream_prefix = 'data: '

    def __init__( 
            self,
            module : str = 'module',
            network: Optional[bool] = 'local',
            key : Optional[str]= None ,
            **kwargs
        ):
        self.serializer = c.module('serializer')()
        self.network = network
        self.loop =  c.get_event_loop()
        self.key  = c.get_key(key, create_if_not_exists=True)
        self.module = module
        self.address = self.resolve_module_address(module)
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
        return client.forward(fn=fn, args=args, kwargs=kwargs, timeout=timeout, **extra_kwargs)

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

    def __str__ ( self ):
        return "Client(address={})".format(self.address) 
    def __repr__ ( self ):
        return self.__str__()

    def __repr__(self) -> str:
        return super().__repr__()

    def resolve_module_address(self, module, mode='http'):
        if  c.is_address(module):
            url = module
        else:
            namespace = c.namespace(network=self.network)
            if module in namespace:
                url = namespace[module]
            else:
                raise Exception(f'Module {module} not found in namespace {namespace}')
        url = f'{mode}://' + url if not url.startswith(f'{mode}://') else url
        return url

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
         
    def request(self, url: str,
                 data: dict, 
                headers: dict, 
                timeout: int = 10, 
                stream: bool = True):
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

    def get_data(self, args=[], kwargs={}, **extra_kwargs):
        # derefernece
        args = c.copy(args or [])
        kwargs = c.copy(kwargs or {})
        if isinstance(args, dict):
            kwargs = {**kwargs, **args}
            args = []
        if extra_kwargs:
            kwargs = {**kwargs, **extra_kwargs}
        data =  {  "args": args, "kwargs": kwargs}
        data = self.serializer.serialize(data)
        return data

    def forward(self, 
                fn  = 'info', 
                args : str = [],
                kwargs : str = {},
                timeout:int=2, 
                key : str = None,
                mode: str  = 'http',
                headers = None,
                data = None,
                **extra_kwargs):
        key = self.resolve_key(key)
        url = self.get_url(fn=fn, mode=mode)
        data = data or self.get_data(args=args,  kwargs=kwargs,**extra_kwargs)
        headers = { 
                    'Content-Type': 'application/json', 
                    'key': key.ss58_address, 
                    'hash': c.hash(data),
                    'crypto_type': str(key.crypto_type),
                    'time': str(c.time())
                   }
                   
        headers['signature'] = key.sign({'data': headers['hash'], 'time': headers['time']}).hex()
        return self.request(url=url, 
                            data=data,
                            headers=headers, 
                            timeout=timeout)
    
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

    def process_stream_line(self, line):
        event_data = line.decode('utf-8')
        if event_data.startswith(self.stream_prefix):
            event_data = event_data[len(self.stream_prefix):] 
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
        protected_attributes = [ 'client', 'remote_call']
        
        def __init__(self, client: str ='ReactAgentModule'):
            if isinstance(client, str):
                client = c.connect(client)
            self.client = client
        
        def remote_call(self, *args, remote_fn, timeout:int=10, key=None, **kwargs):
            result =  self.client.forward(fn=remote_fn, args=args, kwargs=kwargs, timeout=timeout, key=key)
            return result

        def __str__(self):
            return str(self.client)

        def __repr__(self):
            return self.__str__()
            
        def __getattr__(self, key):

            if key in self.protected_attributes :
                return getattr(self, key)
            else:
                return lambda *args, **kwargs : self.remote_call(*args, remote_fn=key, **kwargs)
            

    def forcurl(self, 
                fn: str = 'info', 
                args: list = None, 
                kwargs: dict = None, 
                timeout: int = 2,
                key: str = None,
                **extra_kwargs) -> str:
        """
        Generate a cURL command for the equivalent HTTP request
        
        Args:
            fn (str): Function name to call
            args (list): Arguments list
            kwargs (dict): Keyword arguments
            timeout (int): Request timeout in seconds
            key (str): Key for authentication
            **extra_kwargs: Additional keyword arguments
        
        Returns:
            str: cURL command string
        """
        # Resolve the key and URL
        key = self.resolve_key(key)
        url = self.get_url(fn=fn)
        
        # Prepare the data
        data = self.get_data(args=args or [], kwargs=kwargs or {}, **extra_kwargs)
        
        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'key': key.ss58_address,
            'hash': c.hash(data),
            'crypto_type': str(key.crypto_type),
            'time': str(c.time())
        }
        
        # Add signature
        headers['signature'] = key.sign({
            'data': headers['hash'], 
            'time': headers['time']
        }).hex()
        
        # Build curl command
        curl_cmd = ['curl']
        
        # Add method
        curl_cmd.append('-X POST')
        
        # Add headers
        for header_name, header_value in headers.items():
            curl_cmd.append(f"-H '{header_name}: {header_value}'")
        
        # Add data
        if isinstance(data, str):
            data_str = data
        else:
            data_str = json.dumps(data)
        curl_cmd.append(f"-d '{data_str}'")
        
        # Add URL
        curl_cmd.append(f"'{url}'")
        
        # Add timeout
        curl_cmd.append(f'--max-time {timeout}')
        
        # now get the dict of the response and return it
        # make the request in the os and return the response
        import os
        response = os.popen(' '.join(curl_cmd)).read()


        return response
        