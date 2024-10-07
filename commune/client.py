

from typing import *
import asyncio
import commune as c
import json
import requests

# from .pool import ClientPool


class ClientVirtual:
    protected_attributes = [ 'client', 'remote_call']
    
    def __init__(self, client: str ='ReactAgentModule'):
        if isinstance(client, str):
            client = c.connect(client)
        self.client = client
    
    def remote_call(self, *args, remote_fn, return_future= False, timeout:int=10, key=None, **kwargs):
        result =  self.client.forward(fn=remote_fn, args=args, kwargs=kwargs, timeout=timeout, key=key, return_future=return_future)
        return result

    def __str__(self):
        return str(self.client)

    def __repr__(self):
        return self.__str__()
        
    def __getattr__(self, key):

        if key in self.protected_attributes :
            return getattr(self, key)
        else:
            return lambda *args, **kwargs : self.remote_call( remote_fn=key, *args, **kwargs)
        


class Client(c.Module):
    network2namespace = {}
    def __init__( 
            self,
            module : str = 'module',
            network: bool = 'local',
            key = None,
            stream_prefix = 'data: ',
            virtual = False,
            **kwargs
        ):
        self.serializer = c.module('serializer')()
        self.network = network
        self.loop =  c.get_event_loop()
        self.key  = c.get_key(key, create_if_not_exists=True)
        self.module = module
        self.stream_prefix = stream_prefix
        self.address = self.resolve_module_address(module, network=network)
        self.virtual = bool(virtual)
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
                    virtual=virtual,
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
                data = None,
                **extra_kwargs):
        network = network or self.network
        key = self.resolve_key(key)
        url = self.get_url(fn=fn, mode=mode,  network=network)
        data = data or self.get_data(args=args,  kwargs=kwargs, params=params, **extra_kwargs)
        headers = { 
                    'Content-Type': 'application/json', 
                    'key': key.ss58_address, 
                    'hash': c.hash(data),
                    'crypto_type': str(key.crypto_type),
                    'timestamp': str(c.timestamp())
                   }
        signature_data = {'data': headers['hash'], 'timestamp': headers['timestamp']}
        headers['signature'] = key.sign(signature_data).hex()
        result = self.request(url=url, 
                              data=data,
                              headers=headers,
                              timeout=timeout)
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

    def process_stream_line(self, line , stream_prefix=None):
        stream_prefix = stream_prefix or self.stream_prefix
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
        
    def get_curl(self, 
                        fn='info', 
                        params=None, 
                        args=None,
                        kwargs=None,
                        module=None,
                        key=None,
                        headers={'Content-Type': 'application/json'},
                        network=None,
                        version=1,
                        mode='http',
                        **extra_kwargs):
            key = self.resolve_key(key)
            network = network or self.network
            url = self.get_url(fn=fn, mode=mode, network=network)
            kwargs = {**(kwargs or {}), **extra_kwargs}
            input_data = self.get_params(args=args, kwargs=kwargs, params=params, version=version)
            headers_str = ' '.join([f'-H "{k}: {v}"' for k, v in headers.items()])
            data_str = json.dumps(input_data).replace('"', '\\"')
            # Construct the curl command
            curl_command = f'curl -X POST {headers_str} -d "{data_str}" "{url}"'

            return curl_command
    

    def run_curl(self, *args, **kwargs):
        curl_command = self.get_curl(*args, **kwargs)
        # get the output of the curl command
        import subprocess
        output = subprocess.check_output(curl_command, shell=True)
        return output.decode('utf-8')

