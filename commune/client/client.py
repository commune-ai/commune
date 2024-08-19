

from typing import *
import asyncio
import commune as c
import aiohttp
import json
from .pool import ClientPool
from .virtual import VirtualClient
# from .pool import ClientPool

class Client(c.Module, ClientPool):
    network2namespace = {}
    def __init__( 
            self,
            module : str = 'module',
            network: bool = 'local',
            key = None,
            **kwargs
        ):
        self.serializer = c.module('serializer')()
        self.network = network
        self.loop = asyncio.get_event_loop()
        self.key  = c.get_key(key, create_if_not_exists=True)
        self.module = module
        self.address = self.resolve_module_address(module)

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
                trials = 10, 
                timeout=40,
                **extra_kwargs) -> None:
        
        if '/' in str(fn):
            module = '.'.join(fn.split('/')[:-1])
            fn = fn.split('/')[-1]
        else:
            module = fn
            fn = 'info'
        for i in range(trials):
            try:
                client = cls.connect(module, virtual=False, network=network)
                response =  client.forward(fn=fn, 
                                        args=args,
                                        kwargs=kwargs, 
                                        params=params,
                                        key=key  ,
                                        timeout=timeout, 
                                        **extra_kwargs)
            except Exception as e:
                response = c.detailed_error(e)
            return response

    @classmethod
    def connect(cls,
                module:str = 'module', 
                network : str = 'local',
                virtual:bool = True, 
                **kwargs):
        
        client = cls(module=module, 
                    virtual=virtual, 
                    network=network,
                    **kwargs)
        # if virtual turn client into a virtual client, making it act like if the server was local
        if virtual:
            return client.virtual()
        
        return client
    
    def test(self, module='module::test_client'):
        c.serve(module)
        c.sleep(1)

        info = c.call(module+'/info')
        key  = c.get_key(module)
        assert info['key'] == key.ss58_address
        return {'info': info, 'key': str(key)}

    def __str__ ( self ):
        return "Client({})".format(self.address) 
    def __repr__ ( self ):
        return self.__str__()


    def virtual(self):
        return VirtualClient(module = self)
    
    def __repr__(self) -> str:
        return super().__repr__()

    def resolve_module_address(self, module, mode='http'):
        if not c.is_address(module):
            namespace = self.resolve_namespace(self.network)
            url = namespace[module]
        else:
            url = module
        url = f'{mode}://' + url if not url.startswith(f'{mode}://') else url

        return url

    def get_url(self, fn, mode='http', network=None):
        network = network or self.network
        if '://' in str(fn):
            mode ,fn = fn.split('://')
        if '/' in str(fn):  
            module, fn = module.split('/')
        else:
            module = self.module
        module_address = self.resolve_module_address(module, mode=mode)
        url = f"{module_address}/{fn}/"
        return url        


    async def async_request(self, url : str, data:dict, headers:dict, timeout:int=10):
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:   
                try:             
                    if response.content_type == 'text/event-stream':
                        return self.iter_over_async(self.stream_generator(response))
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

    def forward(self, fn  = 'info', 
                            params = None, 
                            args=None,
                            kwargs = None,
                            timeout:int=10, 
                            key = None,
                            verbose=False, 
                            network = None,
                            mode = 'http',
                            return_future = False,
                            **extra_kwargs):
        
        # step 1: preparing data
        kwargs = kwargs or {}
        args = args or []
        network = network or self.network
        if isinstance(params, dict):
            kwargs = {**kwargs, **params}
        kwargs.update(extra_kwargs)
        if isinstance(params, list):
            args = args + params
        data =  { 
                    "args": args or [],
                    "kwargs": params or kwargs or {},
                    }
        data = self.serializer.serialize(data)

        # step 2: preparing headers
        key = self.resolve_key(key)
        headers = {'Content-Type': 'application/json', 
                    'key': key.ss58_address, 
                    'hash': c.hash(data),
                    'crypto_type': str(key.crypto_type),
                    'timestamp': str(c.timestamp())
                   }
        signature_data = {'data': data, 'timestamp': headers['timestamp']}
        headers['signature'] = key.sign(signature_data).hex()
        url = self.get_url(fn=fn,mode=mode,  network=network)
        kwargs = {**(kwargs or {}), **extra_kwargs}
        result = self.async_request( url=url,data=data,headers= headers,timeout= timeout)
        if  return_future:
            return result
        return self.loop.run_until_complete(result)



    
    def __del__(self):
        try:
            if hasattr(self, 'session'):
                asyncio.run(self.session.close())
        except:
            pass

    def iter_over_async(self, ait):
        # helper async fn that just gets the next element
        # from the async iterator
        def get_next():
            try:
                obj = self.loop.run_until_complete(ait.__anext__())
                return obj
            except StopAsyncIteration:
                return 'done'
        # actual sync iterator (implemented using a generator)
        while True:
            obj = get_next() 
            if obj == 'done':
                break
            yield obj

    async def stream_generator(self, response):
        async for line in response.content:
            event =  self.process_stream_line(line)
            if event == '':
                continue
            yield event
        
        


    def resolve_key(self,key=None):
        if key == None:
            key = self.key
        if isinstance(key, str):
            key = c.get_key(key)
        return key
    
    def process_stream_line(self, line ):
        STREAM_PREFIX = 'data: '
        event_data = line.decode('utf-8')
        event_data = event_data[len(STREAM_PREFIX):] if event_data.startswith(STREAM_PREFIX) else event_data
        event_data = event_data.strip() # remove leading and trailing whitespaces
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
                        timeout=10, 
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

            # Convert the headers to curl format
            headers_str = ' '.join([f'-H "{k}: {v}"' for k, v in headers.items()])

            # Convert the input data to JSON string
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
