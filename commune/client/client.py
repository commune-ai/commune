

from typing import *
import asyncio
import commune as c
import aiohttp
import json

STREAM_PREFIX = 'data: '
BYTES_PER_MB = 1e6

class Client(c.Module):
    count = 0
    def __init__( 
            self,
            address : str = '0.0.0.0:8000',
            network: bool = 'local',
            key : str = None,
            save_history: bool = True,
            history_path : str = 'history',
            loop: 'asyncio.EventLoop' = None, 
            debug: bool = False,
            serializer= 'serializer',
            default_fn = 'info',

            **kwargs
        ):
        self.loop = c.get_event_loop() if loop == None else loop

        self.set_client(address = address, network=network)
        self.serializer = c.module(serializer)()
        self.key = c.get_key(key)
        self.start_timestamp = c.timestamp()
        self.save_history = save_history
        self.history_path = history_path
        self.debug = debug
        self.default_fn = default_fn

    def prepare_request(self, args: list = None, kwargs: dict = None, params=None, message_type = "v0"):

        if isinstance(args, dict):
            kwargs = args
            args = None

        if params != None:
            assert type(params) in [list, dict], f'params must be a list or dict, not {type(params)}'
            if isinstance(params, list):
                args = params
            elif isinstance(params, dict):
                kwargs = params  
        kwargs = kwargs or {}
        args = args if args else []
        kwargs = kwargs if kwargs else {}
        
        # serialize this into a json string
        if message_type == "v0":
            """
            {
                'data' : {
                'args': args,
                'kwargs': kwargs,
                'timestamp': timestamp,
                }
                'signature': signature
            }
            
            """

            input =  { 
                        "args": args,
                        "kwargs": kwargs,
                        "timestamp": c.timestamp(),
                        }
            request = self.serializer.serialize(input)
            request = self.key.sign(request, return_json=True)
            # key emoji 
        elif message_type == "v1":

            inputs = {'params': kwargs,
                      'ticket': self.key.ticket() }
            if len(args) > 0:
                inputs['args'] = args
            request = self.serializer.serialize(input)
        else:
            raise ValueError(f"Invalid message_type: {message_type}")
    
        return request
    
    
    async def send_request(self, url:str, request: dict, headers=None, timeout:int=10, verbose=False):
        # start a client session and send the request

        if not url.startswith('http'):
            url = 'http://' + url
        
        c.print(f"🛰️ Call {url} 🛰️  (🔑{self.key.ss58_address})", color='green', verbose=verbose)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request, headers=headers) as response:
                
                if response.content_type == 'application/json':
                    result = await asyncio.wait_for(response.json(), timeout=timeout)
        
                elif response.content_type == 'text/plain':
                    result = await asyncio.wait_for(response.text(), timeout=timeout)
                
                elif response.content_type == 'text/event-stream':
                    if self.debug:
                        progress_bar = c.tqdm(desc='MB per Second', position=0)
                    result = {}
                    async for line in response.content:

                        event_data = line.decode('utf-8')
                        event_bytes  = len(event_data)
                        
                        if self.debug :
                            progress_bar.update(event_bytes/(BYTES_PER_MB))
                        
                        # remove the "data: " prefix
                        if event_data.startswith(STREAM_PREFIX):
                            event_data = event_data[len(STREAM_PREFIX):]

                        event_data = event_data.strip()
                        
                        # skip empty lines
                        if event_data == "":
                            continue

                        # if the data is formatted as a json string, load it {data: ...}
                        if isinstance(event_data, bytes):
                            event_data = event_data.decode('utf-8')

                        # if the data is formatted as a json string, load it {data: ...}
                        if isinstance(event_data, str):
                            if event_data.startswith('{') and event_data.endswith('}') and 'data' in event_data:
                                event_data = json.loads(event_data)['data']
                            result += [event_data]
                        
                    # process the result if its a json string
                    if result.startswith('{') and result.endswith('}') or \
                        result.startswith('[') and result.endswith(']'):
                        result = ''.join(result)
                        result = json.loads(result)
                else:
                    raise ValueError(f"Invalid response content type: {response.content_type}")
        if type(result) in [str, dict]:
            result = self.serializer.deserialize(result)
        if isinstance(result, dict) and 'data' in result:
            result = result['data']

        return result


    def process_output(self, result):
        ## handles 
        if isinstance(result, str):
            result = json.loads(result)
        if 'data' in result:
            result = self.serializer.deserialize(result)
            return result['data']
        else:
            return result


    def resolve_key(self,key=None):
        if key == None:
            key = self.key
        if isinstance(key, str):
            key = c.get_key(key)
        return key
    
    def prepare_url(self, address, fn):
        address = address or self.address
        fn = fn or self.default_fn
        if '/' in address.split('://')[-1]:
            address = address.split('://')[-1]
        url = f"{address}/{fn}/"
        return url

    async def async_forward(self,
        fn: str,
        args: list = None,
        kwargs: dict = None,
        params: dict = None,
        address : str = None,
        timeout: int = 10,
        headers : dict ={'Content-Type': 'application/json'},
        message_type = "v0",
        key : str = None,
        verbose = False,
        **extra_kwargs
        ):
        key = self.resolve_key(key)
        url = self.prepare_url(address, fn)
        # resolve the kwargs at least
        kwargs =kwargs or {}
        kwargs.update(extra_kwargs)
        request = self.prepare_request(args=args, kwargs=kwargs, params=params, message_type=message_type)
        result = await self.send_request(url=url, request=request, headers=headers, timeout=timeout, verbose=verbose)
        
        if self.save_history:
            input = self.serializer.deserialize(request)
            path =  self.history_path+ '/' + self.key.ss58_address + '/' + self.address+ '/'+  str(input['timestamp'])
            output = {
                'address': address,
                'fn': fn,
                'input': input,
                'result': result,
                'latency': c.time() - input['timestamp'],
            }
            self.put(path, output)
        return result
    
    
    def age(self):
        return  self.start_timestamp - c.timestamp()

    def set_client(self,
            address : str = None,
            verbose: bool = 1,
            network : str = 'local',
            possible_modes = ['http', 'https'],
            ):
        # we dont want to load the namespace if we have the address
        if not c.is_address(address):
            module = address # we assume its a module name
            assert module != None, 'module must be provided'
            namespace = c.get_namespace(search=module, network=network)
            if module in namespace:
                address = namespace[module]
            else:    
                address = module
        if '://' in address:
            mode = address.split('://')[0]
            assert mode in possible_modes, f'Invalid mode {mode}'
            address = address.split('://')[-1]
        address = address.replace(c.ip(), '0.0.0.0')
        self.address = address
        return {'address': self.address}

    @classmethod
    def history(cls, key=None, history_path='history'):
        key = c.get_key(key)
        return cls.ls(history_path + '/' + key.ss58_address)
    
        
    def forward(self,*args,return_future:bool=False, timeout:str=4, **kwargs):
        forward_future = asyncio.wait_for(self.async_forward(*args, **kwargs), timeout=timeout)
        if return_future:
            return forward_future
        else:
            return self.loop.run_until_complete(forward_future)
        


    
    
    @classmethod
    def call_search(cls, 
                    search : str, 
                *args,
                timeout : int = 10,
                network:str = 'local',
                key:str = None,
                kwargs = None,
                return_future:bool = False,
                **extra_kwargs) -> None:
        if '/' in search:
            search, fn = search.split('/')
        namespace = c.namespace(search=search, network=network)
        future2module = {}
        for module, address in namespace.items():
            future = c.submit(c.call,
                                args = list(args),
                      
                               kwargs = { 'module': module, 'fn': fn, 'timeout': timeout, 
                                         'network': network, 'key': key, 
                                         'kwargs': kwargs,
                                         **extra_kwargs} )
            future2module[future] = module
        futures = list(future2module.keys())
        result = {}
        progress_bar = c.tqdm(len(futures))
        for future in c.as_completed(futures):
            module = future2module.pop(future)
            futures.remove(future)
            progress_bar.update(1)
            result[module] = future.result()

        return result
            

        
    __call__ = forward

    def __str__ ( self ):
        return "Client({})".format(self.address) 
    def __repr__ ( self ):
        return self.__str__()
    def __exit__ ( self ):
        self.__del__()

    def virtual(self):
        from .virtual import VirtualClient
        return VirtualClient(module = self)
    
    def __repr__(self) -> str:
        return super().__repr__()


    @classmethod
    def connect(cls,
                module:str, 
                network : str = 'local',
                mode = 'http',
                virtual:bool = True, 
                **kwargs):
        
        
        
        client = cls(address=module, 
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
        c.print(c.server_exists(module))
        c.print('Module started')

        c.print(c.call(module+'/info'))
