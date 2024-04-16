

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

        

    async def async_forward(self,
        fn: str,
        args: list = None,
        kwargs: dict = None,
        params: dict = None,
        address : str = None,
        timeout: int = 10,
        headers : dict ={'Content-Type': 'application/json'},
        message_type = "v0",
        default_fn = 'info',
        verbose = False,
        debug = True,
        **extra_kwargs
        ):
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
        kwargs.update(extra_kwargs) 
        fn = fn or default_fn
        
        address = address or self.address
        args = args if args else []
        kwargs = kwargs if kwargs else {}
        
        input =  { 
                        "args": args,
                        "kwargs": kwargs,
                        "ip": c.ip(),
                        "timestamp": c.timestamp(),
                        }
        self.count += 1
        # serialize this into a json string
        if message_type == "v0":
            request = self.serializer.serialize(input)
            request = self.key.sign(request, return_json=True)
            # key emoji 
            
        elif message_type == "v1":
            input['ticket'] = self.key.ticket()
            request = self.serializer.serialize(input)
        else:
            raise ValueError(f"Invalid message_type: {message_type}")
        
        url = f"{address}/{fn}/"
        if not url.startswith('http'):
            url = 'http://' + url
        result = await self.process_request(url, request, headers=headers, timeout=timeout)

        c.print(f"🛰️ Call {self.address}/{fn} 🛰️  (🔑{self.key.ss58_address})", color='green', verbose=verbose)

        if self.save_history:
            input['fn'] = fn
            input['result'] = result
            input['module']  = self.address
            input['latency'] =  c.time() - input['timestamp']
            path = self.history_path+ '/' + self.key.ss58_address + '/' + self.address+ '/'+  str(input['timestamp'])
            self.put(path, input)
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

    async def process_request(self, url:str, request: dict, headers=None, timeout:int=10):
        # start a client session and send the request
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

    @classmethod
    def history(cls, key=None, history_path='history'):
        key = c.get_key(key)
        return cls.ls(history_path + '/' + key.ss58_address)
    
    def process_output(self, result):
        ## handles 
        if isinstance(result, str):
            result = json.loads(result)
        if 'data' in result:
            result = self.serializer.deserialize(result)
            return result['data']
        else:
            return result
        
    def forward(self,*args,return_future:bool=False, timeout:str=4, **kwargs):
        forward_future = asyncio.wait_for(self.async_forward(*args, **kwargs), timeout=timeout)
        if return_future:
            return forward_future
        else:
            return self.loop.run_until_complete(forward_future)
        
        
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
